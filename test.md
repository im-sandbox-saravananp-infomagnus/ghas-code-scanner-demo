#!/usr/bin/env python3
"""
SVN Folder Inventory (folder-scoped, history-aware, resumable)
---------------------------------------------------------------
Collects inventory for one or more folders inside an SVN repository, e.g.
https://svn.example.com/repos/myrepo/trunk/FolderX, WITHOUT scanning every
repository revision:

  1. Revisions that touched the folder (follows renames/copies by default)
  2. First/last revision and date, number of authors
  3. Merge commits (message heuristic) and directory property-change commits
  4. Size, file count and directory count at the pinned HEAD revision
  5. Files >= threshold (default 100 MiB) at HEAD
  6. Files >= threshold in ANY revision of the folder's history, including
     deleted, shrunk and renamed files

How it stays fast: a single 'svn log -v' call returns every changed path of
the folder. Only file versions that were added/modified/replaced, and
directories copied into the folder, are size-checked - many per 'svn list'
call, several calls in parallel. The cost depends on the folder's own changes,
not on the total number of repository revisions.

Progress is saved under --state-dir; re-running the same command resumes.

Tip: on the SVN server itself, pass file:///path/to/repo/trunk/FolderX as the
folder URL for maximum speed (no network, no authentication).

Requirements: Python 3.7+ and the 'svn' command-line client on PATH.

Usage:
  python3 svn_folder_inventory.py <folder_url> [<folder_url> ...] [options]

Examples:
  python3 svn_folder_inventory.py https://svn.example.com/repos/myrepo/trunk/FolderA \\
      https://svn.example.com/repos/myrepo/trunk/FolderB --username alice
  python3 svn_folder_inventory.py --urls-file folders.txt --jobs 8 \\
      --output-prefix customer --log-file customer.log
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from urllib.parse import quote, unquote

MIB = 1024 * 1024
MERGE_RE = re.compile(r"\bmerge[d]?\b", re.IGNORECASE)
# Keep each 'svn list' command line well below the Windows 32K limit.
MAX_BATCH_CHARS = 24000

logger = logging.getLogger("svn_folder_inventory")


class SvnError(RuntimeError):
    """An SVN command failed."""


# ─────────────────────────────────────────────────────────────────
# SVN command runner
# ─────────────────────────────────────────────────────────────────

class Svn:
    def __init__(self, username: str | None, password: str | None, retries: int):
        self.auth: list[str] = ["--non-interactive"]
        if username:
            self.auth += ["--username", username]
        if password:
            self.auth += ["--password", password, "--no-auth-cache"]
        self.password = password
        self.retries = max(1, retries)

    def _redact(self, text: str) -> str:
        return text.replace(self.password, "[REDACTED]") if self.password else text

    def run(
        self,
        args: list[str],
        out_path: str | None = None,
        attempts: int = 1,
        allow_fail: bool = False,
    ) -> bytes | None:
        """Run 'svn <args>'; return stdout bytes (b"" when written to out_path)."""
        cmd = ["svn", *args, *self.auth]
        error = ""
        for attempt in range(1, attempts + 1):
            try:
                if out_path:
                    with open(out_path, "wb") as handle:
                        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.PIPE)
                else:
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError:
                raise SvnError("'svn' command not found. Install the SVN client and add it to PATH.") from None
            if proc.returncode == 0:
                return b"" if out_path else proc.stdout
            error = self._redact(" | ".join(
                line.strip() for line in proc.stderr.decode("utf-8", "replace").splitlines() if line.strip()
            ))
            if attempt < attempts:
                logger.warning("svn %s failed (attempt %d/%d), retrying: %s", args[0], attempt, attempts, error)
                time.sleep(min(60, 5 * attempt))
        if allow_fail:
            return None
        raise SvnError(f"svn {args[0]} failed: {error}")


# ─────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────

@dataclass
class Target:
    """A path version to size-check: a file version or a directory copied into the folder."""
    path: str      # repository-absolute path, e.g. /trunk/FolderX/a.bin
    rev: int
    kind: str      # "file", "dir", or "" when the server did not report it
    scope: str     # folder path at that revision (differs before a rename)
    copied: bool = False

    @property
    def key(self) -> str:
        return f"{self.path}@{self.rev}"


@dataclass
class LargeRecord:
    path: str      # relative to the folder
    max_size: int
    peak_revision: int
    first_large_revision: int
    last_large_revision: int
    large_revision_count: int = 1
    head_size: int | None = None
    threshold: int = 100 * MIB

    @property
    def status(self) -> str:
        if self.head_size is None:
            return "Absent at HEAD (deleted or moved)"
        return "Large at HEAD" if self.head_size >= self.threshold else "Below threshold at HEAD"


@dataclass
class FolderResult:
    url: str
    scope: str = ""
    oldest_scope: str = ""
    head_revision: int = 0
    revisions: int = 0
    first_revision: int | None = None
    first_date: str = ""
    last_revision: int | None = None
    last_date: str = ""
    authors: set = field(default_factory=set)
    merge_msg_commits: int = 0
    dir_prop_commits: int = 0
    versions_checked: int = 0
    failures: list = field(default_factory=list)
    head_files: int = 0
    head_dirs: int = 0
    head_size: int = 0
    head_large: list = field(default_factory=list)
    history_large: list = field(default_factory=list)
    ext_sizes: dict = field(default_factory=dict)
    error: str = ""


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:,.2f} {unit}"
        value /= 1024.0
    return f"{value:,.2f} PB"


def path_url(root_url: str, path: str, rev: int) -> str:
    return f"{root_url.rstrip('/')}{quote(path, safe='/')}@{rev}"


def under(path: str, scope: str) -> bool:
    return path.startswith(scope + "/")


def get_info(svn: Svn, url: str) -> dict:
    root = ET.fromstring(svn.run(["info", "--xml", url], attempts=svn.retries))
    entry = root.find("entry")
    if entry is None:
        raise SvnError(f"No 'svn info' entry for {url}")
    relative = entry.findtext("relative-url", "")
    if not relative.startswith("^/"):
        raise SvnError("SVN client too old: 'relative-url' missing from 'svn info' (need SVN 1.8+).")
    return {
        "revision": int(entry.get("revision", "0")),
        "kind": entry.get("kind", ""),
        "root": entry.findtext("repository/root", ""),
        "scope": "/" + unquote(relative[2:]).strip("/"),
    }


def state_dir_for(base: str, url: str, stop_on_copy: bool, threshold: int) -> str:
    digest = hashlib.sha1(f"{url}|{stop_on_copy}|{threshold}".encode()).hexdigest()[:10]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", unquote(url.rstrip("/")).split("/")[-1]) or "root"
    return os.path.join(base, f"{name}_{digest}")


# ─────────────────────────────────────────────────────────────────
# Step 1 - folder history (single 'svn log -v' call)
# ─────────────────────────────────────────────────────────────────

def parse_history(log_path: str, scope: str, stop_on_copy: bool, result: FolderResult) -> list[Target]:
    """Walk the log newest -> oldest, tracking the folder's path across renames/copies."""
    targets: dict[str, Target] = {}

    def add(target: Target) -> None:
        targets.setdefault(target.key, target)

    for _event, elem in ET.iterparse(log_path, events=("end",)):
        if elem.tag != "logentry":
            continue
        rev = int(elem.get("revision", "0"))
        date = elem.findtext("date", "")
        result.revisions += 1
        result.authors.add(elem.findtext("author", ""))
        if result.last_revision is None or rev > result.last_revision:
            result.last_revision, result.last_date = rev, date
        if result.first_revision is None or rev < result.first_revision:
            result.first_revision, result.first_date = rev, date
        if MERGE_RE.search(elem.findtext("msg", "") or ""):
            result.merge_msg_commits += 1

        next_scope = scope
        dir_prop = False
        for node in elem.iterfind("paths/path"):
            path = (node.text or "").rstrip("/")
            action = node.get("action", "")
            kind = node.get("kind", "")
            copyfrom = node.get("copyfrom-path")
            if path == scope or under(scope, path):
                # The folder itself or one of its parents.
                if path == scope and node.get("prop-mods") == "true":
                    dir_prop = True
                if action in ("A", "R") and copyfrom:
                    if stop_on_copy:
                        add(Target(scope, rev, "dir", scope, copied=True))
                    else:
                        next_scope = copyfrom.rstrip("/") + scope[len(path):]
                continue
            if not under(path, scope):
                continue
            if kind == "dir" and node.get("prop-mods") == "true":
                dir_prop = True
            if action not in ("A", "M", "R"):
                continue
            if kind == "file":
                if action == "M" and node.get("text-mods") == "false":
                    continue  # property-only change; size unchanged
                add(Target(path, rev, "file", scope, copied=bool(copyfrom)))
            elif kind == "dir":
                if copyfrom:
                    add(Target(path, rev, "dir", scope, copied=True))
            else:
                add(Target(path, rev, "", scope, copied=bool(copyfrom)))
        if dir_prop:
            result.dir_prop_commits += 1
        if next_scope != scope:
            logger.info("  r%d: folder was copied/renamed from %s", rev, next_scope)
            scope = next_scope
        elem.clear()

    result.oldest_scope = scope
    return list(targets.values())


def resolve_unknown_kinds(svn: Svn, root_url: str, targets: list[Target]) -> list[Target]:
    """Older servers omit 'kind' in the log; ask 'svn info' for those paths."""
    unknown = [t for t in targets if not t.kind]
    if not unknown:
        return targets
    logger.info("  Resolving node kind for %d paths (server did not report it)", len(unknown))
    for target in unknown:
        out = svn.run(["info", "--xml", path_url(root_url, target.path, target.rev)],
                      attempts=svn.retries, allow_fail=True)
        entry = ET.fromstring(out).find("entry") if out else None
        target.kind = entry.get("kind", "") if entry is not None else "file"
    return [t for t in targets if t.kind == "file" or t.copied]


# ─────────────────────────────────────────────────────────────────
# Step 2 - size checks (batched, parallel, resumable)
# ─────────────────────────────────────────────────────────────────

def make_batches(targets: list[Target], root_url: str, batch_size: int) -> list[list[Target]]:
    batches: list[list[Target]] = []
    current: list[Target] = []
    chars = 0
    for target in targets:
        length = len(path_url(root_url, target.path, target.rev)) + 3
        if current and (len(current) >= batch_size or chars + length > MAX_BATCH_CHARS):
            batches.append(current)
            current, chars = [], 0
        current.append(target)
        chars += length
    if current:
        batches.append(current)
    return batches


def collect_large(target: Target, listing: ET.Element, threshold: int) -> list[list]:
    large = []
    for entry in listing.iterfind("entry"):
        if entry.get("kind") != "file":
            continue
        size = int(entry.findtext("size", "0"))
        if size < threshold:
            continue
        full = target.path if target.kind == "file" else f"{target.path}/{entry.findtext('name', '')}"
        large.append([full[len(target.scope) + 1:] if under(full, target.scope) else full, size])
    return large


def check_batch(svn: Svn, root_url: str, batch: list[Target], threshold: int) -> list[tuple]:
    """Return [(target, large_files | None, error | None)] for a batch of targets."""
    urls = [path_url(root_url, t.path, t.rev) for t in batch]
    attempts = svn.retries if len(batch) == 1 else 1
    out = svn.run(["list", "--xml", "--depth", "infinity", *urls], attempts=attempts, allow_fail=True)
    if out is not None:
        try:
            lists = ET.fromstring(out).findall("list")
        except ET.ParseError:
            lists = []
        if len(lists) == len(batch):
            return [(t, collect_large(t, lst, threshold), None) for t, lst in zip(batch, lists)]
    if len(batch) == 1:
        return [(batch[0], None, "svn list failed")]
    # One failing target fails the whole call; retry each target separately.
    results = []
    for target in batch:
        results += check_batch(svn, root_url, [target], threshold)
    return results


def run_size_checks(svn: Svn, root_url: str, targets: list[Target], cache_path: str,
                    threshold: int, jobs: int, batch_size: int, result: FolderResult) -> dict:
    done: dict[str, list] = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial last line after an interruption
                done[item["k"]] = item["large"]
    pending = [t for t in targets if t.key not in done]
    total = len(targets)
    logger.info("  Size checks: %d file/dir versions (%d already done, %d pending)",
                total, total - len(pending), len(pending))
    batches = make_batches(pending, root_url, batch_size)
    last_report = 0.0
    with open(cache_path, "a", encoding="utf-8") as cache, ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(check_batch, svn, root_url, b, threshold) for b in batches]
        for future in as_completed(futures):
            for target, large, error in future.result():
                if error:
                    result.failures.append(target.key)
                    logger.warning("  Could not size-check %s: %s", target.key, error)
                    continue
                done[target.key] = large
                cache.write(json.dumps({"k": target.key, "large": large}) + "\n")
            cache.flush()
            now = time.time()
            if now - last_report >= 15 or len(done) + len(result.failures) == total:
                last_report = now
                finished = len(done) + len(result.failures)
                logger.info("  Size checks: %d / %d (%.1f%%)", finished, total, 100.0 * finished / max(total, 1))
    result.versions_checked = len(done)
    return done


# ─────────────────────────────────────────────────────────────────
# Per-folder driver
# ─────────────────────────────────────────────────────────────────

def inventory_folder(svn: Svn, url: str, args: argparse.Namespace, threshold: int) -> FolderResult:
    result = FolderResult(url=url)
    info = get_info(svn, url)
    if info["kind"] != "dir":
        raise SvnError(f"{url} is not a directory.")
    root_url, result.scope = info["root"], info["scope"]

    state = state_dir_for(args.state_dir, url, args.stop_on_copy, threshold)
    if args.fresh and os.path.isdir(state):
        shutil.rmtree(state)
    os.makedirs(state, exist_ok=True)
    meta_path = os.path.join(state, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            pinned = json.load(handle)["head"]
        logger.info("  Resuming previous run (state: %s), pinned HEAD r%d", state, pinned)
    else:
        pinned = info["revision"]
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump({"url": url, "head": pinned, "scope": result.scope}, handle)
    result.head_revision = pinned
    pinned_url = f"{url}@{pinned}"

    # Step 1: history
    log_path = os.path.join(state, "log.xml")
    if not os.path.exists(log_path):
        logger.info("  [1/3] Fetching folder history (one 'svn log -v' call; can take a while) ...")
        log_cmd = ["log", "-v", "--xml", "-r", f"{pinned}:0", pinned_url]
        if args.stop_on_copy:
            log_cmd.insert(1, "--stop-on-copy")
        svn.run(log_cmd, out_path=log_path + ".part", attempts=svn.retries)
        os.replace(log_path + ".part", log_path)
    else:
        logger.info("  [1/3] Using cached folder history")
    targets = parse_history(log_path, result.scope, args.stop_on_copy, result)
    targets = resolve_unknown_kinds(svn, root_url, targets)
    logger.info("  Revisions: %d, file/dir versions to check: %d", result.revisions, len(targets))

    # Step 2: historical sizes
    logger.info("  [2/3] Checking sizes of every changed file version ...")
    done = run_size_checks(svn, root_url, targets, os.path.join(state, "sizes.jsonl"),
                           threshold, args.jobs, args.batch_size, result)
    records: dict[str, LargeRecord] = {}

    def record(rel: str, size: int, rev: int) -> None:
        rec = records.get(rel)
        if rec is None:
            records[rel] = LargeRecord(rel, size, rev, rev, rev, threshold=threshold)
            return
        rec.first_large_revision = min(rec.first_large_revision, rev)
        rec.last_large_revision = max(rec.last_large_revision, rev)
        rec.large_revision_count += 1
        if size > rec.max_size:
            rec.max_size, rec.peak_revision = size, rev

    for target in targets:
        for rel, size in done.get(target.key) or []:
            record(rel, size, target.rev)

    # Step 3: HEAD snapshot
    head_path = os.path.join(state, "head_list.xml")
    if not os.path.exists(head_path):
        logger.info("  [3/3] Listing folder at r%d ...", pinned)
        svn.run(["list", "--xml", "--depth", "infinity", pinned_url],
                out_path=head_path + ".part", attempts=svn.retries)
        os.replace(head_path + ".part", head_path)
    else:
        logger.info("  [3/3] Using cached HEAD listing")
    for _event, elem in ET.iterparse(head_path, events=("end",)):
        if elem.tag != "entry":
            continue
        if elem.get("kind") == "dir":
            result.head_dirs += 1
        elif elem.get("kind") == "file":
            name = elem.findtext("name", "")
            size = int(elem.findtext("size", "0"))
            result.head_files += 1
            result.head_size += size
            ext = os.path.splitext(name)[1].lower() or "(none)"
            count, total = result.ext_sizes.get(ext, (0, 0))
            result.ext_sizes[ext] = (count + 1, total + size)
            if size >= threshold:
                result.head_large.append((name, size))
                if name not in records:
                    commit = elem.find("commit")
                    record(name, size, int(commit.get("revision", pinned)) if commit is not None else pinned)
            if name in records:
                records[name].head_size = size
        elem.clear()

    result.head_large.sort(key=lambda item: (-item[1], item[0]))
    result.history_large = sorted(records.values(), key=lambda r: (-r.max_size, r.path))
    return result


# ─────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────

def print_report(result: FolderResult, threshold: int) -> None:
    sep = "=" * 72
    label = format_size(threshold)
    print(sep)
    print(f"  Folder : {result.url}")
    if result.error:
        print(f"  FAILED : {result.error}")
        print(sep)
        return
    print(f"  Path   : {result.scope}")
    if result.oldest_scope and result.oldest_scope != result.scope:
        print(f"  History followed back to: {result.oldest_scope}")
    print(sep)
    print(f"  Pinned HEAD                 : r{result.head_revision}")
    print(f"  Revisions touching folder   : {result.revisions:,}"
          + (f"   (r{result.first_revision} .. r{result.last_revision})" if result.revisions else ""))
    print(f"  First commit                : {result.first_date}")
    print(f"  Last commit                 : {result.last_date}")
    print(f"  Authors                     : {len(result.authors):,}")
    print(f"  Merge commits (by message)  : {result.merge_msg_commits:,}")
    print(f"  Dir property-change commits : {result.dir_prop_commits:,}   (possible svn:mergeinfo merges)")
    print(f"  File versions size-checked  : {result.versions_checked:,}   (failed: {len(result.failures):,})")
    print(f"  Size at HEAD                : {format_size(result.head_size)}  ({result.head_size:,} bytes)")
    print(f"  Files / dirs at HEAD        : {result.head_files:,} / {result.head_dirs:,}")
    print(f"  Files >= {label} at HEAD : {len(result.head_large):,}")
    print(f"  Paths ever >= {label}    : {len(result.history_large):,}")
    for rec in result.history_large:
        head = "absent" if rec.head_size is None else format_size(rec.head_size)
        print(f"    {format_size(rec.max_size):>12}  r{rec.peak_revision:<8}  HEAD: {head:<12}  {rec.path}")
    if result.ext_sizes:
        print("  Top extensions at HEAD by size:")
        top = sorted(result.ext_sizes.items(), key=lambda item: -item[1][1])[:10]
        for ext, (count, total) in top:
            print(f"    {ext:<12} {count:>9,} files  {format_size(total):>14}")
    if result.failures:
        print(f"  WARNING: {len(result.failures)} versions could not be checked; re-run to retry them.")
    print()


def write_csvs(results: list[FolderResult], prefix: str, threshold: int) -> None:
    summary_path, large_path = f"{prefix}_summary.csv", f"{prefix}_large_files.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "folder_url", "folder_path", "history_start_path", "pinned_head", "revisions",
            "first_revision", "first_date", "last_revision", "last_date", "authors",
            "merge_commits_by_message", "dir_property_change_commits", "versions_checked",
            "versions_failed", "head_size_bytes", "head_size", "head_files", "head_dirs",
            f"head_files_ge_{threshold}_bytes", f"paths_ever_ge_{threshold}_bytes", "status",
        ])
        for r in results:
            complete = not r.error and not r.failures
            writer.writerow([
                r.url, r.scope, r.oldest_scope, r.head_revision, r.revisions,
                r.first_revision or "", r.first_date, r.last_revision or "", r.last_date, len(r.authors),
                r.merge_msg_commits, r.dir_prop_commits, r.versions_checked, len(r.failures),
                r.head_size, format_size(r.head_size), r.head_files, r.head_dirs,
                len(r.head_large), len(r.history_large),
                "OK" if complete else (f"FAILED: {r.error}" if r.error else "INCOMPLETE - re-run to retry"),
            ])
    with open(large_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "folder_url", "path", "max_size_bytes", "max_size", "peak_revision",
            "first_large_revision", "last_large_revision", "large_revision_count",
            "head_size_bytes", "status",
        ])
        for r in results:
            for rec in r.history_large:
                writer.writerow([
                    r.url, rec.path, rec.max_size, format_size(rec.max_size), rec.peak_revision,
                    rec.first_large_revision, rec.last_large_revision, rec.large_revision_count,
                    "" if rec.head_size is None else rec.head_size, rec.status,
                ])
    logger.info("CSV written: %s, %s", summary_path, large_path)


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Folder-scoped SVN inventory: revisions, size, and large files across history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if __doc__ else None,
    )
    parser.add_argument("folder_urls", nargs="*", help="Folder URL(s), e.g. https://host/repos/r/trunk/FolderX")
    parser.add_argument("--urls-file", metavar="FILE", help="Text file with one folder URL per line (# = comment).")
    parser.add_argument("-u", "--username", default=None, help="SVN username.")
    parser.add_argument("-p", "--password", default=None,
                        help="SVN password (visible in 'ps'; prefer caching credentials with 'svn info' first).")
    parser.add_argument("--threshold-mb", type=float, default=100.0, metavar="MB",
                        help="Large-file threshold in MiB (default: 100).")
    parser.add_argument("--jobs", type=int, default=4, metavar="N",
                        help="Parallel 'svn list' calls (default: 4).")
    parser.add_argument("--batch-size", type=int, default=50, metavar="N",
                        help="File versions per 'svn list' call (default: 50).")
    parser.add_argument("--retries", type=int, default=3, metavar="N",
                        help="Attempts per SVN call before giving up (default: 3).")
    parser.add_argument("--stop-on-copy", action="store_true",
                        help="Do not follow the folder's history across copies/renames.")
    parser.add_argument("--state-dir", default=".svn_folder_inventory_state", metavar="DIR",
                        help="Where progress is saved for resuming (default: .svn_folder_inventory_state).")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore saved progress and start again (re-pins HEAD).")
    parser.add_argument("--output-prefix", default="folder_inventory", metavar="PREFIX",
                        help="CSV prefix: PREFIX_summary.csv and PREFIX_large_files.csv (default: folder_inventory).")
    parser.add_argument("--log-file", default=None, metavar="FILE", help="Also write log messages to FILE.")
    args = parser.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers)

    urls = [u.strip().rstrip("/") for u in args.folder_urls]
    if args.urls_file:
        with open(args.urls_file, encoding="utf-8") as handle:
            urls += [line.strip().rstrip("/") for line in handle if line.strip() and not line.lstrip().startswith("#")]
    urls = list(dict.fromkeys(u for u in urls if u))
    if not urls:
        parser.error("provide at least one folder URL (or --urls-file)")
    if args.jobs < 1 or args.batch_size < 1 or args.threshold_mb <= 0:
        parser.error("--jobs, --batch-size and --threshold-mb must be positive")

    threshold = int(args.threshold_mb * MIB)
    svn = Svn(args.username, args.password, args.retries)
    os.makedirs(args.state_dir, exist_ok=True)

    results: list[FolderResult] = []
    for index, url in enumerate(urls, 1):
        logger.info("Folder %d/%d: %s", index, len(urls), url)
        try:
            result = inventory_folder(svn, url, args, threshold)
        except (SvnError, ET.ParseError, OSError, ValueError) as exc:
            logger.error("  Folder failed: %s", exc)
            result = FolderResult(url=url, error=str(exc))
        results.append(result)
        print_report(result, threshold)
        write_csvs(results, args.output_prefix, threshold)

    incomplete = [r.url for r in results if r.error or r.failures]
    if incomplete:
        logger.error("Incomplete folders (re-run the same command to retry): %s", ", ".join(incomplete))
        return 1
    logger.info("All %d folder(s) completed.", len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
