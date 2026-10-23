"""
Generalized Recursive SHA-256 Mirror Synchronization & Verification Tool
Milestone 4: Parity Automation between amazing-bose and E:\\autonew

Usage:
    py scratch/sync_and_verify.py                   # Default: Sync updated/added files + verify parity
    py scratch/sync_and_verify.py --verify          # Pure read-only verification (exit code 0 on match)
    py scratch/sync_and_verify.py --sync            # Explicit sync + verification
    py scratch/sync_and_verify.py --sync --dry-run  # Dry-run sync (preview changes without copying)
    py scratch/sync_and_verify.py --sync --prune    # Sync and delete orphan files in destination
    py scratch/sync_and_verify.py --help            # View all options
"""

import os
import sys
import stat
import shutil
import hashlib
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_SRC = r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose"
DEFAULT_DST = r"E:\autonew"

DEFAULT_EXCLUDES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".agents",
    "dist",
    "build",
    ".venv",
    "venv",
    "env",
    ".vscode",
    ".idea",
}

DEFAULT_FILE_EXCLUDES = {
    ".DS_Store",
    "Thumbs.db",
}

DEFAULT_SUFFIX_EXCLUDES = (
    ".pyc",
    ".pyo",
    ".pyd",
    ".egg-info",
)


def compute_sha256(filepath: str | Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hexadecimal digest for a file using chunked binary reading."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def is_path_excluded(
    rel_path: str,
    dir_excludes: Set[str] = DEFAULT_EXCLUDES,
    file_excludes: Set[str] = DEFAULT_FILE_EXCLUDES,
    suffix_excludes: Tuple[str, ...] = DEFAULT_SUFFIX_EXCLUDES,
) -> bool:
    """Evaluates whether a relative path should be excluded from sync/verification."""
    norm = os.path.normpath(rel_path)
    parts = norm.split(os.sep)

    # Check directory components
    for p in parts:
        if p in dir_excludes:
            return True
        if any(p.endswith(sfx) for sfx in suffix_excludes):
            return True

    # Check filename and extension
    fname = parts[-1]
    if fname in file_excludes:
        return True
    if any(fname.endswith(sfx) for sfx in suffix_excludes):
        return True

    return False


@dataclass
class FileEntry:
    rel_path: str
    abs_path: str
    size: int
    mtime: float
    sha256: Optional[str] = None


@dataclass
class ParityComparison:
    src_root: str
    dst_root: str
    src_files: Dict[str, FileEntry] = field(default_factory=dict)
    dst_files: Dict[str, FileEntry] = field(default_factory=dict)
    only_in_src: List[str] = field(default_factory=list)
    only_in_dst: List[str] = field(default_factory=list)
    matched: List[str] = field(default_factory=list)
    mismatched: List[Tuple[str, str, str]] = field(default_factory=list)  # (rel, src_hash, dst_hash)
    scan_duration: float = 0.0

    @property
    def is_perfect_match(self) -> bool:
        """True if all source files exist in destination with 100% bit-for-bit SHA-256 parity."""
        return len(self.only_in_src) == 0 and len(self.mismatched) == 0


def scan_tree(root: str, excludes: Set[str] = DEFAULT_EXCLUDES) -> Dict[str, FileEntry]:
    """Recursively scans directory tree skipping excluded directories and files."""
    root_abs = os.path.abspath(root)
    entries: Dict[str, FileEntry] = {}

    if not os.path.exists(root_abs):
        return entries

    for dirpath, dirnames, filenames in os.walk(root_abs):
        # Prune excluded directories in-place to prevent descending into them
        dirnames[:] = [
            d for d in dirnames
            if not is_path_excluded(os.path.relpath(os.path.join(dirpath, d), root_abs), dir_excludes=excludes)
        ]

        for fname in filenames:
            abs_fp = os.path.join(dirpath, fname)
            rel_fp = os.path.normpath(os.path.relpath(abs_fp, root_abs))
            if not is_path_excluded(rel_fp, dir_excludes=excludes):
                try:
                    stat_res = os.stat(abs_fp)
                    entries[rel_fp] = FileEntry(
                        rel_path=rel_fp,
                        abs_path=abs_fp,
                        size=stat_res.st_size,
                        mtime=stat_res.st_mtime,
                    )
                except OSError:
                    # Ignore inaccessible files or broken symlinks
                    continue

    return entries


def compare_trees(
    src_root: str,
    dst_root: str,
    excludes: Set[str] = DEFAULT_EXCLUDES,
    compute_all_hashes: bool = True,
    progress_cb=None,
) -> ParityComparison:
    """Compares source and destination trees and computes SHA-256 hashes for common files."""
    t0 = time.time()
    src_files = scan_tree(src_root, excludes=excludes)
    dst_files = scan_tree(dst_root, excludes=excludes)

    src_keys = set(src_files.keys())
    dst_keys = set(dst_files.keys())

    only_in_src = sorted(list(src_keys - dst_keys))
    only_in_dst = sorted(list(dst_keys - src_keys))
    common_keys = sorted(list(src_keys & dst_keys))

    matched: List[str] = []
    mismatched: List[Tuple[str, str, str]] = []

    total_common = len(common_keys)
    for idx, rel in enumerate(common_keys):
        s_entry = src_files[rel]
        d_entry = dst_files[rel]

        if progress_cb and (idx % 100 == 0 or idx == total_common - 1):
            progress_cb(idx + 1, total_common)

        # Quick size shortcut: if sizes differ, hashes are guaranteed to differ
        if s_entry.size != d_entry.size:
            s_hash = compute_sha256(s_entry.abs_path)
            d_hash = compute_sha256(d_entry.abs_path)
            s_entry.sha256 = s_hash
            d_entry.sha256 = d_hash
            mismatched.append((rel, s_hash, d_hash))
            continue

        if compute_all_hashes:
            s_hash = compute_sha256(s_entry.abs_path)
            d_hash = compute_sha256(d_entry.abs_path)
            s_entry.sha256 = s_hash
            d_entry.sha256 = d_hash
            if s_hash == d_hash:
                matched.append(rel)
            else:
                mismatched.append((rel, s_hash, d_hash))
        else:
            matched.append(rel)

    t1 = time.time()
    return ParityComparison(
        src_root=src_root,
        dst_root=dst_root,
        src_files=src_files,
        dst_files=dst_files,
        only_in_src=only_in_src,
        only_in_dst=only_in_dst,
        matched=matched,
        mismatched=mismatched,
        scan_duration=t1 - t0,
    )


def safe_copy(src_path: str, dst_path: str) -> None:
    """Copies file to destination, clearing read-only attributes if needed on Windows."""
    dst_dir = os.path.dirname(dst_path)
    os.makedirs(dst_dir, exist_ok=True)

    if os.path.exists(dst_path):
        # Clear read-only bit if present
        try:
            os.chmod(dst_path, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass

    shutil.copy2(src_path, dst_path)


def run_sync(
    src_root: str,
    dst_root: str,
    excludes: Set[str] = DEFAULT_EXCLUDES,
    dry_run: bool = False,
    prune: bool = False,
    verbose: bool = False,
) -> bool:
    """Executes mirror synchronization followed by complete SHA-256 verification."""
    print("================================================================================")
    print("           RECURSIVE SHA-256 MIRROR SYNCHRONIZATION & VERIFICATION              ")
    print("================================================================================")
    print(f" Source Root : {src_root}")
    print(f" Target Mirror: {dst_root}")
    print(f" Mode        : {'DRY-RUN (Preview)' if dry_run else 'SYNC & VERIFY'}")
    print(f" Prune Mode  : {'ENABLED (Delete orphans)' if prune else 'DISABLED (Preserve orphans)'}")
    print("--------------------------------------------------------------------------------")

    if not os.path.exists(src_root):
        print(f"[ERROR] Source root does not exist: {src_root}")
        return False

    os.makedirs(dst_root, exist_ok=True)

    print("[*] Scanning directory trees and computing initial parity...")
    comparison = compare_trees(src_root, dst_root, excludes=excludes)

    print(f"[+] Scan completed in {comparison.scan_duration:.2f}s:")
    print(f"    - Source eligible files : {len(comparison.src_files)}")
    print(f"    - Mirror eligible files : {len(comparison.dst_files)}")
    print(f"    - Common files in both  : {len(comparison.src_files) - len(comparison.only_in_src)}")
    print(f"    - Existing SHA-256 match: {len(comparison.matched)}")
    print(f"    - Hash mismatches       : {len(comparison.mismatched)}")
    print(f"    - Missing in mirror     : {len(comparison.only_in_src)}")
    print(f"    - Orphans only in mirror: {len(comparison.only_in_dst)}")
    print("--------------------------------------------------------------------------------")

    files_to_sync = comparison.only_in_src + [m[0] for m in comparison.mismatched]

    if not files_to_sync and not (prune and comparison.only_in_dst):
        print("[OK] Mirror is already 100% in sync with source! No copy operations needed.")
    else:
        print(f"[*] Synchronizing {len(files_to_sync)} files to mirror...")
        copied_count = 0
        failed_count = 0

        for rel in files_to_sync:
            src_fp = os.path.join(src_root, rel)
            dst_fp = os.path.join(dst_root, rel)

            is_new = rel in comparison.only_in_src
            action_tag = "ADD" if is_new else "UPDATE"

            if verbose or len(files_to_sync) <= 30:
                print(f"  [{action_tag}] {rel}")

            if not dry_run:
                try:
                    safe_copy(src_fp, dst_fp)
                    # Verify immediately
                    s_hash = compute_sha256(src_fp)
                    d_hash = compute_sha256(dst_fp)
                    if s_hash != d_hash:
                        print(f"  [ERROR] Verification failed after copy for: {rel}")
                        failed_count += 1
                    else:
                        copied_count += 1
                except Exception as ex:
                    print(f"  [ERROR] Failed to copy {rel}: {ex}")
                    failed_count += 1
            else:
                copied_count += 1

        if dry_run:
            print(f"[DRY-RUN] Would sync {copied_count} files (0 actually copied).")
        else:
            print(f"[+] Successfully synced {copied_count} files (Failures: {failed_count}).")

        # Handle pruning of orphans if requested
        if prune and comparison.only_in_dst:
            print(f"[*] Pruning {len(comparison.only_in_dst)} orphan files in mirror...")
            pruned_count = 0
            for rel in comparison.only_in_dst:
                dst_fp = os.path.join(dst_root, rel)
                if verbose:
                    print(f"  [DELETE] {rel}")
                if not dry_run:
                    try:
                        os.chmod(dst_fp, stat.S_IWRITE)
                        os.remove(dst_fp)
                        pruned_count += 1
                    except Exception as ex:
                        print(f"  [ERROR] Failed to delete orphan {rel}: {ex}")
                else:
                    pruned_count += 1
            action_word = "Would prune" if dry_run else "Pruned"
            print(f"[+] {action_word} {pruned_count} orphan files.")

    print("--------------------------------------------------------------------------------")
    print("[*] Running final verification check across all eligible source files...")
    final_comp = compare_trees(src_root, dst_root, excludes=excludes)

    all_matched = final_comp.is_perfect_match
    print(f" Final Parity Status : {'100% IDENTICAL (MATCH)' if all_matched else 'DIVERGENT (MISMATCH)'}")
    print(f" Total Source Files  : {len(final_comp.src_files)}")
    print(f" Verified Matches    : {len(final_comp.matched)} / {len(final_comp.src_files)}")
    print(f" Unsynced/Mismatched : {len(final_comp.mismatched)}")
    print(f" Missing in Mirror   : {len(final_comp.only_in_src)}")
    print(f" Mirror Orphans      : {len(final_comp.only_in_dst)}")
    print("================================================================================")

    if all_matched:
        print("[SUCCESS] 100% bit-for-bit SHA-256 byte parity verified between amazing-bose and E:\\autonew.")
        return True
    else:
        print("[FAILURE] Mirror parity check failed. Inspect mismatches above.")
        return False


def run_verify_only(
    src_root: str,
    dst_root: str,
    excludes: Set[str] = DEFAULT_EXCLUDES,
    verbose: bool = False,
) -> bool:
    """Executes read-only SHA-256 parity verification between source and mirror."""
    print("================================================================================")
    print("             READ-ONLY SHA-256 MIRROR PARITY VERIFICATION                       ")
    print("================================================================================")
    print(f" Source Root : {src_root}")
    print(f" Target Mirror: {dst_root}")
    print(" Mode        : VERIFY ONLY (Read-Only)")
    print("--------------------------------------------------------------------------------")

    if not os.path.exists(src_root):
        print(f"[ERROR] Source root does not exist: {src_root}")
        return False
    if not os.path.exists(dst_root):
        print(f"[ERROR] Target mirror does not exist: {dst_root}")
        return False

    print("[*] Computing SHA-256 hashes across all eligible files...")
    comp = compare_trees(src_root, dst_root, excludes=excludes)

    print(f"[+] Comparison completed in {comp.scan_duration:.2f}s:")
    print(f"    - Source eligible files : {len(comp.src_files)}")
    print(f"    - Mirror eligible files : {len(comp.dst_files)}")
    print(f"    - Verified SHA-256 Match: {len(comp.matched)}")
    print(f"    - Hash Mismatches       : {len(comp.mismatched)}")
    print(f"    - Missing in Mirror     : {len(comp.only_in_src)}")
    print(f"    - Mirror Orphans        : {len(comp.only_in_dst)}")
    print("--------------------------------------------------------------------------------")

    if comp.only_in_src:
        print(f"[!] Files present in source but missing in mirror ({len(comp.only_in_src)}):")
        for rel in comp.only_in_src[:20]:
            print(f"    - MISSING: {rel}")
        if len(comp.only_in_src) > 20:
            print(f"    ... and {len(comp.only_in_src) - 20} more.")

    if comp.mismatched:
        print(f"[!] Files with SHA-256 byte divergence ({len(comp.mismatched)}):")
        for rel, s_h, d_h in comp.mismatched[:20]:
            print(f"    != {rel}")
            print(f"       src: {s_h}")
            print(f"       dst: {d_h}")
        if len(comp.mismatched) > 20:
            print(f"    ... and {len(comp.mismatched) - 20} more.")

    if comp.only_in_dst and verbose:
        print(f"[i] Mirror orphan files ({len(comp.only_in_dst)}):")
        for rel in comp.only_in_dst[:20]:
            print(f"    + ORPHAN: {rel}")
        if len(comp.only_in_dst) > 20:
            print(f"    ... and {len(comp.only_in_dst) - 20} more.")

    print("================================================================================")
    if comp.is_perfect_match:
        print("[SUCCESS] 100% BIT-FOR-BIT SHA-256 PARITY VERIFIED (EXIT 0)")
        return True
    else:
        print("[FAILURE] PARITY MISMATCH DETECTED (EXIT 1)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generalized Recursive SHA-256 Mirror Synchronization & Verification Tool"
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"Source root path (default: {DEFAULT_SRC})")
    parser.add_argument("--dst", default=DEFAULT_DST, help=f"Destination mirror path (default: {DEFAULT_DST})")
    parser.add_argument("--verify", action="store_true", help="Perform read-only verification without modifying files")
    parser.add_argument("--sync", action="store_true", help="Synchronize updated/added files to mirror and verify")
    parser.add_argument("--dry-run", action="store_true", help="Simulate sync operations without writing changes")
    parser.add_argument("--prune", action="store_true", help="Delete destination files not present in source")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose per-file output")

    args = parser.parse_args()

    # Determine mode: default to sync if neither --verify nor --sync specified, maintaining compatibility
    if args.verify and not args.sync:
        success = run_verify_only(args.src, args.dst, verbose=args.verbose)
    else:
        success = run_sync(
            args.src,
            args.dst,
            dry_run=args.dry_run,
            prune=args.prune,
            verbose=args.verbose,
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
