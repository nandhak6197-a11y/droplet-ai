"""
Unit Tests for Generalized Recursive SHA-256 Mirror Synchronization & Verification Tool
Verifies:
  - Chunked SHA-256 hashing against standard vectors
  - Directory and file exclusion filtering
  - Tree scanning and comparison logic
  - Safe copy handling including read-only files on Windows
  - Sync and verify execution modes (dry-run, prune, full sync)
  - CLI argument parsing, flags, and exit code semantics (0 on parity, 1 on divergence)
"""

import os
import sys
import stat
import shutil
import hashlib
import tempfile
import subprocess
from pathlib import Path
import pytest

from scratch.sync_and_verify import (
    compute_sha256,
    is_path_excluded,
    scan_tree,
    compare_trees,
    safe_copy,
    run_sync,
    run_verify_only,
    DEFAULT_EXCLUDES,
    DEFAULT_FILE_EXCLUDES,
    DEFAULT_SUFFIX_EXCLUDES,
)


def test_compute_sha256_known_vector(tmp_path: Path):
    sample_file = tmp_path / "sample.txt"
    sample_bytes = b"Screen-Aware Deep Self-Reasoning Spatial Droplet Engine\n"
    sample_file.write_bytes(sample_bytes)

    expected_hash = hashlib.sha256(sample_bytes).hexdigest()
    computed_hash = compute_sha256(sample_file)

    assert computed_hash == expected_hash
    assert len(computed_hash) == 64


def test_is_path_excluded_directories_and_files():
    # Excluded directories
    assert is_path_excluded(".git/HEAD")
    assert is_path_excluded("foo/.git/config")
    assert is_path_excluded("__pycache__/module.cpython-310.pyc")
    assert is_path_excluded("spatial_droplet/__pycache__/reasoner.cpython-310.pyc")
    assert is_path_excluded(".pytest_cache/v/cache/lastfailed")
    assert is_path_excluded(".agents/teamwork/plan.md")
    assert is_path_excluded("dist/package.whl")
    assert is_path_excluded("build/temp")
    assert is_path_excluded(".venv/bin/python")
    assert is_path_excluded("venv/Scripts/python.exe")
    assert is_path_excluded(".vscode/settings.json")
    assert is_path_excluded("foo.egg-info/SOURCES.txt")

    # Excluded filenames and suffixes
    assert is_path_excluded("foo/.DS_Store")
    assert is_path_excluded("Thumbs.db")
    assert is_path_excluded("module.pyc")
    assert is_path_excluded("native.pyd")

    # Included paths (must NOT be excluded)
    assert not is_path_excluded("spatial_droplet/core/models.py")
    assert not is_path_excluded("scratch/test_mirror_sync_and_verify.py")
    assert not is_path_excluded("pytest.ini")
    assert not is_path_excluded("PROJECT.md")
    assert not is_path_excluded("TEST_READY.md")


def test_scan_tree_skips_excluded_dirs_and_files(tmp_path: Path):
    # Set up test filesystem structure
    (tmp_path / "valid_dir").mkdir()
    (tmp_path / "valid_dir" / "file1.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "valid_dir" / "file2.txt").write_text("sample content", encoding="utf-8")
    (tmp_path / "root_file.md").write_text("# Readme", encoding="utf-8")

    # Excluded directories
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("git config", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.pyc").write_bytes(b"\x00\x01\x02")
    (tmp_path / ".agents").mkdir()
    (tmp_path / ".agents" / "meta.json").write_text("{}", encoding="utf-8")

    # Excluded files in valid dir
    (tmp_path / "valid_dir" / ".DS_Store").write_bytes(b"\x00")
    (tmp_path / "valid_dir" / "compiled.pyc").write_bytes(b"\x00")

    entries = scan_tree(str(tmp_path))

    expected_rel_paths = {
        os.path.normpath("valid_dir/file1.py"),
        os.path.normpath("valid_dir/file2.txt"),
        os.path.normpath("root_file.md"),
    }
    scanned_rel_paths = set(entries.keys())

    assert scanned_rel_paths == expected_rel_paths
    for rel_path, entry in entries.items():
        assert entry.size > 0
        assert os.path.exists(entry.abs_path)


def test_compare_trees_identical(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    (src / "file_a.txt").write_text("alpha", encoding="utf-8")
    (dst / "file_a.txt").write_text("alpha", encoding="utf-8")
    (src / "sub").mkdir()
    (dst / "sub").mkdir()
    (src / "sub" / "file_b.txt").write_text("beta", encoding="utf-8")
    (dst / "sub" / "file_b.txt").write_text("beta", encoding="utf-8")

    comp = compare_trees(str(src), str(dst))

    assert comp.is_perfect_match
    assert len(comp.only_in_src) == 0
    assert len(comp.only_in_dst) == 0
    assert len(comp.mismatched) == 0
    assert len(comp.matched) == 2


def test_compare_trees_detects_mismatches_and_missing(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    # Identical file
    (src / "matched.txt").write_text("common", encoding="utf-8")
    (dst / "matched.txt").write_text("common", encoding="utf-8")

    # Missing in dst (only in src)
    (src / "only_src.txt").write_text("new file", encoding="utf-8")

    # Orphan in dst (only in dst)
    (dst / "orphan.txt").write_text("old file", encoding="utf-8")

    # Hash mismatch (same size, different content)
    (src / "diff_content.txt").write_text("ABCDEF", encoding="utf-8")
    (dst / "diff_content.txt").write_text("UVWXYZ", encoding="utf-8")

    # Hash mismatch (different size)
    (src / "diff_size.txt").write_text("short", encoding="utf-8")
    (dst / "diff_size.txt").write_text("much longer content here", encoding="utf-8")

    comp = compare_trees(str(src), str(dst))

    assert not comp.is_perfect_match
    assert os.path.normpath("only_src.txt") in comp.only_in_src
    assert os.path.normpath("orphan.txt") in comp.only_in_dst
    assert len(comp.matched) == 1
    assert os.path.normpath("matched.txt") in comp.matched

    mismatched_names = [m[0] for m in comp.mismatched]
    assert os.path.normpath("diff_content.txt") in mismatched_names
    assert os.path.normpath("diff_size.txt") in mismatched_names


def test_safe_copy_handles_readonly_dest(tmp_path: Path):
    src_file = tmp_path / "source.txt"
    dst_file = tmp_path / "target.txt"

    src_file.write_text("new updated content", encoding="utf-8")
    dst_file.write_text("old read-only content", encoding="utf-8")

    # Set destination to read-only
    os.chmod(str(dst_file), stat.S_IREAD)

    # safe_copy should clear read-only flag and overwrite cleanly
    safe_copy(str(src_file), str(dst_file))

    assert dst_file.read_text(encoding="utf-8") == "new updated content"


def test_run_sync_achieves_complete_parity(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    (src / "file1.txt").write_text("content 1", encoding="utf-8")
    (src / "sub").mkdir()
    (src / "sub" / "file2.txt").write_text("content 2", encoding="utf-8")

    # Before sync, dst is empty
    assert not run_verify_only(str(src), str(dst))

    # Run sync
    sync_success = run_sync(str(src), str(dst), verbose=False)
    assert sync_success is True

    # Post-sync verification must pass 100%
    verify_success = run_verify_only(str(src), str(dst))
    assert verify_success is True

    # Modifying a file in src introduces divergence again
    (src / "file1.txt").write_text("modified content 1", encoding="utf-8")
    assert not run_verify_only(str(src), str(dst))

    # Re-sync restores parity
    re_sync_success = run_sync(str(src), str(dst))
    assert re_sync_success is True
    assert run_verify_only(str(src), str(dst)) is True


def test_run_sync_dry_run_does_not_mutate(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    (src / "file.txt").write_text("source content", encoding="utf-8")

    # Execute dry-run sync
    run_sync(str(src), str(dst), dry_run=True)

    # Destination should still not contain the file
    assert not (dst / "file.txt").exists()


def test_run_sync_prune_removes_orphans(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    (src / "valid.txt").write_text("source file", encoding="utf-8")
    (dst / "valid.txt").write_text("source file", encoding="utf-8")
    (dst / "orphan.txt").write_text("orphan to prune", encoding="utf-8")

    # Without prune, orphan is preserved
    run_sync(str(src), str(dst), prune=False)
    assert (dst / "orphan.txt").exists()

    # With prune, orphan is deleted
    run_sync(str(src), str(dst), prune=True)
    assert not (dst / "orphan.txt").exists()
    assert (dst / "valid.txt").exists()


def test_cli_execution_and_exit_codes(tmp_path: Path):
    sync_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "sync_and_verify.py"))
    assert os.path.exists(sync_script)

    # 1. Test --help flag exits with 0
    res_help = subprocess.run([sys.executable, sync_script, "--help"], capture_output=True, text=True)
    assert res_help.returncode == 0
    assert "Generalized Recursive SHA-256 Mirror Synchronization" in res_help.stdout

    src = tmp_path / "cli_src"
    dst = tmp_path / "cli_dst"
    src.mkdir()
    dst.mkdir()

    (src / "alpha.txt").write_text("alpha data", encoding="utf-8")

    # 2. Test --verify fails on divergence (exit code 1)
    res_verify_fail = subprocess.run(
        [sys.executable, sync_script, "--src", str(src), "--dst", str(dst), "--verify"],
        capture_output=True,
        text=True,
    )
    assert res_verify_fail.returncode == 1
    assert "PARITY MISMATCH DETECTED (EXIT 1)" in res_verify_fail.stdout

    # 3. Test --sync achieves parity (exit code 0)
    res_sync = subprocess.run(
        [sys.executable, sync_script, "--src", str(src), "--dst", str(dst), "--sync"],
        capture_output=True,
        text=True,
    )
    assert res_sync.returncode == 0
    assert "100% bit-for-bit SHA-256 byte parity verified" in res_sync.stdout

    # 4. Test --verify succeeds on parity (exit code 0)
    res_verify_ok = subprocess.run(
        [sys.executable, sync_script, "--src", str(src), "--dst", str(dst), "--verify"],
        capture_output=True,
        text=True,
    )
    assert res_verify_ok.returncode == 0
    assert "100% BIT-FOR-BIT SHA-256 PARITY VERIFIED (EXIT 0)" in res_verify_ok.stdout
