"""
dedupe_ttl_rows.py — Archive all timestamped TTL files, then keep only the
latest file per row number in the source directory.

Filename pattern:  {YYYYMMDD_HHMMSS}_{row}_{identifier}.ttl
The row number is the zero-padded integer between the timestamp and the identifier.

Steps:
  1. Copy every matching TTL file to --archive-dir (default: assets/triples/archive/)
  2. For each row number, identify the file with the latest timestamp
  3. Remove all older duplicates from --ttl-dir

Usage:
    python dedupe_ttl_rows.py [--ttl-dir <path>] [--archive-dir <path>] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PATTERN = re.compile(r"^(\d{8}_\d{6})_(\d+)_([0-9a-f]+)\.ttl$")

DEFAULT_TTL_DIR     = Path(__file__).resolve().parents[2] / "assets" / "triples"
DEFAULT_ARCHIVE_DIR = DEFAULT_TTL_DIR / "archive"


def parse_args():
    parser = argparse.ArgumentParser(description="Archive and deduplicate timestamped TTL files")
    parser.add_argument(
        "--ttl-dir", default=str(DEFAULT_TTL_DIR),
        help=f"Directory containing timestamped TTL files (default: {DEFAULT_TTL_DIR})",
    )
    parser.add_argument(
        "--archive-dir", default=str(DEFAULT_ARCHIVE_DIR),
        help=f"Directory to copy all TTL files into before deduplication (default: {DEFAULT_ARCHIVE_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without making any changes",
    )
    return parser.parse_args()


def collect_files(ttl_dir: Path) -> dict[int, list[tuple[str, Path]]]:
    """Return {row_number: [(timestamp, path), ...]} for all matching files."""
    by_row: dict[int, list[tuple[str, Path]]] = defaultdict(list)
    for f in ttl_dir.iterdir():
        m = PATTERN.match(f.name)
        if not m:
            continue
        timestamp, row_str, _ = m.groups()
        by_row[int(row_str)].append((timestamp, f))
    return by_row


def main():
    args = parse_args()
    ttl_dir     = Path(args.ttl_dir).resolve()
    archive_dir = Path(args.archive_dir).resolve()
    dry_run     = args.dry_run

    if not ttl_dir.exists():
        logger.error(f"TTL directory not found: {ttl_dir}")
        return

    by_row = collect_files(ttl_dir)
    if not by_row:
        logger.info("No timestamped TTL files found — nothing to do.")
        return

    all_files = [(ts, f) for files in by_row.values() for ts, f in files]
    logger.info(f"Found {len(all_files)} timestamped TTL files across {len(by_row)} row numbers")

    # Step 1: archive everything
    if dry_run:
        logger.info(f"[dry-run] Would create archive dir: {archive_dir}")
        logger.info(f"[dry-run] Would copy {len(all_files)} files to {archive_dir}")
    else:
        archive_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for _, src in all_files:
            dst = archive_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
        logger.info(f"Archived {copied} new file(s) to {archive_dir}")

    # Step 2: deduplicate — keep latest timestamp per row, remove the rest
    to_keep   = []
    to_remove = []

    for row_num in sorted(by_row):
        files = by_row[row_num]
        files_sorted = sorted(files, key=lambda x: x[0], reverse=True)  # latest first
        keep_ts, keep_path = files_sorted[0]
        to_keep.append((row_num, keep_ts, keep_path))
        for ts, path in files_sorted[1:]:
            to_remove.append((row_num, ts, path))

    logger.info(f"Keeping  {len(to_keep)} file(s) (one per row)")
    logger.info(f"Removing {len(to_remove)} older duplicate(s)")

    if to_remove:
        logger.info("─" * 60)
        for row_num, ts, path in to_remove:
            logger.info(f"  [row {row_num:03d}] removing {path.name}  (superseded by later run)")
            if not dry_run:
                path.unlink()
        logger.info("─" * 60)

    if dry_run:
        logger.info("[dry-run] No files were changed.")
    else:
        logger.info("Done.")
        logger.info(f"Archive: {archive_dir}  ({len(all_files)} files)")
        logger.info(f"Active:  {ttl_dir}  ({len(to_keep)} files after deduplication)")


if __name__ == "__main__":
    main()
