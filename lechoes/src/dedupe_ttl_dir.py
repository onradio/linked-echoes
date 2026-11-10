"""
dedupe_ttl_dir.py

For each row index, keeps only the latest TTL file (by timestamp prefix)
and removes all older ones.

Usage:
    python dedupe_ttl_dir.py --ttl-dir ../../assets/kg_triple_store_gt/ttl
    python dedupe_ttl_dir.py --ttl-dir ../../assets/kg_triple_store_gt/ttl --dry-run
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

TTL_RE = re.compile(r"^(\d{8}_\d{6})_(\d+)_([0-9a-f]+)\.ttl$")


def main():
    parser = argparse.ArgumentParser(description="Keep only the latest TTL per row index")
    parser.add_argument("--ttl-dir", type=Path,
                        default=Path("../../assets/kg_triple_store_gt/ttl"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be removed without deleting")
    args = parser.parse_args()

    # Group files by row index
    by_row: dict[int, list[tuple[str, Path]]] = defaultdict(list)
    for f in args.ttl_dir.glob("*.ttl"):
        m = TTL_RE.match(f.name)
        if not m:
            continue
        ts, row_str, _ = m.groups()
        by_row[int(row_str)].append((ts, f))

    removed = 0
    kept = 0
    for row_idx, files in sorted(by_row.items()):
        if len(files) == 1:
            kept += 1
            continue
        # Sort by timestamp descending — latest first
        files.sort(key=lambda x: x[0], reverse=True)
        latest_ts, latest_path = files[0]
        for ts, path in files[1:]:
            if args.dry_run:
                print(f"[dry-run] would remove {path.name} (superseded by {latest_path.name})")
            else:
                path.unlink()
                print(f"Removed {path.name} → kept {latest_path.name}")
            removed += 1
        kept += 1

    print(f"\n{'[dry-run] ' if args.dry_run else ''}Done: {kept} rows kept, {removed} old files removed.")


if __name__ == "__main__":
    main()
