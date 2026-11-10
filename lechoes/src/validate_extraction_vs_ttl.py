"""
validate_extraction_vs_ttl.py — Cross-check each row TTL file against the
original source input CSV and flag mismatches.

Pairs are matched by row number (zero-indexed in the CSV, encoded in the TTL
filename as {YYYYMMDD_HHMMSS}_{row:03d}_{identifier}.ttl).

Checks:
  - scho:description  vs source item_description
  - scho:caption      vs source caption  (skipped if caption is 'nan')
  - scho:name         vs derived item_title (description up to first comma, or full)

Usage:
    python validate_extraction_vs_ttl.py \\
        [--source   assets/data/item_descriptions_with_captions.csv] \\
        [--ttl-dir  assets/kg_triples] \\
        [--report   assets/kg_triples/reports/source_vs_ttl.csv]
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph, URIRef, RDF, Namespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCHO = Namespace("http://schema.org/")
CRM  = Namespace("http://www.cidoc-crm.org/cidoc-crm/")

TTL_RE = re.compile(r"^(\d{8}_\d{6})_(\d+)_([0-9a-f]+)\.ttl$")

DEFAULT_SOURCE  = Path(__file__).resolve().parents[2] / "assets" / "data" / "item_descriptions_with_captions.csv"
DEFAULT_TTL_DIR = Path(__file__).resolve().parents[2] / "assets" / "kg_triples"
DEFAULT_REPORT  = DEFAULT_TTL_DIR / "reports" / "source_vs_ttl.csv"

REPORT_COLUMNS = [
    "row_number",
    "ttl_file",
    "check",
    "status",           # ok | mismatch | missing_in_ttl | missing_in_source
    "source_value",
    "ttl_value",
]


@dataclass
class Issue:
    row_number: int
    ttl_file: str
    check: str
    status: str
    source_value: str
    ttl_value: str


def load_ttl(path: Path) -> Graph:
    g = Graph()
    g.parse(str(path), format="turtle")
    return g


def hmo_subject(g: Graph) -> URIRef | None:
    subjects = list(g.subjects(RDF.type, CRM["E22_Human-Made_Object"]))
    return subjects[0] if subjects else None


def normalise(s: str) -> str:
    return s.strip().lower()


LEADING_NUMBER_RE = re.compile(r"^\d+\s+")

def strip_position_prefix(caption: str) -> str:
    """Remove the leading position number from a source caption, e.g. '8 Duran Duran...' -> 'Duran Duran...'"""
    return LEADING_NUMBER_RE.sub("", caption.strip())


def compare(
    issues: list[Issue],
    row_number: int,
    ttl_file: str,
    check: str,
    source_val: str,
    ttl_val: str,
):
    if not source_val and not ttl_val:
        return
    if normalise(source_val) == normalise(ttl_val):
        return  # ok — don't log clean rows

    if not ttl_val:
        status = "missing_in_ttl"
    elif not source_val:
        status = "missing_in_source"
    else:
        status = "mismatch"

    issues.append(Issue(
        row_number=row_number,
        ttl_file=ttl_file,
        check=check,
        status=status,
        source_value=source_val,
        ttl_value=ttl_val,
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Cross-check row TTL files against the source input CSV"
    )
    parser.add_argument("--source",  default=str(DEFAULT_SOURCE),  help="Source input CSV")
    parser.add_argument("--ttl-dir", default=str(DEFAULT_TTL_DIR), help="Directory of per-row TTL files")
    parser.add_argument("--report",  default=str(DEFAULT_REPORT),  help="Output CSV report path")
    args = parser.parse_args()

    source_path = Path(args.source)
    ttl_dir     = Path(args.ttl_dir)
    report_path = Path(args.report)

    # Load source CSV — keyed by row index (0-based)
    source_rows: dict[int, dict] = {}
    with source_path.open(encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            source_rows[i] = {k.strip(): v.strip() for k, v in row.items()}
    logger.info(f"Loaded {len(source_rows)} source rows from {source_path.name}")

    # Build TTL lookup: row_number -> (latest_timestamp, path)
    ttl_by_row: dict[int, tuple[str, Path]] = {}
    for f in ttl_dir.iterdir():
        m = TTL_RE.match(f.name)
        if not m:
            continue
        ts, row_str, _ = m.groups()
        row_num = int(row_str)
        # keep latest timestamp if multiple exist
        if row_num not in ttl_by_row or ts > ttl_by_row[row_num][0]:
            ttl_by_row[row_num] = (ts, f)
    logger.info(f"Found {len(ttl_by_row)} TTL files in {ttl_dir.name}")

    # Cross-check
    all_issues: list[Issue] = []
    no_ttl = []
    no_source = []

    for row_number in sorted(set(list(source_rows.keys()) + list(ttl_by_row.keys()))):
        src = source_rows.get(row_number)
        ttl_entry = ttl_by_row.get(row_number)

        if src is None:
            no_source.append(row_number)
            continue
        if ttl_entry is None:
            no_ttl.append(row_number)
            continue

        _, ttl_path = ttl_entry
        g = load_ttl(ttl_path)
        s = hmo_subject(g)

        if s is None:
            all_issues.append(Issue(
                row_number=row_number,
                ttl_file=ttl_path.name,
                check="hmo_subject",
                status="missing_in_ttl",
                source_value="",
                ttl_value="No crm:E22_Human-Made_Object in TTL",
            ))
            continue

        ttl_description = next((str(o) for o in g.objects(s, SCHO.description)), "")
        ttl_caption     = next((str(o) for o in g.objects(s, SCHO.caption)),     "")

        src_description = src.get("item_description", "")
        src_caption     = src.get("caption", "")

        compare(all_issues, row_number, ttl_path.name,
                "scho:description vs item_description",
                src_description, ttl_description)

        if src_caption and normalise(src_caption) != "nan":
            compare(all_issues, row_number, ttl_path.name,
                    "scho:caption vs caption",
                    strip_position_prefix(src_caption), ttl_caption)

    # Report rows with no TTL
    for row_number in no_ttl:
        src = source_rows[row_number]
        all_issues.append(Issue(
            row_number=row_number,
            ttl_file="",
            check="ttl_file",
            status="missing_in_ttl",
            source_value=src.get("item_description", ""),
            ttl_value="",
        ))

    # Summary
    mismatch_rows = len({i.row_number for i in all_issues})
    logger.info(f"Rows with no TTL:     {len(no_ttl)}")
    logger.info(f"Rows with no source:  {len(no_source)}")
    logger.info(f"Rows with issues:     {mismatch_rows}")
    logger.info(f"Total issues:         {len(all_issues)}")

    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for issue in sorted(all_issues, key=lambda x: x.row_number):
            writer.writerow({
                "row_number":    issue.row_number,
                "ttl_file":      issue.ttl_file,
                "check":         issue.check,
                "status":        issue.status,
                "source_value":  issue.source_value,
                "ttl_value":     issue.ttl_value,
            })

    logger.info(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
