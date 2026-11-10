#!/usr/bin/env python3
"""Build the mention-level annotation sheet for the BME entity-linking gold standard.

Reads the object-level gold standard CSV (one row per museum object) and emits
one row per distinct (normalised label, entity type) pair, together with the
objects that mention it. The Wikidata QID and notes columns are left blank for
the annotator to complete.
"""
import argparse
import csv
import re
import unicodedata
from collections import OrderedDict

# metamodel column -> entity type used by the linker (and by the ECK registry)
COLUMN_TO_TYPE = OrderedDict([
    ("music_artist", "music artist"),
    ("music_work",   "music work"),
    ("music_album",  "music album"),
    ("event",        "event"),
    ("venue",        "venue"),
    ("place",        "place"),
    ("country",      "country"),
    ("genre",        "music genre"),
    ("material",     "material"),
    ("colour",       "colour"),
])

MULTIVALUE_SEP = "|"
YEARLIKE = re.compile(r"^(1[5-9]|20)\d{2}$")


def norm(value: str) -> str:
    """Whitespace/unicode normalisation only - spelling and case are preserved,
    because surface errors in the legacy data are part of the linking task."""
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", value).strip()


def key_of(label: str, entity_type: str):
    return (entity_type, label.casefold())


def flags_for(label: str, entity_type: str) -> str:
    """Heuristic anomaly flags for the annotator; never silently drops a value."""
    out = []
    if YEARLIKE.match(label):
        out.append("year-like value in a non-year column")
    if len(label) <= 2:
        out.append("very short label")
    if label != label.strip() or "  " in label:
        out.append("whitespace anomaly")
    return "; ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("objects_csv")
    ap.add_argument("-o", "--out", default="annotation_sheet.csv")
    ap.add_argument("--id-prefix", default="M")
    args = ap.parse_args()

    mentions = OrderedDict()

    with open(args.objects_csv, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=1):
            obj_id = f"OBJ-{i:03d}"
            obj_name = norm(row.get("name") or row.get("description") or "")
            for column, entity_type in COLUMN_TO_TYPE.items():
                raw = row.get(column) or ""
                for part in raw.split(MULTIVALUE_SEP):
                    label = norm(part)
                    if not label:
                        continue
                    k = key_of(label, entity_type)
                    rec = mentions.setdefault(k, {
                        "entity_type": entity_type,
                        "variants": OrderedDict(),
                        "objects": [],
                    })
                    rec["variants"][label] = rec["variants"].get(label, 0) + 1
                    rec["objects"].append((obj_id, obj_name, row.get("case_number", "")))

    fieldnames = [
        "mention_id", "label", "entity_type", "n_objects", "object_ids",
        "object_refs", "surface_variants", "auto_flag",
        "wikidata_qid", "alt_qids", "nil_reason", "notes",
    ]
    rows = []
    # sort by type (in metamodel order), then by descending frequency, then label
    type_order = {t: n for n, t in enumerate(COLUMN_TO_TYPE.values())}
    ordered = sorted(
        mentions.items(),
        key=lambda kv: (type_order[kv[1]["entity_type"]], -len(kv[1]["objects"]), kv[0][1]),
    )
    for n, (k, rec) in enumerate(ordered, start=1):
        canonical = max(rec["variants"].items(), key=lambda kv: kv[1])[0]
        refs = [f"{name} (case {case})" for _, name, case in rec["objects"]]
        shown = refs[:4] + ([f"+{len(refs) - 4} more"] if len(refs) > 4 else [])
        rows.append({
            "mention_id": f"{args.id_prefix}-{n:03d}",
            "label": canonical,
            "entity_type": rec["entity_type"],
            "n_objects": len(rec["objects"]),
            "object_ids": " ".join(o for o, _, _ in rec["objects"]),
            "object_refs": " ; ".join(shown),
            "surface_variants": " ; ".join(v for v in rec["variants"] if v != canonical),
            "auto_flag": flags_for(canonical, rec["entity_type"]),
            "wikidata_qid": "", "alt_qids": "", "nil_reason": "", "notes": "",
        })

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} distinct mentions written to {args.out}")
    by_type = OrderedDict()
    for r in rows:
        by_type.setdefault(r["entity_type"], [0, 0])
        by_type[r["entity_type"]][0] += 1
        by_type[r["entity_type"]][1] += r["n_objects"]
    for t, (d, m) in by_type.items():
        print(f"  {t:<14} {d:>4} distinct  {m:>4} mentions")
    print(f"  {'TOTAL':<14} {sum(v[0] for v in by_type.values()):>4} distinct "
          f" {sum(v[1] for v in by_type.values()):>4} mentions")


if __name__ == "__main__":
    main()
