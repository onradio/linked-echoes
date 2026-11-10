#!/usr/bin/env python3
"""Build the object-level annotation sheet for the classification gold standard.

One row per museum object, single-label with an accepted alternative: each object has
a primary category and, for the rare composite records, a secondary one. Type
classification uses the primary category and therefore a single primary type, with
`type_alt_uris` holding any equally defensible alternatives.

Data-quality warnings are printed to the console at build time; they are not written
into the sheet.
"""
import argparse
import csv
import re
import unicodedata
from collections import defaultdict

CATEGORIES = [
    "Musical Instrument", "Clothing", "Audiovisual Media",
    "Textual & Graphic Work", "Merchandise", "Personal Effects",
    "Award & Recognition",
]

VOCAB_MAP = {
    "Musical Instrument": "MIMO",
    "Clothing": "Europeana",
    "Audiovisual Media": "PBCore",
    "Textual & Graphic Work": "AAT",
    "Merchandise": "AAT",
    "Personal Effects": "AAT",
    "Award & Recognition": "AAT",
}


def norm(v: str) -> str:
    v = unicodedata.normalize("NFKC", v or "")
    v = v.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", v).strip()


def split_multi(v: str):
    return [p for p in (norm(x) for x in (v or "").split("|")) if p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("objects_csv")
    ap.add_argument("-o", "--out", default="classification_annotation_sheet.csv")
    args = ap.parse_args()

    with open(args.objects_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    type_to_cats = defaultdict(set)
    for r in rows:
        t = norm(r.get("type"))
        if t:
            type_to_cats[t.casefold()].add(norm(r.get("category")))

    out_rows, warnings = [], []
    for i, r in enumerate(rows, start=1):
        oid = f"OBJ-{i:03d}"
        cats = split_multi(r.get("category"))
        raw_type = norm(r.get("type"))
        primary = cats[0] if cats else ""
        secondary = cats[1] if len(cats) > 1 else ""

        if not cats:
            warnings.append(f"{oid}: no category")
        for c in cats:
            if c not in CATEGORIES:
                warnings.append(f"{oid}: unknown category '{c}'")
        if len(cats) > 2:
            warnings.append(f"{oid}: {len(cats)} categories - only primary and secondary are kept")
        if not raw_type:
            warnings.append(f"{oid}: no type hint")
        if raw_type.endswith("?"):
            warnings.append(f"{oid}: uncertain type hint '{raw_type}'")
        if "|" in (r.get("type") or ""):
            warnings.append(f"{oid}: composite object '{raw_type}' - pick the primary type, "
                            f"put the other in type_alt_uris")
        if raw_type and len(type_to_cats[raw_type.casefold()]) > 1:
            warnings.append(f"{oid}: type hint '{raw_type}' used under >1 category - check")

        out_rows.append({
            "object_id": oid,
            "case_number": norm(r.get("case_number")),
            "name": norm(r.get("name")) or norm(r.get("description")),
            "description": norm(r.get("description")),
            "caption": norm(r.get("caption")),
            "category_gold_primary": primary,
            "category_gold_secondary": secondary,
            "raw_type_hint": raw_type,
            "vocab_hint": VOCAB_MAP.get(primary, ""),
            "type_gold_uri": "",
            "type_gold_label": "",
            "type_alt_uris": "",
            "type_nil_reason": "",
            "notes": "",
        })

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"{len(out_rows)} objects written to {args.out}\n")

    prim, sec = defaultdict(int), defaultdict(int)
    for r in out_rows:
        prim[r["category_gold_primary"]] += 1
        if r["category_gold_secondary"]:
            sec[r["category_gold_secondary"]] += 1
    print(f"{'category':<26}{'primary':>8}{'secondary':>11}")
    for c in CATEGORIES:
        note = "   <-- unattested" if prim.get(c, 0) == 0 and sec.get(c, 0) == 0 else ""
        print(f"{c:<26}{prim.get(c, 0):>8}{sec.get(c, 0):>11}{note}")
    n_sec = sum(sec.values())
    print(f"\nobjects with a secondary category: {n_sec} "
          f"({100 * n_sec / len(out_rows):.1f}%)")

    vocabs = defaultdict(int)
    for r in out_rows:
        vocabs[r["vocab_hint"]] += 1
    print("objects routed per vocabulary: " +
          ", ".join(f"{k or '(none)'} {v}" for k, v in sorted(vocabs.items())))

    cases = defaultdict(int)
    for r in out_rows:
        cases[r["case_number"]] += 1
    print(f"cases: {len(cases)} | objects per case: {sorted(cases.values())}")

    if warnings:
        print(f"\n{len(warnings)} data-quality warnings (not written to the sheet):")
        for w_ in warnings:
            print(f"  {w_}")


if __name__ == "__main__":
    main()
