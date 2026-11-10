#!/usr/bin/env python3
"""Stage 3b — the adjudication sheet, built only because §4.5's conditional fired.

    python build_adjudication_sheet.py            # build data/adjudication_sheet.csv
    python build_adjudication_sheet.py --check    # is adjudication required at all?

§4.5 makes human adjudication **conditional on the diagnosis**, and the diagnosis said yes:
`max_s hallucination_share(s)` is above `fp_diagnosis.hallucination_share_threshold`, so the
ambiguous class is large enough to matter. This script produces the sample the annotators
judge; it refuses to run if the conditional has not fired, because building the sheet anyway
would invite the study the spec deliberately made optional.

What is adjudicated is **only the hallucination pool** — not all false positives. The other
classes are settled by the gold itself: a `misattribution` is recorded against a sibling, a
`wrong_field` is recorded under another field, a `normalisation` case is a matching artefact.
Only a value occurring nowhere in the evidence is genuinely ambiguous between *wrong* and
*true but unrecorded*, and that is the only thing worth a human's time.

The pool is deduplicated on `(object_id, field, normalised_value)`, so one judgement serves
every system that made the same proposal and the correction cannot reorder the systems
arbitrarily. `proposed_by` records which systems proposed it, for audit — it is **not** shown
to the annotators, who must not be able to infer which system they are marking.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

import gold as G
import predictions as P
import score as S

HERE = Path(__file__).resolve().parent


def hallucination_pool(gold_obj: G.Gold, config: dict, manifest: dict) -> dict:
    """
    Every `(object_id, field, value)` any system proposed and the attribution classed
    `hallucination`, with the systems that proposed it.
    """
    policies = config["matching"]["policy"]
    pool: dict = collections.defaultdict(set)

    for entry in manifest["entries"].values():
        path = P.prediction_path(config, entry["system"], entry["model_slug"],
                                 entry["run"])
        if not path.exists():
            continue
        scope = entry["evidence_scope"]
        for record in P.read(path):
            if record["predicted"] is None:
                continue
            oid = record["object_id"]
            for field in G.SCORED:
                policy = policies[field]
                preds, _ = S.prepare_predicted(record, field, config)
                golds = sorted(gold_obj.gold_values(oid, field))
                _, fps, _ = S.match_cell(field, preds, golds, policy)
                for value in fps:
                    if S.attribute_fp(gold_obj, oid, field, value, scope,
                                      policy) == "hallucination":
                        pool[(oid, field, value)].add(entry["system"])
    return pool


def stratified_sample(pool: dict, config: dict) -> list:
    """
    Up to `sample_size` rows, stratified by field with proportional allocation and a floor
    of `min_per_field` for any field that has any; fields with fewer are taken whole. The
    seed is in `config.yaml` and both it and the drawn sample go to disk (§4.5).
    """
    cfg = config["adjudication"]
    size, floor = int(cfg["sample_size"]), int(cfg["min_per_field"])
    rng = np.random.default_rng(int(cfg["seed"]))

    by_field: dict = collections.defaultdict(list)
    for key in sorted(pool):
        by_field[key[1]].append(key)

    total = sum(len(v) for v in by_field.values())
    if total <= size:
        chosen = [k for keys in by_field.values() for k in keys]
        return sorted(chosen)

    allocation = {}
    for field, keys in by_field.items():
        allocation[field] = min(len(keys), max(floor, round(size * len(keys) / total)))
    # Proportional allocation plus a floor can overshoot; trim from the largest strata.
    while sum(allocation.values()) > size:
        biggest = max(allocation, key=lambda f: (allocation[f], f))
        if allocation[biggest] <= floor:
            break
        allocation[biggest] -= 1

    chosen = []
    for field, keys in sorted(by_field.items()):
        take = allocation[field]
        idx = rng.choice(len(keys), size=take, replace=False) if take < len(keys) else \
            range(len(keys))
        chosen.extend(keys[i] for i in idx)
    return sorted(chosen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="report whether adjudication is required and exit")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing sheet")
    args = ap.parse_args()

    config = G.CONFIG
    gold_obj = G.Gold()
    results = P.paths(config)["results"]
    stats = json.loads((results / "stats.json").read_text(encoding="utf-8"))
    diagnosis = stats["fp_diagnosis"]

    print(f"max hallucination share {diagnosis['max_hallucination_share']:.3f} "
          f"({diagnosis['max_system']}) against threshold "
          f"{diagnosis['hallucination_share_threshold']}")
    if not diagnosis["adjudication_required"]:
        print("adjudication is NOT required — §4.5 reports the decomposition and stops. "
              "No sheet is built.")
        return 0
    print("adjudication IS required (§4.5)")
    if args.check:
        return 0

    out = HERE / "data" / "adjudication_sheet.csv"
    if out.exists() and not args.force:
        print(f"{out.name} exists — use --force to rebuild "
              f"(rebuilding after judging would discard the verdicts)")
        return 0

    manifest = json.loads(
        (P.paths(config)["predictions"] / "manifest.json").read_text(encoding="utf-8"))
    pool = hallucination_pool(gold_obj, config, manifest)
    by_field = collections.Counter(field for _, field, _ in pool)
    print(f"\nhallucination pool: {len(pool)} distinct (object, field, value) triples")
    for field, n in by_field.most_common():
        print(f"  {field:15s} {n:4d}")

    sample = stratified_sample(pool, config)
    rows = []
    for i, (oid, field, value) in enumerate(sample, 1):
        obj = gold_obj.by_id[oid]
        rows.append({
            "adjudication_id": f"ADJ-{i:04d}",
            "object_id": oid,
            "field": field,
            "value": value,
            "item_description": obj["item_description"],
            "caption": obj["caption"],
            # Audit only. Never shown to the annotators: knowing which systems proposed a
            # value is exactly the information that would bias the judgement.
            "proposed_by": "|".join(sorted(pool[(oid, field, value)])),
            "verdict": "",
            "notes": "",
        })

    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (HERE / "data" / "adjudication_sample.json").write_text(json.dumps({
        "seed": config["adjudication"]["seed"],
        "sample_size": config["adjudication"]["sample_size"],
        "min_per_field": config["adjudication"]["min_per_field"],
        "pool_size": len(pool),
        "drawn": len(rows),
        "by_field": dict(collections.Counter(r["field"] for r in rows)),
        "keys": [[oid, field, value] for oid, field, value in sample],
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\ndrew {len(rows)} rows -> {out.name} "
          f"(seed {config['adjudication']['seed']}, sample recorded in "
          f"adjudication_sample.json)")
    print("verdicts: correct | incorrect | unverifiable — `correct` means true of the "
          "object and missing from the record.")
    print("Next: three annotators fill `verdict` to a single agreed value, save as "
          "adjudication_sheet_done.csv, then score.py estimates adjudicated precision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
