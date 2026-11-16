#!/usr/bin/env python3
"""Supplementary check — what the annotated type hint carries, measured without an LLM.

    python build_indices.py --similarity cosine_normalised   # once; indices are gitignored
    python check_type_hint.py

No LLM and no network: FAISS plus the local all-MiniLM-L6-v2, with the query template,
routing and AAT fallback of Experiment B (`run_retrieval.py`) reused, not re-implemented.

Why this exists
---------------
The first sweep (configurations `term_hint`, `combine_name`) took its hint from the
sheet's `raw_type_hint`. That column is not a model output: it is the object type
recorded in the test-set annotation, revised by hand towards vocabulary wording
(`Flyer` -> `leaflets`, `LP` -> `Vinyl recording`, `Event Programme` -> `concert
programs`). On 49 of the 50 strong-tier objects it is the accepted term's label up to
case, plural or one `|`-separated part (30 verbatim); on the 14 weak-tier objects it
never is. The deployed pipeline passes instead the `specific_type` that metadata
completion predicts, which the second sweep (`term_predhint`, `combine_name_predhint`)
uses. This script measures the difference without inference:

* `data/results/hint_check.csv` — hit@k, MRR@50 and T0 for the hint-prefixed query
  under each hint source (annotated; predicted by M5/M3/M1 in each of their runs; none).
* `data/results/hint_lookup.csv` — a retrieval-free baseline: the hint looked up by
  label in the routed vocabulary (case- and plural-insensitive, with the pipeline's AAT
  fallback; no match counts as an abstention), scored with the same outcome rules as
  T0/T1. With the predicted hint it is an "LLM without retrieval" baseline, since the
  predicted type is an LLM output produced without a candidate list; with the
  annotated hint it shows how much of the answer that hint contains.
* `data/results/hint_lookup_vs_t1.csv` — that baseline (predicted hint, the column
  the second sweep uses) against every T1 cell of the configurations that show the
  selector the predicted hint, paired per object with the McNemar test of `score.py`
  on majority-voted outcomes: overall correctness, and the accepted term found on the
  strong tier. The two can disagree: the lookup abstains whenever no label matches,
  which accuracy credits on every weak row but which never finds a term.

Objects without a prediction (the three held out as in-context examples in the
metadata-extraction evaluation) get no hint under the predicted sources, as in the
second sweep and in the pipeline. Rows are reported over all 65 objects and over the
62 that have a prediction.

T0 can differ from the first sweep's T0 by a couple of objects for the annotated hint:
some vocabulary terms share a label and an embedding (AAT has three `magazines`), and
which duplicate ranks first is a tie that a rebuilt index may break differently.
Compare variants within this script's output.
"""
import argparse
import contextlib
import csv
import io
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
sys.path.insert(0, str(SRC))      # bare import: lechoes/__init__ needs the serving deps
sys.path.insert(0, str(HERE))

from build_indices import index_stem                                     # noqa: E402
from gold import load_gold, score_outcome                                # noqa: E402
from run_retrieval import IndexSet, build_query, no_llm_key, rank_of_gold  # noqa: E402
from score import mcnemar, modal_correct                                 # noqa: E402

HINTS = HERE / "data" / "predicted_type_hints.csv"
OUT_RETRIEVAL = HERE / "data" / "results" / "hint_check.csv"
OUT_LOOKUP = HERE / "data" / "results" / "hint_lookup.csv"
OUT_VS_T1 = HERE / "data" / "results" / "hint_lookup_vs_t1.csv"
CORRECT = ("TP", "TA", "TA_weak")


def hint_variants(objects: list) -> dict:
    """{name: {object_id: hint}} for every hint source; '' means no hint."""
    rows = list(csv.DictReader(open(HINTS, newline="", encoding="utf-8")))
    variants = {"annotated": {o["object_id"]: o["raw_type_hint"] for o in objects},
                "none": {o["object_id"]: "" for o in objects}}
    for col in (c for c in rows[0] if c.startswith("predicted_")):
        variants[col.replace("predicted_", "predicted ")] = {
            r["object_id"]: r[col] for r in rows}      # '' where there is no prediction
    return variants


def retrieval_row(index_set, objects, hints, subset, k_grid):
    """hit@k (AAT fallback pool included), MRR at k_max and T0 for one hint source."""
    k_max = max(k_grid)
    hits = {k: 0 for k in k_grid}
    rr = 0.0
    reachable = t0_correct = n = 0
    for obj in objects:
        if obj["object_id"] not in subset:
            continue
        o = dict(obj, raw_type_hint=hints[obj["object_id"]])
        query, _ = build_query(o, "hint_description")
        with contextlib.redirect_stdout(io.StringIO()):
            pool = index_set.search(o["routed_vocab"], query, k_max)
            fallback = (index_set.search("AAT", query, k_max)
                        if o["routed_vocab"] != "AAT" else None)
        n += 1                     # T0 as in run_selection.py: routed top-1, no fallback
        t0_correct += score_outcome(o, pool[0]["uri"] if pool else None, False) in CORRECT
        gold = o["gold_uris"]
        if not gold:
            continue
        reachable += 1
        for k in k_grid:
            hits[k] += bool(rank_of_gold(pool[:k], gold)
                            or (fallback and rank_of_gold(fallback[:k], gold)))
        ranks = [r for r in (rank_of_gold(pool, gold),
                             rank_of_gold(fallback, gold) if fallback else None) if r]
        rr += 1 / min(ranks) if ranks else 0.0
    return ({k: hits[k] / reachable for k in k_grid}, rr / reachable,
            t0_correct / n, reachable, n)


def label_forms(text: str) -> set:
    text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())).strip()
    return {text, text[:-1] if text.endswith("s") else text + "s"} if text else set()


def label_index(indices: Path, similarity: str) -> dict:
    """vocabulary -> {label form: [uri, ...]}, in index order, from the index metadata."""
    out = {}
    for vocab in ("AAT", "MIMO", "Europeana", "PBCore"):
        meta = pickle.load(open(indices / f"{index_stem(vocab, similarity, 'term')}_metadata.pkl",
                                "rb"))
        table = {}
        for _, row in meta.iterrows():
            for form in label_forms(row["term"]):
                table.setdefault(form, []).append(row["uri"])
        out[vocab] = table
    return out


def lookup_uris(labels, obj, hint: str) -> list:
    """Every term whose label matches the hint, routed vocabulary first, then AAT."""
    parts = [p for p in hint.split("|") if p.strip()]
    uris = [u for p in parts for f in label_forms(p)
            for u in labels[obj["routed_vocab"]].get(f, [])]
    if not uris and obj["routed_vocab"] != "AAT":              # the pipeline's fallback
        uris = [u for p in parts for f in label_forms(p)
                for u in labels["AAT"].get(f, [])]
    return uris


def lookup_row(labels, objects, hints, subset):
    """Accuracy of the hint looked up by label: first match, and any matching duplicate."""
    first_ok = any_ok = n = 0
    for obj in objects:
        if obj["object_id"] not in subset:
            continue
        n += 1
        uris = lookup_uris(labels, obj, hints[obj["object_id"]])
        first = uris[0] if uris else None
        first_ok += score_outcome(obj, first, first is None) in CORRECT
        hit = next((u for u in uris if u in obj["gold_uris"]), first)
        any_ok += score_outcome(obj, hit, hit is None) in CORRECT
    return first_ok / n, any_ok / n, n


def lookup_vs_t1(labels, objects, hints, configurations, k_grid, model_slug) -> list:
    """
    The first-match lookup against each T1 cell, paired per object. Each object is
    one pair: T1's outcome is its majority over runs (as in every test of score.py).
    """
    lookup = {}
    for obj in objects:
        uris = lookup_uris(labels, obj, hints[obj["object_id"]])
        lookup[obj["object_id"]] = score_outcome(obj, uris[0] if uris else None, not uris)
    strong = {o["object_id"] for o in objects if o["tier"] == "strong"}
    lk_correct = {o: v in CORRECT for o, v in lookup.items()}
    lk_tp = {o: lookup[o] == "TP" for o in strong}
    rows = []
    for config in configurations:
        for k in k_grid:
            files = sorted((HERE / "data" / "predictions").glob(
                f"selection_T1_{config}_{model_slug}_k{k}_run*.jsonl"))
            if not files:
                continue
            records = [json.loads(line) for f in files for line in open(f, encoding="utf-8")]
            t1 = modal_correct(records)
            tp_runs = defaultdict(list)
            for r in records:
                tp_runs[r["object_id"]].append(r["outcome"] == "TP")
            t1_tp = {o: sum(v) * 2 > len(v) for o, v in tp_runs.items() if o in strong}
            overall, exact = mcnemar(lk_correct, t1), mcnemar(lk_tp, t1_tp)
            rows.append({
                "configuration": config, "k": k, "runs": len(files),
                "t1_accuracy_majority": round(sum(t1.values()) / len(t1), 4),
                "lookup_accuracy": round(sum(lk_correct.values()) / len(lk_correct), 4),
                "t1_only": overall.get("b_correct_only", 0),
                "lookup_only": overall.get("a_correct_only", 0),
                "p_value": round(overall["p_value"], 4),
                "strong_n": len(strong),
                "strong_t1_accepted_term": sum(t1_tp.values()),
                "strong_lookup_accepted_term": sum(lk_tp.values()),
                "strong_t1_only": exact.get("b_correct_only", 0),
                "strong_lookup_only": exact.get("a_correct_only", 0),
                "strong_p_value": round(exact["p_value"], 4)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    r_cfg = cfg["retrieval"]
    similarity = r_cfg["similarity"][0]
    k_grid = sorted(r_cfg["k_grid"])
    indices = HERE / cfg["paths"]["indices"]
    if not any(indices.glob(f"*_{similarity}_term.index")):
        print(f"no {similarity} indices under {indices}: run "
              f"`python build_indices.py --similarity {similarity}` first")
        return 1

    objects = load_gold(HERE / cfg["paths"]["annotations"])
    variants = hint_variants(objects)
    with_prediction = {r["object_id"] for r in csv.DictReader(open(HINTS, encoding="utf-8"))
                       if r["metadata_extraction_object_id"]}
    subsets = {f"all {len(objects)}": {o["object_id"] for o in objects},
               f"{len(with_prediction)} with a prediction": with_prediction}

    retrieval_rows = []
    for strategy in ("term", "combine"):
        with no_llm_key(), contextlib.redirect_stdout(io.StringIO()):
            index_set = IndexSet(indices, similarity, strategy, r_cfg["embedding_model"])
        for subset_name, subset in subsets.items():
            print(f"\n[{strategy} index, {subset_name}]  hint + description")
            print(f"  {'hint':22s} " + " ".join(f"@{k:<4d}" for k in k_grid)
                  + "  MRR@50  T0")
            for name, hints in variants.items():
                hit, mrr, t0, reachable, n = retrieval_row(index_set, objects, hints,
                                                           subset, k_grid)
                print(f"  {name:22s} " + " ".join(f"{100 * hit[k]:5.1f}" for k in k_grid)
                      + f"  {mrr:.3f}  {100 * t0:5.1f}")
                retrieval_rows.append({
                    "index_strategy": strategy, "objects": subset_name, "hint": name,
                    "n_reachable": reachable, "n": n,
                    **{f"hit@{k}": round(hit[k], 4) for k in k_grid},
                    "mrr@50": round(mrr, 4), "t0_accuracy": round(t0, 4)})

    labels = label_index(indices, similarity)
    lookup_rows = []
    print("\n[label lookup, no retrieval]  accuracy (first match / any duplicate)")
    for subset_name, subset in subsets.items():
        for name, hints in variants.items():
            first, anyd, n = lookup_row(labels, objects, hints, subset)
            print(f"  {subset_name:22s} {name:22s} {100 * first:5.1f} / {100 * anyd:5.1f}")
            lookup_rows.append({"objects": subset_name, "hint": name, "n": n,
                                "accuracy_first_match": round(first, 4),
                                "accuracy_any_duplicate": round(anyd, 4)})

    # The lookup of the hint the second sweep shows the selector, against its T1 cells.
    from models import load_models, reference_model
    hint_name = cfg["type_hint"]["predicted"]["column"].replace("predicted_", "predicted ")
    predicted_configs = [c["id"] for c in cfg["selection"]["configurations"]
                         if c.get("prompt_hint") == "predicted"]
    vs_rows = lookup_vs_t1(labels, objects, variants[hint_name], predicted_configs,
                           sorted(cfg["selection"]["k_grid"]),
                           reference_model(load_models(Path(args.config)))["slug"])
    print(f"\n[label lookup ({hint_name}) vs T1, paired; majority over runs]")
    for r in vs_rows:
        print(f"  {r['configuration']:22s} k={r['k']:<3d} overall T1-only {r['t1_only']:2d} / "
              f"lookup-only {r['lookup_only']:2d} p={r['p_value']:.3f} | accepted term on "
              f"{r['strong_n']} strong: T1 {r['strong_t1_accepted_term']} vs lookup "
              f"{r['strong_lookup_accepted_term']} ({r['strong_t1_only']}/"
              f"{r['strong_lookup_only']}, p={r['strong_p_value']:.4f})")

    OUT_RETRIEVAL.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((OUT_RETRIEVAL, retrieval_rows), (OUT_LOOKUP, lookup_rows),
                       (OUT_VS_T1, vs_rows)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
