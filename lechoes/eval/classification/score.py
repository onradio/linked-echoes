#!/usr/bin/env python3
"""Stage 3 — score every prediction on disk (spec §4). No numbers are computed elsewhere.

    python score.py                 # everything that has predictions
    python score.py --no-bootstrap  # skip the CIs (fast, for iterating)

Reads `data/predictions/` plus the gold sheet and writes `data/results/`:

    selection_cells.csv     one row per (configuration, system, k): the §4.2 metrics
    selection_by_tier.csv   the same, split strong / weak / nil
    selection_by_vocab.csv  the same, split by routed vocabulary
    attribution.csv         §4.2.6 error attribution, one cause per object
    retrieval_hits.csv      hit@k per configuration, system-independent (§3.2)
    category_cells.csv      Experiment A, when its predictions exist
    stats.json              McNemar tests, headline numbers, run-to-run variance

Two conventions, both consequences of the gold design rather than of taste:

* **Runs are averaged for reporting and majority-voted for paired tests.** Three runs at
  temperature 0 differ by one or two objects; averaging is the honest summary, but McNemar
  needs one binary outcome per object, so the modal outcome across runs is used and the
  disagreement rate is reported beside it.
* **Bootstrap resamples display cases, not objects.** The sample was drawn 3 per case and
  objects within a case share text almost verbatim — the three Suede lyrics sheets are not
  independent draws, so an object-level bootstrap would understate the intervals.
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRED = HERE / "data" / "predictions"
RESULTS = HERE / "data" / "results"

# Outcomes that count as a correct decision (§4.2.1).
CORRECT = {"TP", "TA", "TA_weak"}
BOOTSTRAP_RESAMPLES = 10000
SEED = 20260812


def read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def safe_div(num: float, den: float):
    return num / den if den else None


def f1(precision, recall):
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------
# Metrics over a set of prediction records
# --------------------------------------------------------------------------

def metrics(records: list) -> dict:
    """
    The §4.2 metric family for one collection of records.

    `records` may span several runs; every count is divided by the number of
    runs, so the result reads as "per run" and is comparable to a single cell.
    """
    runs = len({r["run"] for r in records}) or 1
    # Records per run, NOT the number of distinct object ids: a bootstrap
    # resample draws cases with replacement, so the same object legitimately
    # appears several times and the denominator has to count it each time.
    # Counting distinct ids instead inflates every rate — enough to push the
    # accuracy CI above 100%.
    n_objects = len(records) / runs
    c = Counter(r["outcome"] for r in records)
    per_run = {k: v / runs for k, v in c.items()}
    g = lambda key: per_run.get(key, 0.0)

    tp, wt, fa, ta, taw, sa = (g("TP"), g("WT"), g("FA"), g("TA"),
                               g("TA_weak"), g("SA"))
    err = g("error_malformed")

    # Weak-row assertions that missed: the abstention-accuracy denominator needs
    # them, since asserting a third term where only an approximation exists is
    # exactly the failure the STRICT RULE is meant to prevent.
    wt_weak = sum(1 for r in records
                  if r["outcome"] == "WT" and r["tier"] == "weak") / runs

    precision = safe_div(tp, tp + wt + sa)
    recall = safe_div(tp, tp + wt + fa)          # TA_weak deliberately excluded (§4.2.3)

    strong = [r for r in records if r["tier"] == "strong" and r["gold_reachable"]]
    s = Counter(r["outcome"] for r in strong)
    s_tp, s_wt, s_fa = s["TP"] / runs, s["WT"] / runs, s["FA"] / runs
    p_strong = safe_div(s_tp, s_tp + s_wt)
    r_strong = safe_div(s_tp, s_tp + s_wt + s_fa)

    return {
        "n_objects": n_objects, "runs": runs,
        "TP": tp, "WT": wt, "FA": fa, "TA": ta, "TA_weak": taw, "SA": sa,
        "error_malformed": err,
        "accuracy": safe_div(tp + ta + taw, n_objects),
        "term_precision": precision,
        "term_recall": recall,
        "term_f1": f1(precision, recall),
        "precision_strong": p_strong,
        "recall_strong": r_strong,
        "f1_strong": f1(p_strong, r_strong),
        "abstention_accuracy": safe_div(ta + taw, ta + taw + sa + wt_weak),
        "abstention_denominator": ta + taw + sa + wt_weak,
    }


def bootstrap_ci(records: list, metric_key: str, resamples: int) -> tuple:
    """
    Percentile CI for one metric, resampling display cases with replacement.

    Returns (low, high), or (None, None) when the metric is undefined on enough
    resamples to make an interval meaningless.
    """
    by_case = defaultdict(list)
    for r in records:
        by_case[r["case_number"]].append(r)
    cases = list(by_case)
    if len(cases) < 2:
        return None, None

    rng = random.Random(SEED)
    values = []
    for _ in range(resamples):
        drawn = []
        for _ in range(len(cases)):
            drawn.extend(by_case[rng.choice(cases)])
        value = metrics(drawn).get(metric_key)
        if value is not None:
            values.append(value)
    if len(values) < resamples * 0.5:
        return None, None
    values.sort()
    return (values[int(0.025 * len(values))], values[int(0.975 * len(values))])


# --------------------------------------------------------------------------
# Paired testing
# --------------------------------------------------------------------------

def modal_correct(records: list) -> dict:
    """object_id -> whether the object was decided correctly in most runs."""
    per_object = defaultdict(list)
    for r in records:
        per_object[r["object_id"]].append(r["outcome"] in CORRECT)
    return {oid: sum(v) * 2 > len(v) for oid, v in per_object.items()}


def mcnemar(a: dict, b: dict) -> dict:
    """
    Exact McNemar on the discordant pairs, a two-sided binomial test at p=0.5.

    Reported with the discordant counts, which carry more information than the
    p-value on a sample this size.
    """
    shared = set(a) & set(b)
    b01 = sum(1 for o in shared if not a[o] and b[o])     # b right, a wrong
    b10 = sum(1 for o in shared if a[o] and not b[o])     # a right, b wrong
    n = b01 + b10
    if n == 0:
        return {"n_discordant": 0, "a_only": 0, "b_only": 0, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(b01, b10) + 1)) / (2 ** n)
    return {"n_discordant": n, "a_correct_only": b10, "b_correct_only": b01,
            "p_value": min(1.0, 2 * tail)}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

NO_MODEL = "-"      # systems that call no LLM (T0, C0, C1)


def model_of(record: dict) -> str:
    """The model slug a record belongs to, or NO_MODEL for the LLM-free systems."""
    return record.get("model_slug") or NO_MODEL


def reference_slug(models_seen: list) -> str | None:
    """
    The model every cross-model comparison is taken against: the first entry of
    `config.yaml`'s model list, which is the one the pipeline deploys.

    Read from the configuration rather than inferred from the data, so that
    adding a model — or scoring a partial sweep that happens to lack the
    reference — cannot silently move the baseline.
    """
    try:
        from models import load_models, reference_model
        configured = reference_model(load_models())["slug"]
    except Exception:                      # config unavailable: fall back
        return models_seen[0] if models_seen else None
    if configured not in models_seen:
        return None
    return configured


def load_selection() -> dict:
    """
    (configuration, system, k, model) -> records, pooled across runs.

    The model is part of the key: without it a multi-model sweep would pool
    every model into one cell and report their mean as though it were a system.
    """
    cells = defaultdict(list)
    for path in sorted(PRED.glob("selection_*.jsonl")):
        for r in read_jsonl(path):
            cells[(r["configuration"], r["system"], r["k"], model_of(r))].append(r)
    return cells


def load_retrieval() -> dict:
    """(index_strategy, query, k) -> records."""
    cells = defaultdict(list)
    for path in sorted(PRED.glob("retrieval_*.jsonl")):
        for r in read_jsonl(path):
            cells[(r["index_strategy"], r["query"], r["k"])].append(r)
    return cells


CATEGORIES = [
    "Musical Instrument", "Clothing", "Audiovisual Media",
    "Textual & Graphic Work", "Merchandise", "Personal Effects",
    "Award & Recognition",
]


def category_metrics(records: list) -> dict:
    """
    §4.1 for one system: accuracy, strict accuracy and macro-F1.

    Per-class figures use the **primary** gold only, so that support sums to the
    sample and the confusion matrix and the class table agree. Accuracy is the
    lenient one, crediting a prediction that matches either gold.
    """
    runs = len({r["run"] for r in records}) or 1
    n = len(records) / runs
    per_class = {}
    for c in CATEGORIES:
        tp = sum(1 for r in records if r["predicted"] == c and r["gold_primary"] == c)
        predicted = sum(1 for r in records if r["predicted"] == c)
        support = sum(1 for r in records if r["gold_primary"] == c)
        p = safe_div(tp, predicted)
        rec = safe_div(tp, support)
        per_class[c] = {"support": support / runs, "predicted": predicted / runs,
                        "precision": p, "recall": rec, "f1": f1(p, rec)}

    attested = [c for c in CATEGORIES if per_class[c]["support"] > 0]

    def macro(key):
        # Undefined cells count as 0, the usual zero_division convention: a
        # class the system never predicts has no precision to average, and
        # silently skipping it would reward never predicting a hard class.
        values = [per_class[c][key] or 0.0 for c in attested]
        return sum(values) / len(values) if values else None

    # Predictions that land in a class with no gold instances. Macro-averaging
    # over attested classes alone cannot see these — they cost the true class
    # its recall, but their own precision term is outside the average — so the
    # count is reported beside it rather than folded in.
    unattested = sum(per_class[c]["predicted"] for c in CATEGORIES
                     if per_class[c]["support"] == 0)

    return {
        "n_objects": n, "runs": runs,
        "accuracy": sum(1 for r in records if r["correct"]) / (n * runs),
        "accuracy_strict": sum(1 for r in records if r["correct_strict"]) / (n * runs),
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_f1": macro("f1"),
        "n_attested_classes": len(attested),
        "predictions_in_unattested": unattested,
        "per_class": per_class,
    }


def score_category(stats: dict, args) -> int:
    """Scores Experiment A if its predictions exist; returns the systems scored."""
    files = sorted(PRED.glob("category_*.jsonl"))
    if not files:
        print("category: no predictions yet (Experiment A not run)")
        return 0

    by_system = defaultdict(list)
    for path in files:
        for r in read_jsonl(path):
            # A system is (id, model): C2 under two models is two systems, and
            # pooling them would report their mean as though it were one.
            by_system[(r["system"], model_of(r))].append(r)

    rows, class_rows, confusion_rows = [], [], []
    for (system, model), records in sorted(by_system.items()):
        m = category_metrics(records)
        row = {"system": system, "model": model,
               "model_label": records[0].get("model_label") or "",
               "n_objects": m["n_objects"], "runs": m["runs"],
               "accuracy": m["accuracy"], "accuracy_strict": m["accuracy_strict"],
               "macro_precision": m["macro_precision"],
               "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"],
               "n_attested_classes": m["n_attested_classes"],
               "predictions_in_unattested": m["predictions_in_unattested"]}
        if not args.no_bootstrap:
            lo, hi = bootstrap_ci_generic(records, args.resamples,
                                          lambda rs: category_metrics(rs)["accuracy"])
            row["accuracy_lo"], row["accuracy_hi"] = lo, hi
        rows.append(row)

        for c, v in m["per_class"].items():
            class_rows.append({"system": system, "model": model,
                               "category": c, **v})
        # Confusion counts summed over runs; `runs` is on the row so a reader
        # can normalise. Rows are the primary gold, columns the prediction.
        counts = Counter((r["gold_primary"], r["predicted"]) for r in records)
        for gold in CATEGORIES:
            entry = {"system": system, "model": model, "runs": m["runs"],
                     "gold_primary": gold}
            for pred in CATEGORIES:
                entry[pred] = counts.get((gold, pred), 0)
            confusion_rows.append(entry)

    write_csv(RESULTS / "category_cells.csv", rows,
              ["system", "model", "model_label", "n_objects", "runs", "accuracy", "accuracy_lo",
               "accuracy_hi", "accuracy_strict", "macro_precision",
               "macro_recall", "macro_f1", "n_attested_classes",
               "predictions_in_unattested"])
    write_csv(RESULTS / "category_per_class.csv", class_rows,
              ["system", "model", "category", "support", "predicted", "precision",
               "recall", "f1"])
    write_csv(RESULTS / "category_confusion.csv", confusion_rows,
              ["system", "model", "runs", "gold_primary"] + CATEGORIES)

    # C2 is the system. Each model's C2 is compared against the LLM-free
    # baselines and against the reference model's C2. Every model pair would
    # multiply tests without a matching claim.
    reference = reference_slug([m for (s_, m) in by_system if s_ == "C2"])
    for (system, model), records in sorted(by_system.items()):
        if system != "C2":
            continue
        c2 = modal_correct_flag(records)
        for (other_system, other_model), other in sorted(by_system.items()):
            if (other_system, other_model) == (system, model):
                continue
            if other_system == "C2" and (other_model != reference
                                         or model == reference):
                continue
            # Keep the system id in the label: C2 and C2-scoped under the same
            # model would otherwise render identically.
            label = (other_system if other_model == NO_MODEL
                     else f"{other_system} [{other_model}]")
            stats["mcnemar"].append({
                "configuration": "category", "k": None, "model": model,
                "comparison": f"C2 [{model}] vs {label}",
                "a": label, "b": f"C2 [{model}]",
                **mcnemar(modal_correct_flag(other), c2)})
    print(f"category: {len(rows)} systems scored "
          f"({len({m for _, m in by_system if m != NO_MODEL})} model(s))")
    return len(rows)


def modal_correct_flag(records: list) -> dict:
    """object_id -> correct in most runs, using the `correct` flag."""
    per_object = defaultdict(list)
    for r in records:
        per_object[r["object_id"]].append(bool(r["correct"]))
    return {oid: sum(v) * 2 > len(v) for oid, v in per_object.items()}


def bootstrap_ci_generic(records: list, resamples: int, fn) -> tuple:
    """Case-clustered percentile CI for an arbitrary metric function."""
    by_case = defaultdict(list)
    for r in records:
        by_case[r["case_number"]].append(r)
    cases = list(by_case)
    if len(cases) < 2:
        return None, None
    rng = random.Random(SEED)
    values = []
    for _ in range(resamples):
        drawn = []
        for _ in range(len(cases)):
            drawn.extend(by_case[rng.choice(cases)])
        value = fn(drawn)
        if value is not None:
            values.append(value)
    if not values:
        return None, None
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values))]


def write_csv(path: Path, rows: list, columns: list) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: (round(row[c], 4) if isinstance(row.get(c), float)
                            else row.get(c)) for c in columns})


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="Skip confidence intervals")
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    selection = load_selection()
    retrieval = load_retrieval()
    if not selection:
        print("no selection predictions found — run run_selection.py first")
        return 1

    # ---- retrieval ceiling ------------------------------------------------
    retrieval_rows = []
    for (strategy, query, k), records in sorted(retrieval.items()):
        scored = [r for r in records if r["hit"] is not None]
        retrieval_rows.append({
            "index_strategy": strategy, "query": query, "k": k,
            "n_scored": len(scored),
            "hit_routed": safe_div(sum(1 for r in scored if r["hit"]), len(scored)),
            "hit_with_fallback": safe_div(
                sum(1 for r in scored if r["hit_with_fallback"]), len(scored)),
            "mrr": safe_div(sum(1 / r["gold_rank_with_fallback"] for r in scored
                                if r["gold_rank_with_fallback"]), len(scored)),
        })
    write_csv(RESULTS / "retrieval_hits.csv", retrieval_rows,
              ["index_strategy", "query", "k", "n_scored", "hit_routed",
               "hit_with_fallback", "mrr"])

    # hit@k lookup for the selection table and the attribution
    hit_at = {(r["index_strategy"], r["query"], r["k"]): r["hit_with_fallback"]
              for r in retrieval_rows}

    # ---- selection cells --------------------------------------------------
    cell_rows, tier_rows, vocab_rows = [], [], []
    for (config, system, k, model), records in sorted(selection.items()):
        m = metrics(records)
        strategy = records[0]["index_strategy"]
        query = records[0]["query"]
        row = {"configuration": config, "system": system, "k": k, "model": model,
               "model_label": records[0].get("model_label") or "",
               "index_strategy": strategy, "query": query,
               "hit_at_k": hit_at.get((strategy, query, k)), **m}
        if not args.no_bootstrap:
            for key in ("accuracy", "term_precision", "term_recall", "term_f1"):
                lo, hi = bootstrap_ci(records, key, args.resamples)
                row[f"{key}_lo"], row[f"{key}_hi"] = lo, hi
        cell_rows.append(row)

        for tier in ("strong", "weak", "nil"):
            subset = [r for r in records if r["tier"] == tier]
            if subset:
                tier_rows.append({"configuration": config, "system": system,
                                  "k": k, "model": model, "tier": tier,
                                  **metrics(subset)})
        for vocab in sorted({r["vocab"] for r in records}):
            subset = [r for r in records if r["vocab"] == vocab]
            vocab_rows.append({"configuration": config, "system": system, "k": k,
                               "model": model, "vocab": vocab, **metrics(subset)})

    cell_columns = ["configuration", "system", "k", "model", "model_label",
                    "index_strategy", "query",
                    "n_objects", "runs", "TP", "WT", "FA", "TA", "TA_weak", "SA",
                    "error_malformed", "accuracy", "accuracy_lo", "accuracy_hi",
                    "term_precision", "term_precision_lo", "term_precision_hi",
                    "term_recall", "term_recall_lo", "term_recall_hi",
                    "term_f1", "term_f1_lo", "term_f1_hi",
                    "precision_strong", "recall_strong", "f1_strong",
                    "abstention_accuracy", "abstention_denominator", "hit_at_k"]
    write_csv(RESULTS / "selection_cells.csv", cell_rows, cell_columns)
    write_csv(RESULTS / "selection_by_tier.csv", tier_rows,
              ["configuration", "system", "k", "model", "tier", "n_objects", "TP", "WT",
               "FA", "TA", "TA_weak", "SA", "accuracy", "term_precision"])
    write_csv(RESULTS / "selection_by_vocab.csv", vocab_rows,
              ["configuration", "system", "k", "model", "vocab", "n_objects", "TP", "WT",
               "FA", "TA", "TA_weak", "SA", "accuracy", "term_precision"])

    # ---- error attribution (§4.2.6) ---------------------------------------
    # One cause per object, in precedence order. `retrieval_miss` needs to know
    # whether the gold was in the pool the selector actually saw, which is the
    # matching Experiment B record for the same configuration and k.
    retrieved_gold = {}
    for (strategy, query, k), records in retrieval.items():
        for r in records:
            retrieved_gold[(strategy, query, k, r["object_id"])] = r["hit_with_fallback"]

    attribution_rows = []
    for (config, system, k, model), records in sorted(selection.items()):
        if system != "T1":
            continue
        strategy, query = records[0]["index_strategy"], records[0]["query"]
        causes = Counter()
        for r in records:
            # Precedence per §4.2.6: retrieval_miss → selection_error →
            # false_abstention → spurious_assertion. A miss is attributed first
            # because nothing the selector could have done would have helped.
            if r["outcome"] in CORRECT:
                cause = "correct"
            elif retrieved_gold.get((strategy, query, k, r["object_id"])) is False:
                cause = "retrieval_miss"
            elif r["outcome"] == "WT":
                cause = "selection_error"
            elif r["outcome"] == "FA":
                cause = "false_abstention"
            elif r["outcome"] == "SA":
                cause = "spurious_assertion"
            else:
                cause = r["outcome"]
            causes[cause] += 1
        runs = len({r["run"] for r in records}) or 1
        attribution_rows.append({"configuration": config, "k": k, "model": model,
                                 **{c: n / runs for c, n in causes.items()}})
    write_csv(RESULTS / "attribution.csv", attribution_rows,
              ["configuration", "k", "model", "correct", "retrieval_miss", "selection_error",
               "false_abstention", "spurious_assertion"])

    # ---- paired tests -----------------------------------------------------
    stats = {"mcnemar": [], "run_variance": [], "headline": {}}
    models_seen = [m for m in dict.fromkeys(
        model for (_, s_, _, model) in selection if s_ == "T1") ]
    reference = reference_slug(models_seen)

    for (config, system, k, model), records in sorted(selection.items()):
        if system != "T1":
            continue
        t0 = selection.get((config, "T0", k, NO_MODEL))
        if not t0:
            continue
        stats["mcnemar"].append({
            "configuration": config, "k": k, "model": model,
            "comparison": "T1 vs T0", "a": "T0", "b": "T1",
            **mcnemar(modal_correct(t0), modal_correct(records))})

    # Every model against the reference, holding configuration and k fixed. The
    # reference is the deployed model, so these are the comparisons the article
    # needs; comparing every pair would multiply tests without adding a claim.
    for (config, system, k, model), records in sorted(selection.items()):
        if system != "T1" or model == reference or reference is None:
            continue
        base = selection.get((config, "T1", k, reference))
        if not base:
            continue
        stats["mcnemar"].append({
            "configuration": config, "k": k, "model": model,
            "comparison": f"{model} vs {reference}",
            "a": reference, "b": model,
            **mcnemar(modal_correct(base), modal_correct(records))})

    # T1 at the shallowest k against every deeper k, within a configuration.
    # The unpaired CIs overlap heavily, so this is the test that can actually
    # settle whether depth costs accuracy — the objects are the same in both
    # cells, and only the pool depth differs.
    for config, model in sorted({(c, m) for (c, s, _, m) in selection if s == "T1"}):
        depths = sorted({k for (c, s, k, m) in selection
                         if s == "T1" and c == config and m == model})
        if len(depths) < 2:
            continue
        base = selection[(config, "T1", depths[0], model)]
        for k in depths[1:]:
            stats["mcnemar"].append({
                "configuration": config, "k": k, "model": model,
                "comparison": f"T1 k={depths[0]} vs T1 k={k}",
                "a": f"k={depths[0]}", "b": f"k={k}",
                **mcnemar(modal_correct(base),
                          modal_correct(selection[(config, "T1", k, model)]))})

    # T1 across configurations at the same k, paired per object
    for k, model in sorted({(k, m) for (_, s, k, m) in selection if s == "T1"}):
        a = selection.get(("term_hint", "T1", k, model))
        b = selection.get(("combine_name", "T1", k, model))
        if a and b:
            stats["mcnemar"].append({
                "configuration": "term_hint vs combine_name", "k": k,
                "model": model,
                "comparison": "T1 term_hint vs T1 combine_name",
                "a": "term_hint", "b": "combine_name",
                **mcnemar(modal_correct(a), modal_correct(b))})

    for (config, system, k, model), records in sorted(selection.items()):
        by_run = defaultdict(dict)
        for r in records:
            by_run[r["run"]][r["object_id"]] = r["outcome"]
        if len(by_run) < 2:
            continue
        objects = set.intersection(*(set(v) for v in by_run.values()))
        unstable = sum(1 for o in objects
                       if len({by_run[run][o] for run in by_run}) > 1)
        stats["run_variance"].append({
            "configuration": config, "system": system, "k": k, "model": model,
            "n_runs": len(by_run), "n_objects": len(objects),
            "objects_with_differing_outcomes": unstable})

    best = max((r for r in cell_rows if r["system"] == "T1"),
               key=lambda r: r["accuracy"])
    stats["headline"] = {
        "reference_model": reference,
        "best_T1_cell": {key: best[key] for key in
                         ("configuration", "k", "model", "accuracy", "term_precision",
                          "term_recall", "term_f1", "abstention_accuracy")},
        "T0_by_configuration": {r["configuration"]: r["accuracy"]
                                for r in cell_rows if r["system"] == "T0"},
        "n_objects": best["n_objects"],
        "bootstrap": {"resamples": 0 if args.no_bootstrap else args.resamples,
                      "cluster_by": "case_number"},
    }

    # ---- Experiment A ------------------------------------------------------
    n_category = score_category(stats, args)

    (RESULTS / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nwrote {len(cell_rows)} selection cells, {len(retrieval_rows)} retrieval "
          f"cells -> {RESULTS}")
    b = stats["headline"]["best_T1_cell"]
    print(f"best T1 cell: {b['configuration']} k={b['k']}  "
          f"acc={b['accuracy']:.3f}  P={b['term_precision']:.3f}  "
          f"R={b['term_recall']:.3f}  F1={b['term_f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
