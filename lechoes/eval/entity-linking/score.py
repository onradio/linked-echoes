#!/usr/bin/env python3
"""Stage 6 (part 1) — score the predictions (spec §8).

Reads `data/predictions/*.jsonl` plus the gold sheet and the frozen pools, and
writes everything the report needs to `data/results/`. No numbers are computed
anywhere else.

    python score.py

Outputs
-------
results/per_cell.csv        one row per (system, k, run): the §8.2 metrics
results/per_type.csv        per entity type at the headline k
results/retrieval.csv       retrieval recall @ k, system-independent
results/frr.csv             validator false-rejection rate per type
results/attribution.csv     §8.3 error attribution
results/stats.json          McNemar + bootstrap CIs, and the headline numbers
results/hard_cases.csv      mentions where S2 and S3 disagree (for Table 3)
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = DATA / "results"

# Spec §8.1 outcome codes
TP, WL, FN, TN, SL = "TP", "WL", "FN", "TN", "SL"
CORRECT = {TP, TN}


# ──────────────────────────────────────────────────────────────────────────
# loading
# ──────────────────────────────────────────────────────────────────────────

def gold_set(row):
    raw = (row["wikidata_qid"] + ";" + row["alt_qids"]).replace(",", ";")
    return {q.strip() for q in raw.split(";") if q.strip()}


def load(sheet_path, pools_path, valid_path, pred_dir, exclude_reasons,
         model=None):
    sheet = {r["mention_id"]: r for r in csv.DictReader(
        open(sheet_path, newline="", encoding="utf-8"))}
    scored = {m: r for m, r in sheet.items()
              if r["nil_reason"].strip() not in exclude_reasons}
    pools = {json.loads(l)["mention_id"]: json.loads(l)
             for l in open(pools_path, encoding="utf-8")}
    valid = {}
    for line in open(valid_path, encoding="utf-8"):
        r = json.loads(line)
        v = r["eck_valid"]
        valid[(r["mention_id"], r["wikidata_id"])] = (
            None if v == "error_api" else bool(v))
    # Cells stay keyed by (system, k, run); the model is a *filter* rather than
    # part of the key, because every metric below reports one model at a time.
    # Two models' S3_k10_run1 would otherwise collide on one key and the file
    # read last would silently win.
    cells, models_seen = {}, set()
    for f in sorted(pred_dir.glob("*.jsonl")):
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        r0 = recs[0]
        slug = r0.get("model_slug")
        if slug:
            models_seen.add(slug)
            if model and slug != model:
                continue
        key = (r0["system"], r0["k"], r0["run"])
        if key in cells:
            raise SystemExit(
                f"two prediction files share the cell {key}: {f.name} collides "
                f"with an earlier file. Pass --model to score one model.")
        cells[key] = recs
    return sheet, scored, pools, valid, cells, sorted(models_seen)


# ──────────────────────────────────────────────────────────────────────────
# scoring
# ──────────────────────────────────────────────────────────────────────────

def outcome_of(gold, predicted):
    if gold:
        if predicted and predicted in gold:
            return TP
        return WL if predicted else FN
    return SL if predicted else TN


def attribute(code, rec, gold, pool, valid, system, k):
    """§8.3 — exactly one cause per error, in precedence order."""
    if code in CORRECT:
        return ""
    in_pool = {c["wikidata_id"] for c in pool["candidates"][:k]} & gold
    if gold and not in_pool:
        return "retrieval_miss"
    if system in ("S1", "S3") and gold:
        if all(valid.get((pool["mention_id"], q)) is False for q in in_pool):
            return "validator_rejection"
    if code == WL:
        return "llm_misselection"
    if code == FN:
        return "llm_false_abstention"
    return "spurious_link"


def metrics(counts, weights=None):
    """counts: {code: n} or, when weighted, {code: summed weight}."""
    c = counts
    n = sum(c.values())
    tp, wl, fn, tn, sl = (c.get(x, 0) for x in (TP, WL, FN, TN, SL))
    asserted = tp + wl + sl
    attempted = tp + wl + fn
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "link_precision": tp / asserted if asserted else 0.0,
        "link_recall": tp / attempted if attempted else 0.0,
        "TP": tp, "WL": wl, "FN": fn, "TN": tn, "SL": sl,
    }


def score_cell(recs, scored, pools, valid, system, k):
    per_mention = {}
    for r in recs:
        mid = r["mention_id"]
        row = scored.get(mid)
        if row is None:          # excluded (data_error)
            continue
        gold = gold_set(row)
        code = outcome_of(gold, r["predicted_qid"])
        per_mention[mid] = {
            "code": code,
            "cause": attribute(code, r, gold, pools[mid], valid, system, k),
            "weight": int(row["n_objects"]),
            "entity_type": r["entity_type"],
            "n_retrieved": r["n_retrieved"],
            "n_validated": r["n_validated"],
        }
    return per_mention


# ──────────────────────────────────────────────────────────────────────────
# statistics (§8.4)
# ──────────────────────────────────────────────────────────────────────────

def mcnemar_exact(b, c):
    """
    Two-sided exact McNemar: a binomial sign test on the discordant pairs.

    Uses scipy when available and falls back to the closed form otherwise; the
    two agree to machine precision on the cases exercised by this benchmark.
    """
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(b, n, 0.5).pvalue)
    except ImportError:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        return min(1.0, 2 * tail)


def bootstrap_ci(objects_to_mentions, value_fn, resamples, seed=20260810):
    """Cluster bootstrap over objects (§8.4), returning (lo, hi) at 95%."""
    rng = random.Random(seed)
    objs = list(objects_to_mentions)
    stats = []
    for _ in range(resamples):
        sample = [objects_to_mentions[rng.choice(objs)] for _ in objs]
        mentions = [m for group in sample for m in group]
        v = value_fn(mentions)
        if v is not None:
            stats.append(v)
    if not stats:
        return (0.0, 0.0)
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[min(len(stats) - 1, int(0.975 * len(stats)))]
    return (lo, hi)


# ──────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eck-version", default="v1")
    ap.add_argument("--k-headline", type=int, default=10)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--exclude-nil", nargs="*", default=["data_error"])
    ap.add_argument("--model", default=None,
                    help="model slug to score when predictions exist for more "
                         "than one (e.g. google-gemini-2-5-flash)")
    args = ap.parse_args(argv)

    sheet, scored, pools, valid, cells, models_seen = load(
        DATA / "annotation_sheet.csv", DATA / "pools_raw.jsonl",
        DATA / f"pools_valid_{args.eck_version}.jsonl",
        DATA / "predictions", set(args.exclude_nil), model=args.model)
    if len(models_seen) > 1 and not args.model:
        raise SystemExit(
            "predictions exist for more than one model "
            f"({', '.join(models_seen)}); pass --model to choose. Scoring them "
            "together would average different systems into one number.")
    if models_seen:
        print(f"LLM systems scored for model: {args.model or models_seen[0]}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    K = args.k_headline
    print(f"{len(sheet)} mentions, {len(scored)} scored "
          f"(excluded: {sorted(set(args.exclude_nil))}), {len(cells)} cells\n")

    # ── per-cell metrics ────────────────────────────────────────────────
    per_cell, cell_mentions = [], {}
    for (system, k, run), recs in sorted(cells.items()):
        pm = score_cell(recs, scored, pools, valid, system, k)
        cell_mentions[(system, k, run)] = pm
        unw = metrics({c: sum(1 for v in pm.values() if v["code"] == c)
                       for c in (TP, WL, FN, TN, SL)})
        wt = metrics({c: sum(v["weight"] for v in pm.values() if v["code"] == c)
                      for c in (TP, WL, FN, TN, SL)})
        n_ret = [v["n_retrieved"] for v in pm.values()]
        n_val = [v["n_validated"] for v in pm.values()]
        per_cell.append({
            "system": system, "k": k, "run": run,
            **{f"{key}": unw[key] for key in
               ("n", "accuracy", "link_precision", "link_recall",
                "TP", "WL", "FN", "TN", "SL")},
            "accuracy_weighted": wt["accuracy"],
            "link_precision_weighted": wt["link_precision"],
            "mean_N": sum(n_ret) / len(n_ret),
            "mean_M": sum(n_val) / len(n_val),
        })
    write_csv(RESULTS / "per_cell.csv", per_cell)
    print(f"wrote per_cell.csv ({len(per_cell)} rows)")

    # ── retrieval recall @ k (system-independent) ───────────────────────
    ks = sorted({k for _, k, _ in cells})
    retrieval = []
    types = sorted({p["entity_type"] for p in pools.values()})
    for etype in types + ["ALL"]:
        row = {"entity_type": etype}
        for k in ks:
            hit = tot = 0
            for mid, r in scored.items():
                g = gold_set(r)
                if not g or (etype != "ALL" and r["entity_type"] != etype):
                    continue
                tot += 1
                hit += bool(g & {c["wikidata_id"]
                                 for c in pools[mid]["candidates"][:k]})
            row[f"recall@{k}"] = hit / tot if tot else 0.0
            row["n"] = tot
        retrieval.append(row)
    write_csv(RESULTS / "retrieval.csv", retrieval)
    print("wrote retrieval.csv")

    # ── validator false-rejection rate (§8.2 metric 5) ──────────────────
    frr = []
    for etype in types + ["ALL"]:
        n = bad = 0
        for mid, r in scored.items():
            if etype != "ALL" and r["entity_type"] != etype:
                continue
            g = gold_set(r)
            for c in pools[mid]["candidates"][:K]:
                if c["wikidata_id"] in g:
                    n += 1
                    if valid.get((mid, c["wikidata_id"])) is False:
                        bad += 1
        frr.append({"entity_type": etype, "gold_in_pool": n, "rejected": bad,
                    "frr": bad / n if n else 0.0})
    write_csv(RESULTS / "frr.csv", frr)
    print("wrote frr.csv")

    # ── per entity type at headline k (Table 2) ─────────────────────────
    def acc_by_type(system, k, etype):
        vals = []
        for run in sorted({r for s, kk, r in cells if s == system and kk == k}):
            pm = cell_mentions[(system, k, run)]
            sel = [v for v in pm.values()
                   if etype == "ALL" or v["entity_type"] == etype]
            if sel:
                vals.append(sum(1 for v in sel if v["code"] in CORRECT) / len(sel))
        return sum(vals) / len(vals) if vals else 0.0

    n_inst = defaultdict(int)
    n_dist = defaultdict(int)
    for mid, r in scored.items():
        n_dist[r["entity_type"]] += 1
        n_inst[r["entity_type"]] += int(r["n_objects"])
    frr_by = {f["entity_type"]: f["frr"] for f in frr}
    ret_by = {r["entity_type"]: r.get(f"recall@{K}", 0.0) for r in retrieval}
    per_type = []
    for etype in types + ["ALL"]:
        s2, s3 = acc_by_type("S2", K, etype), acc_by_type("S3", K, etype)
        per_type.append({
            "entity_type": etype,
            "n_distinct": n_dist[etype] if etype != "ALL" else sum(n_dist.values()),
            "n_instances": n_inst[etype] if etype != "ALL" else sum(n_inst.values()),
            f"retrieval_recall@{K}": ret_by.get(etype, 0.0),
            "S0_accuracy": acc_by_type("S0", ks[0], etype),
            "S1_accuracy": acc_by_type("S1", K, etype),
            "S2_accuracy": s2, "S3_accuracy": s3, "delta": s3 - s2,
            "validator_frr": frr_by.get(etype, 0.0),
        })
    write_csv(RESULTS / "per_type.csv", per_type)
    print("wrote per_type.csv")

    # ── error attribution (§8.3) ────────────────────────────────────────
    attribution = []
    for system in ("S0", "S1", "S2", "S3"):
        kk = ks[0] if system == "S0" else K
        runs = sorted({r for s, k2, r in cells if s == system and k2 == kk})
        if not runs:
            continue
        agg = defaultdict(float)
        for run in runs:
            for v in cell_mentions[(system, kk, run)].values():
                if v["cause"]:
                    agg[v["cause"]] += 1 / len(runs)
        attribution.append({"system": system, "k": kk, **{c: round(agg.get(c, 0), 2)
                            for c in ("retrieval_miss", "validator_rejection",
                                      "llm_misselection", "llm_false_abstention",
                                      "spurious_link")}})
    write_csv(RESULTS / "attribution.csv", attribution)
    print("wrote attribution.csv")

    # ── statistics (§8.4) ───────────────────────────────────────────────
    # Cluster by object (§8.4). Mentions recur across objects, so assigning each
    # to every object it appears in would resample popular mentions more often
    # and silently shift the estimand from the unweighted per-mention accuracy
    # to the instance-weighted one — which put the point estimate outside its
    # own interval. Each mention is therefore assigned to exactly one cluster,
    # its first object, keeping the clusters a partition of the scored set.
    obj_to_mentions = defaultdict(list)
    for mid, r in scored.items():
        primary = (r["object_ids"].split() or [mid])[0]
        obj_to_mentions[primary].append(mid)

    s2_pm = cell_mentions[("S2", K, 1)]
    s3_pm = cell_mentions[("S3", K, 1)]
    common = sorted(set(s2_pm) & set(s3_pm))
    b = sum(1 for m in common
            if s2_pm[m]["code"] in CORRECT and s3_pm[m]["code"] not in CORRECT)
    c = sum(1 for m in common
            if s2_pm[m]["code"] not in CORRECT and s3_pm[m]["code"] in CORRECT)
    p_value = mcnemar_exact(b, c)

    def acc_fn(pm):
        return lambda ms: (sum(1 for m in ms if m in pm and pm[m]["code"] in CORRECT)
                           / len([m for m in ms if m in pm])
                           if any(m in pm for m in ms) else None)

    def delta_fn(ms):
        sel = [m for m in ms if m in s2_pm and m in s3_pm]
        if not sel:
            return None
        a2 = sum(1 for m in sel if s2_pm[m]["code"] in CORRECT) / len(sel)
        a3 = sum(1 for m in sel if s3_pm[m]["code"] in CORRECT) / len(sel)
        return a3 - a2

    stats = {
        "k_headline": K, "eck_version": args.eck_version,
        "n_scored": len(scored), "n_excluded": len(sheet) - len(scored),
        "mcnemar_S2_vs_S3": {"b_S2correct_S3wrong": b, "c_S2wrong_S3correct": c,
                             "p_value": p_value, "k": K, "run": 1},
        "bootstrap": {"resamples": args.resamples, "cluster": "object"},
    }
    for system in ("S0", "S1", "S2", "S3"):
        kk = ks[0] if system == "S0" else K
        if (system, kk, 1) not in cell_mentions:
            continue
        pm = cell_mentions[(system, kk, 1)]
        lo, hi = bootstrap_ci(obj_to_mentions, acc_fn(pm), args.resamples)
        stats[f"{system}_accuracy_ci95"] = [lo, hi]
    lo, hi = bootstrap_ci(obj_to_mentions, delta_fn, args.resamples)
    stats["delta_S3_minus_S2_ci95"] = [lo, hi]
    (RESULTS / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("wrote stats.json")

    # ── hard cases (Table 3) ────────────────────────────────────────────
    hard = []
    lab = {c["wikidata_id"]: c["label"]
           for p in pools.values() for c in p["candidates"]}
    s2_recs = {r["mention_id"]: r for r in cells[("S2", K, 1)]}
    s3_recs = {r["mention_id"]: r for r in cells[("S3", K, 1)]}
    for mid in common:
        if s2_pm[mid]["code"] in CORRECT or s3_pm[mid]["code"] not in CORRECT:
            continue                      # only S3-right / S2-wrong
        row = scored[mid]
        p2 = s2_recs[mid]["predicted_qid"]
        p3 = s3_recs[mid]["predicted_qid"]
        hard.append({
            "mention_id": mid, "label": row["label"],
            "entity_type": row["entity_type"],
            "gold": row["wikidata_qid"] or f"NIL/{row['nil_reason']}",
            "S2_predicted": p2 or "-", "S2_label": lab.get(p2, ""),
            "S2_outcome": s2_pm[mid]["code"],
            "S3_predicted": p3 or "-", "S3_label": lab.get(p3, ""),
            "S3_outcome": s3_pm[mid]["code"],
        })
    write_csv(RESULTS / "hard_cases.csv", hard)
    print(f"wrote hard_cases.csv ({len(hard)} rows)")

    print(f"\nMcNemar S2 vs S3 @k={K}: b={b}, c={c}, p={p_value:.3g}")
    print(f"Δ accuracy (S3−S2) 95% CI: [{lo:.3f}, {hi:.3f}]")
    return 0


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
