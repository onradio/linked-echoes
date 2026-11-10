#!/usr/bin/env python3
"""Stage 3 — score every prediction on disk (spec §4). No numbers are computed elsewhere.

    python score.py                 # everything that has predictions
    python score.py --no-bootstrap  # skip the CIs (fast, for iterating)
    python score.py --systems M3 M4 # a subset, for checking one change

Reads `data/predictions/` plus `gold.py` and writes `data/results/`:

    system_cells.csv     one row per (system, run): the §4.4 metrics
    system_summary.csv   the same, averaged across runs, with CIs — Table 1's source
    field_breakdown.csv  §4.4.5, per (system, field)
    cell_outcomes.csv    §4.3, per (system, run): HIT/WRONG/MISS/SPURIOUS/ABSTAIN
    attribution.csv      §4.4.7, one row per (system, cause, field)
    ceilings.csv         §2.4 — exhaustive extraction at each scope; recall is a ceiling,
                         precision is a reference point, see gold.ceilings()
    stats.json           McNemar against the reference cell, headline numbers, run variance

Four decisions are load-bearing, and each fixes a defect the superseded evaluator had:

**Precision and recall are not transposed.** `evaluation.py:322` unpacked `tp, fn, fp` from a
function returning `(tp, fp, fn)`, so every per-field and macro figure it produced had P and R
the wrong way round (§6.1); F₁ survived only because the harmonic mean is symmetric. Here the
counts travel in a dataclass with named fields, which is the structural fix rather than a
comment asking the reader to be careful.

**Matching is an optimal assignment, not greedy first-best.** `process.extractOne` gives each
prediction its single best gold candidate and discards it if that candidate is taken, which
under-counts true positives and makes the result depend on list order (§6.5). The threshold is
applied *after* the assignment. Two measured caveats, both in `test_scoring.py`: the spec's own
worked example is unreachable under §4.1's article stripping, since its two predictions
normalise to the same value; and on this corpus greedy and optimal agree on every cell, because
no two gold values within a field score even 70 against each other. The optimal assignment is
kept because it is provably never worse, not because it recovered anything here.

**Everything joins on `object_id`.** Never list position — the defect that displaced all 62 rows
of `eval_folder.py` (§6.12) and mis-indexed four cases in the other of its two modes (§6.13).

**The gold the predictions were scored against is checked, not assumed.** The manifest records the
sheet and alias-table SHA-256s; if either has moved since the run, scoring stops. Growing the
alias table in response to a system's output would be tuning on the evaluation set (§2.5), and
this is what makes that impossible to do by accident.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest

import gold as G
import predictions as P

HERE = Path(__file__).resolve().parent
YEAR_RE = re.compile(r"^\d{4}$")

EXACT, ALIAS, FUZZY, NONE = "exact", "alias", "fuzzy", "none"


# --------------------------------------------------------------------------- matching

@dataclass
class Counts:
    """TP/FP/FN by name, so they cannot be unpacked in the wrong order (§6.1)."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tp_supported: int = 0
    tp_inference: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn,
                      self.tp_supported + other.tp_supported,
                      self.tp_inference + other.tp_inference)

    @property
    def precision(self):
        # Undefined when the system asserts nothing: reporting 1.000 would let a system
        # that says nothing top a precision ranking, and 0.000 would assert errors it did
        # not make (§4.4.1).
        total = self.tp + self.fp
        return self.tp / total if total else None

    @property
    def recall(self):
        total = self.tp + self.fn
        return self.tp / total if total else None

    @property
    def f1(self):
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if (p + r) else 0.0


def pair_score(field: str, pred: str, gold_value: str, policy: dict) -> tuple:
    """
    `(score, kind)` for one prediction against one gold value, both normalised.

    Exact scores 100 and alias 99 so an exact match always wins the assignment over an
    alias to the same canonical form. Fuzzy is only ever *offered* for the six name-like
    fields: for a short closed class it conflates a hypernym with its hyponym —
    `WRatio('rock', 'rock and roll') = 90` would credit `Rock` with `Rock and Roll`, which
    is why §4.1 forbids it there and why `token_sort_ratio` is used where it is allowed
    (it scores that pair 47, and `the rolling stones` against `rolling stones` 88).
    """
    if pred == gold_value:
        return 100.0, EXACT
    if policy.get("alias") and G.canonicalise(field, pred) == G.canonicalise(field, gold_value):
        return 99.0, ALIAS
    fuzzy = policy.get("fuzzy")
    if fuzzy:
        return float(fuzz.token_sort_ratio(pred, gold_value)), FUZZY
    return 0.0, NONE


def admissible(score: float, kind: str, policy: dict) -> bool:
    """The threshold, applied *after* the assignment (§4.1)."""
    if kind in (EXACT, ALIAS):
        return True
    if kind == FUZZY:
        return score >= float(policy["fuzzy"]["threshold"])
    return False


def match_cell(field: str, preds: list, golds: list, policy: dict) -> tuple:
    """
    One (object, field). Returns `(matched_pairs, unmatched_preds, unmatched_golds)`.

    The full score matrix is solved as an optimal one-to-one assignment; only then is the
    threshold applied, so a pairing is never blocked by a prediction that took a better
    slot it did not need (§6.5).
    """
    if not preds or not golds:
        return [], list(preds), list(golds)

    matrix = np.zeros((len(preds), len(golds)))
    kinds: dict = {}
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            score, kind = pair_score(field, p, g, policy)
            matrix[i, j] = score
            kinds[(i, j)] = kind

    rows, cols = linear_sum_assignment(-matrix)
    matched, used_p, used_g = [], set(), set()
    for i, j in zip(rows, cols):
        if admissible(matrix[i, j], kinds[(i, j)], policy):
            matched.append((preds[i], golds[j], matrix[i, j], kinds[(i, j)]))
            used_p.add(i)
            used_g.add(j)
    return (matched,
            [p for i, p in enumerate(preds) if i not in used_p],
            [g for j, g in enumerate(golds) if j not in used_g])


def prepare_predicted(record: dict, field: str, config: dict) -> tuple:
    """
    The normalised, deduplicated predictions for one cell, plus the count discarded.

    Years are coerced to strings first — a model returning the JSON number 1973 must not be
    lost as a type error — and anything that is not four digits afterwards is discarded and
    counted (§4.1 rule 3). Note this *removes* an assertion rather than charging it as a
    false positive, which is generous to the systems; the counts are reported so the effect
    is visible.

    The coercion is of **type only**. Stripping non-digits so that `1970s` becomes `1970`
    was the first thing this function did and it was wrong: a decade is not a year, and
    inventing the precision the model declined to assert would credit or charge it for a
    value it never gave. `1970s` and `68` are both discarded.
    """
    values, discarded, seen = [], 0, set()
    for item in record["predicted"][field]:
        value_in = item["value"]
        raw = (str(int(value_in)) if isinstance(value_in, (int, float))
               and not isinstance(value_in, bool) else str(value_in).strip())
        if field == "years" and not YEAR_RE.match(raw):
            discarded += 1
            continue
        value = G.normalise(raw, field)
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values, discarded


# ---------------------------------------------------------------------- attribution

def attribute_fn(gold_obj: G.Gold, object_id: str, field: str, value: str,
                 scope: str) -> str:
    """§4.4.7 for a false negative, in precedence order."""
    evidence = G.normalise(gold_obj.evidence(object_id, scope))
    if not G.contains(evidence, value):
        return "out_of_scope"
    return "missed"


def attribute_fp(gold_obj: G.Gold, object_id: str, field: str, value: str, scope: str,
                 policy: dict) -> str:
    """§4.4.7 for a false positive, in precedence order."""
    obj = gold_obj.by_id[object_id]

    # 3 — a gold value of a sibling in the same display case.
    for sib in obj["siblings"]:
        if value in gold_obj.gold_values(sib, field):
            return "misattribution"

    # 4 — recorded against this object under a different field.
    for other in G.SCORED:
        if other != field and value in gold_obj.gold_values(object_id, other):
            return "wrong_field"

    # 5 — would have matched this field's gold under a looser policy than the field's.
    for g in gold_obj.gold_values(object_id, field):
        score, kind = pair_score(field, value, g, policy)
        if not admissible(score, kind, policy):
            loose = fuzz.token_sort_ratio(value, g)
            if loose >= 85:
                return "normalisation"

    # 6 — occurs nowhere in this system's evidence.
    if not G.contains(G.normalise(gold_obj.evidence(object_id, scope)), value):
        return "hallucination"

    # 7 — present in the evidence but none of the above.
    return "other"


# -------------------------------------------------------------------- scoring a run

@dataclass
class RunScore:
    system: str
    run: int
    scope: str
    policy: str
    model: str | None
    counts: Counts = dc_field(default_factory=Counts)
    per_field: dict = dc_field(default_factory=dict)
    cells: collections.Counter = dc_field(default_factory=collections.Counter)
    attribution: collections.Counter = dc_field(default_factory=collections.Counter)
    per_cell_exact: dict = dc_field(default_factory=dict)
    hallucinated: list = dc_field(default_factory=list)   # (object_id, field, value)
    per_object: dict = dc_field(default_factory=dict)
    values_asserted: int = 0
    cells_asserted: int = 0
    years_discarded: int = 0
    failures: int = 0


def score_run(gold_obj: G.Gold, config: dict, records: list, *, system: str, run: int,
              scope: str, policy_name: str, model: str | None) -> RunScore:
    policies = config["matching"]["policy"]
    out = RunScore(system=system, run=run, scope=scope, policy=policy_name, model=model)

    for record in records:
        oid = record["object_id"]
        if record["predicted"] is None:
            # An API or parse failure is excluded from the cell's metrics and counted; it
            # is never scored as an abstention (§1.3).
            out.failures += 1
            continue

        object_counts = Counts()
        for field in G.SCORED:
            policy = policies[field]
            preds, discarded = prepare_predicted(record, field, config)
            out.years_discarded += discarded
            golds = sorted(gold_obj.gold_values(oid, field))

            matched, fps, fns = match_cell(field, preds, golds, policy)
            cell = Counts(tp=len(matched), fp=len(fps), fn=len(fns))
            for _, g, _, _ in matched:
                if gold_obj.support_class(oid, field, g) == "supported":
                    cell.tp_supported += 1
                else:
                    cell.tp_inference += 1

            out.counts += cell
            object_counts += cell
            fc = out.per_field.setdefault(field, Counts())
            out.per_field[field] = fc + cell
            out.values_asserted += len(preds)
            if preds:
                out.cells_asserted += 1

            # §4.3 — every (object, field) falls into exactly one cell outcome.
            if golds:
                outcome = "HIT" if matched else ("WRONG" if preds else "MISS")
            else:
                outcome = "SPURIOUS" if preds else "ABSTAIN"
            out.cells[outcome] += 1
            # "Exactly right" for McNemar: the value sets agree under §4.1 matching.
            out.per_cell_exact[(oid, field)] = (not fps and not fns)

            for value in fps:
                cause = attribute_fp(gold_obj, oid, field, value, scope, policy)
                out.attribution[("FP", cause, field)] += 1
                if cause == "hallucination":
                    # Keep the key, not just the count: the adjudication pool was taken
                    # whole, so each one can be looked up rather than estimated.
                    out.hallucinated.append((oid, field, value))
            for value in fns:
                cause = attribute_fn(gold_obj, oid, field, value, scope)
                out.attribution[("FN", cause, field)] += 1

        out.per_object[oid] = object_counts
    return out


# ------------------------------------------------------------------------ statistics

def bootstrap_ci(per_object: dict, gold_obj: G.Gold, *, resamples: int, seed: int = 7):
    """
    Percentile bootstrap over **display cases**, not objects (§4.6).

    Objects in a case share a caption and often near-identical wording — the three Suede
    lyrics sheets are not independent draws — so an object-level bootstrap would understate
    the intervals.
    """
    cases = sorted(gold_obj.cases)
    members = {c: gold_obj.cases[c] for c in cases}
    rng = np.random.default_rng(seed)
    ps, rs, fs = [], [], []
    for _ in range(resamples):
        drawn = rng.choice(len(cases), size=len(cases), replace=True)
        total = Counts()
        for idx in drawn:
            for oid in members[cases[idx]]:
                if oid in per_object:
                    total += per_object[oid]
        if total.precision is not None:
            ps.append(total.precision)
        if total.recall is not None:
            rs.append(total.recall)
        if total.f1 is not None:
            fs.append(total.f1)

    def pct(values):
        if not values:
            return (None, None)
        return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))

    return {"precision": pct(ps), "recall": pct(rs), "f1": pct(fs)}


def mcnemar(system_exact: dict, reference_exact: dict) -> dict:
    """
    Exact McNemar on the per-cell binary outcome *the cell is exactly right*, against the
    reference cell (§4.6). The discordant counts are reported alongside the p-value: on a
    sample this size they carry more information than the p-value does.

    F₁ is not a per-item statistic and McNemar does not apply to it. That is why this test
    is on cell-exactness rather than on the headline numbers.
    """
    keys = sorted(set(system_exact) & set(reference_exact))
    b = sum(1 for k in keys if system_exact[k] and not reference_exact[k])
    c = sum(1 for k in keys if reference_exact[k] and not system_exact[k])
    n = b + c
    p = binomtest(b, n, 0.5).pvalue if n else 1.0
    return {"cells": len(keys), "system_only": b, "reference_only": c,
            "discordant": n, "p_value": float(p)}


def majority_vote(runs: list) -> dict:
    """
    The modal per-cell outcome across runs. Three runs at temperature 0 differ by one or
    two objects; averaging is the honest summary for reporting, but McNemar needs one
    binary outcome per cell, so the mode is used and the disagreement rate reported beside
    it (§4.6).
    """
    keys = set().union(*(set(r.per_cell_exact) for r in runs))
    voted = {}
    for key in keys:
        votes = [r.per_cell_exact.get(key) for r in runs if key in r.per_cell_exact]
        voted[key] = sum(1 for v in votes if v) * 2 > len(votes)
    return voted


def differing_cells(runs: list) -> int:
    """Cells whose outcome is not identical across runs — determinism, measured."""
    if len(runs) < 2:
        return 0
    keys = set(runs[0].per_cell_exact)
    return sum(1 for k in keys
               if len({r.per_cell_exact.get(k) for r in runs}) > 1)



# ------------------------------------------------------------------- adjudication

VERDICTS = ("correct", "incorrect", "unverifiable")


def load_adjudication(config: dict) -> dict | None:
    """
    The annotators' verdicts, keyed by `(object_id, field, value)`.

    Validated against the frozen sample: every drawn key must be present and judged, and no
    verdict outside the configured vocabulary. A sheet that does not match the sample it was
    drawn from is refused rather than partially used.
    """
    path = HERE / "data" / "adjudication_sheet_done.csv"
    sample_path = HERE / "data" / "adjudication_sample.json"
    if not path.exists():
        return None
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))

    verdicts, problems = {}, []
    for row in rows:
        key = (row["object_id"], row["field"], row["value"])
        verdict = row["verdict"].strip().lower()
        if verdict not in VERDICTS:
            problems.append(f"{row['adjudication_id']}: verdict {row['verdict']!r} is not "
                            f"one of {VERDICTS}")
            continue
        verdicts[key] = verdict

    drawn = {tuple(k) for k in sample["keys"]}
    missing = drawn - set(verdicts)
    extra = set(verdicts) - drawn
    if missing:
        problems.append(f"{len(missing)} drawn row(s) unjudged, e.g. {sorted(missing)[:2]}")
    if extra:
        problems.append(f"{len(extra)} row(s) not in the drawn sample, e.g. {sorted(extra)[:2]}")
    if problems:
        raise SystemExit("adjudication sheet is not usable:\n  " + "\n  ".join(problems))

    return {"verdicts": verdicts, "pool_size": sample["pool_size"],
            "drawn": sample["drawn"], "seed": sample["seed"],
            "whole_pool": sample["drawn"] == sample["pool_size"]}


def adjudicate(runs: list, adjudication: dict, exclude: list) -> dict:
    """
    Adjusted precision for one system, as a **bracket** rather than a point estimate.

    §4.5 wrote this as an extrapolation — `p̂_f` from a sample, applied to the field's whole
    hallucination count — because it anticipated sampling up to 200 rows from a larger pool.
    Here the pool is 91 and the cap is 200, so it was taken **whole**: every hallucination
    false positive carries its own verdict and the `correct` and `incorrect` classes need no
    estimating at all. They are looked up.

    What does need a decision is the `unverifiable` class, and it is larger than §4.5
    anticipated because the annotators applied a stricter protocol than the one that section
    describes: a value is `unverifiable` when the description and caption cannot settle it,
    *even if outside knowledge suggests it is true*. That makes the exclusion deliberately
    non-random — it selects for exactly the values whose truth the evidence cannot reach — so
    extrapolating the verifiable rows' rate onto it is the one assumption worth refusing to
    make silently. Three figures are reported instead:

    `lower`  every unverifiable value is wrong. The most conservative reading, and the one
             closest to strict precision.
    `point`  unverifiable values are correct at the rate observed among the judged ones of
             the same field — §4.5's assumption, stated rather than assumed.
    `upper`  every unverifiable value is right. The bound a reviewer asking "how much could
             the catalogue's incompleteness be hiding?" is entitled to.

    Strict precision remains the primary number in every case (§4.5).
    """
    verdicts = adjudication["verdicts"]
    per_run = {"lower": [], "point": [], "upper": [], "correct": [], "unverifiable": []}

    # Rate among *judged* rows of each field, for the point estimate only.
    judged = collections.defaultdict(collections.Counter)
    for (oid, field, value), verdict in verdicts.items():
        judged[field][verdict] += 1
    field_rate, pooled = {}, collections.Counter()
    for field, counter in judged.items():
        pooled.update(counter)
        denominator = counter["correct"] + counter["incorrect"]
        field_rate[field] = (counter["correct"] / denominator) if denominator else None
    pooled_denominator = pooled["correct"] + pooled["incorrect"]
    pooled_rate = (pooled["correct"] / pooled_denominator) if pooled_denominator else 0.0

    for run in runs:
        tp, fp = run.counts.tp, run.counts.fp
        n_correct = n_unverifiable = 0
        expected = 0.0
        for key in run.hallucinated:
            verdict = verdicts.get(key)
            if verdict == "correct":
                n_correct += 1
                expected += 1.0
            elif verdict == "unverifiable":
                n_unverifiable += 1
                rate = field_rate.get(key[1])
                expected += pooled_rate if rate is None else rate
        denominator = tp + fp
        per_run["correct"].append(n_correct)
        per_run["unverifiable"].append(n_unverifiable)
        if denominator:
            per_run["lower"].append((tp + n_correct) / denominator)
            per_run["point"].append((tp + expected) / denominator)
            per_run["upper"].append((tp + n_correct + n_unverifiable) / denominator)

    def mean(key):
        return statistics.fmean(per_run[key]) if per_run[key] else None

    return {"lower": mean("lower"), "point": mean("point"), "upper": mean("upper"),
            "hallucinations_judged_correct": mean("correct"),
            "hallucinations_unverifiable": mean("unverifiable")}


def adjudication_rates(adjudication: dict) -> list:
    """Per-field verdict counts, for `adjudication.csv`."""
    judged = collections.defaultdict(collections.Counter)
    for (_, field, _), verdict in adjudication["verdicts"].items():
        judged[field][verdict] += 1
    rows = []
    for field in sorted(judged, key=lambda f: -sum(judged[f].values())):
        c = judged[field]
        denominator = c["correct"] + c["incorrect"]
        rows.append({"field": field, "judged": sum(c.values()),
                     "correct": c["correct"], "incorrect": c["incorrect"],
                     "unverifiable": c["unverifiable"],
                     "rate_correct": fmt(c["correct"] / denominator) if denominator else "—"})
    total = collections.Counter()
    for c in judged.values():
        total.update(c)
    denominator = total["correct"] + total["incorrect"]
    rows.append({"field": "ALL", "judged": sum(total.values()),
                 "correct": total["correct"], "incorrect": total["incorrect"],
                 "unverifiable": total["unverifiable"],
                 "rate_correct": fmt(total["correct"] / denominator) if denominator else "—"})
    return rows


# --------------------------------------------------------------------------- writing

def fmt(value, digits: int = 3) -> str:
    """`—` for an undefined ratio, never 1.000 and never 0.000 (§4.4.1)."""
    return "—" if value is None else f"{value:.{digits}f}"


def write_csv(path: Path, rows: list, columns: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--systems", nargs="+", help="subset of the configured systems")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--resamples", type=int)
    args = ap.parse_args()

    config = G.CONFIG
    gold_obj = G.Gold()
    baseline = gold_obj.baseline()
    results = P.paths(config)["results"]
    manifest_path = P.paths(config)["predictions"] / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("no manifest — run the systems first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # The gold guard: predictions are only comparable to the gold they were produced
    # against, and the alias table must not have moved since (§2.5).
    recorded = manifest.get("gold", {})
    for key in ("sheet_sha256", "aliases_sha256"):
        if recorded.get(key) and recorded[key] != baseline[key]:
            raise SystemExit(
                f"{key} has changed since the predictions were written "
                f"({recorded[key][:12]}… -> {baseline[key][:12]}…). Re-run the systems, or "
                f"restore the artefact; scoring across a moved gold is not comparable.")

    entries = list(manifest["entries"].values())
    if args.systems:
        entries = [e for e in entries if e["system"] in args.systems]
    systems = sorted({e["system"] for e in entries})
    reference = config["reference_system"]

    # A partial sweep must not be silently scored as a complete one (§1.4).
    expected_runs = int(config["runs"])
    incomplete = []
    for system in systems:
        runs = [e for e in entries if e["system"] == system]
        kinds = {s["id"]: s["kind"] for s in config["systems"]}
        want = 1 if kinds.get(system, "llm") in ("spacy", "gliner") else expected_runs
        if len(runs) != want:
            incomplete.append(f"{system}: {len(runs)} run(s), expected {want}")
        for e in runs:
            if e["objects"] != baseline["objects"]:
                incomplete.append(f"{system} run{e['run']}: {e['objects']} objects, "
                                  f"expected {baseline['objects']}")
    if incomplete:
        raise SystemExit("manifest is incomplete:\n  " + "\n  ".join(incomplete))

    # ------------------------------------------------------------------ score
    scored: dict = collections.defaultdict(list)
    for entry in entries:
        path = P.prediction_path(config, entry["system"], entry["model_slug"],
                                 entry["run"])
        records = P.read(path)
        scored[entry["system"]].append(score_run(
            gold_obj, config, records, system=entry["system"], run=entry["run"],
            scope=entry["evidence_scope"], policy_name=entry["assertion_policy"],
            model=entry["model_slug"]))

    ceilings = gold_obj.ceilings()
    n_cells = baseline["cells"]["total"]
    n_empty = baseline["cells"]["empty"]
    n_nonempty = baseline["cells"]["non_empty"]
    n_supported = baseline["supported"]
    n_inference = baseline["inference_only"]
    resamples = args.resamples or int(config["bootstrap"]["resamples"])

    per_run_rows, summary_rows, field_rows, cell_rows, attr_rows = [], [], [], [], []
    stats: dict = {"reference_system": reference, "gold": recorded or None,
                   "resamples": 0 if args.no_bootstrap else resamples, "systems": {}}

    reference_voted = majority_vote(scored[reference]) if reference in scored else {}
    adjudication = load_adjudication(config)
    exclude = list(config["adjudication"]["exclude_from_rate"])

    for system in systems:
        runs = sorted(scored[system], key=lambda r: r.run)
        scope = runs[0].scope
        for r in runs:
            per_run_rows.append({
                "system": system, "run": r.run, "model": r.model or "",
                "evidence_scope": r.scope, "assertion_policy": r.policy,
                "tp": r.counts.tp, "fp": r.counts.fp, "fn": r.counts.fn,
                "precision": fmt(r.counts.precision), "recall": fmt(r.counts.recall),
                "f1": fmt(r.counts.f1),
                "recall_supported": fmt(r.counts.tp_supported / n_supported),
                "inference_yield": fmt(r.counts.tp_inference / n_inference),
                "values_asserted": r.values_asserted,
                "cells_asserted": r.cells_asserted,
                "years_discarded": r.years_discarded,
                "failures": r.failures,
            })
            cell_rows.append({"system": system, "run": r.run,
                              **{k: r.cells.get(k, 0) for k in
                                 ("HIT", "WRONG", "MISS", "SPURIOUS", "ABSTAIN")},
                              "spurious_field_rate":
                                  fmt(r.cells.get("SPURIOUS", 0) / n_empty),
                              "true_abstention_rate":
                                  fmt(r.cells.get("ABSTAIN", 0) / n_empty),
                              "false_abstention_rate":
                                  fmt(r.cells.get("MISS", 0) / n_nonempty)})

        # Means across runs, and the SD where it is meaningful. M0a/M0b are deterministic
        # and run once: their SD is `—`, never 0.000 (§4.6).
        def mean_of(attr):
            values = [getattr(r.counts, attr) for r in runs]
            values = [v for v in values if v is not None]
            return statistics.fmean(values) if values else None

        def sd_of(attr):
            values = [getattr(r.counts, attr) for r in runs]
            values = [v for v in values if v is not None]
            return statistics.stdev(values) if len(values) > 1 else None

        # The interval must describe the quantity being reported. The point estimate is the
        # mean across runs, so the bootstrap resamples cases over per-object counts averaged
        # across runs — not over run 1 alone, which is what this did first and which would
        # have quietly attached a single run's spread to a three-run mean.
        mean_objects: dict = {}
        for r in runs:
            for oid, counts in r.per_object.items():
                mean_objects[oid] = mean_objects.get(oid, Counts()) + counts
        n_runs = len(runs)
        mean_objects = {oid: Counts(c.tp / n_runs, c.fp / n_runs, c.fn / n_runs,
                                    c.tp_supported / n_runs, c.tp_inference / n_runs)
                        for oid, c in mean_objects.items()}
        ci = ({} if args.no_bootstrap
              else bootstrap_ci(mean_objects, gold_obj, resamples=resamples))

        voted = majority_vote(runs)
        test = (mcnemar(voted, reference_voted)
                if reference_voted and system != reference else None)
        adjusted = adjudicate(runs, adjudication, exclude) if adjudication else None

        summary_rows.append({
            "system": system, "evidence_scope": scope,
            "assertion_policy": runs[0].policy, "runs": len(runs),
            "values_asserted_mean": f"{statistics.fmean(r.values_asserted for r in runs):.1f}",
            "values_per_object": f"{statistics.fmean(r.values_asserted for r in runs) / baseline['objects']:.2f}",
            "precision": fmt(mean_of("precision")), "precision_sd": fmt(sd_of("precision")),
            "recall": fmt(mean_of("recall")), "recall_sd": fmt(sd_of("recall")),
            "f1": fmt(mean_of("f1")), "f1_sd": fmt(sd_of("f1")),
            "precision_ci": (f"[{fmt(ci['precision'][0])}, {fmt(ci['precision'][1])}]"
                             if ci else ""),
            "recall_ci": (f"[{fmt(ci['recall'][0])}, {fmt(ci['recall'][1])}]"
                          if ci else ""),
            "recall_supported":
                fmt(statistics.fmean(r.counts.tp_supported for r in runs) / n_supported),
            "inference_yield":
                fmt(statistics.fmean(r.counts.tp_inference for r in runs) / n_inference),
            "spurious_field_rate":
                fmt(statistics.fmean(r.cells.get("SPURIOUS", 0) for r in runs) / n_empty),
            "false_abstention_rate":
                fmt(statistics.fmean(r.cells.get("MISS", 0) for r in runs) / n_nonempty),
            # `recall_ceiling` is a genuine bound — but only for systems that assert what
            # they can see. `exhaustive_precision` is NOT a bound: it is the precision of
            # copying everything visible, which any well-chosen subset beats. Naming it a
            # ceiling made beating it look like a bug rather than the result (§4.4.6).
            "exhaustive_precision": fmt(ceilings[scope]["precision"]),
            "recall_ceiling": fmt(ceilings[scope]["recall"]),
            "beats_exhaustive_precision":
                "yes" if (mean_of("precision") or 0) > ceilings[scope]["precision"] else "no",
            "respects_recall_ceiling":
                "yes" if (mean_of("recall") or 0) <= ceilings[scope]["recall"] else "no",
            "extractive":
                "yes" if runs[0].policy in ("quote", "n/a") else "no",
            "cells_differing_across_runs": differing_cells(runs),
            "mcnemar_p": fmt(test["p_value"]) if test else "",
            "mcnemar_discordant": (f"{test['system_only']}/{test['reference_only']}"
                                   if test else ""),
            "failures": sum(r.failures for r in runs),
            # Strict precision stays the primary number; these sit beside it (§4.5).
            "adjudicated_precision_lower": fmt(adjusted["lower"]) if adjusted else "",
            "adjudicated_precision": fmt(adjusted["point"]) if adjusted else "",
            "adjudicated_precision_upper": fmt(adjusted["upper"]) if adjusted else "",
        })

        # §4.4.5 — per field, averaged over runs.
        for fieldname in G.SCORED:
            gold_n = baseline["gold_per_field"][fieldname]
            sup_n = baseline["supported_per_field"].get(fieldname, 0)
            tp = statistics.fmean(r.per_field.get(fieldname, Counts()).tp for r in runs)
            fp = statistics.fmean(r.per_field.get(fieldname, Counts()).fp for r in runs)
            fn = statistics.fmean(r.per_field.get(fieldname, Counts()).fn for r in runs)
            c = Counts(tp=tp, fp=fp, fn=fn)
            field_rows.append({
                "system": system, "field": fieldname, "gold": gold_n,
                "supported": sup_n, "inference_only": gold_n - sup_n,
                "values_asserted": f"{tp + fp:.1f}",
                "tp": f"{tp:.1f}", "fp": f"{fp:.1f}", "fn": f"{fn:.1f}",
                "precision": fmt(c.precision), "recall": fmt(c.recall),
                "f1": fmt(c.f1),
                "indicative": "yes" if gold_n < config["fields"]["indicative_below"] else "",
            })

        # §4.4.7 — attribution, averaged over runs.
        causes = collections.Counter()
        for r in runs:
            for key, n in r.attribution.items():
                causes[key] += n
        for (kind, cause, fieldname), n in sorted(causes.items()):
            attr_rows.append({"system": system, "outcome": kind, "cause": cause,
                              "field": fieldname, "count": f"{n / len(runs):.1f}"})

        fp_total = statistics.fmean(r.counts.fp for r in runs)
        hallucinated = sum(n for (kind, cause, _), n in causes.items()
                           if kind == "FP" and cause == "hallucination") / len(runs)
        misattributed = sum(n for (kind, cause, _), n in causes.items()
                            if kind == "FP" and cause == "misattribution") / len(runs)
        stats["systems"][system] = {
            "evidence_scope": scope, "assertion_policy": runs[0].policy,
            "runs": len(runs),
            "precision": mean_of("precision"), "recall": mean_of("recall"),
            "f1": mean_of("f1"),
            "precision_sd": sd_of("precision"), "f1_sd": sd_of("f1"),
            "ci": ci or None,
            "recall_supported":
                statistics.fmean(r.counts.tp_supported for r in runs) / n_supported,
            "inference_yield":
                statistics.fmean(r.counts.tp_inference for r in runs) / n_inference,
            "hallucination_share": (hallucinated / fp_total) if fp_total else None,
            "misattribution_share": (misattributed / fp_total) if fp_total else None,
            "cells_differing_across_runs": differing_cells(runs),
            "mcnemar_vs_reference": test,
            "years_discarded": statistics.fmean(r.years_discarded for r in runs),
            "failures": sum(r.failures for r in runs),
            "adjudicated_precision": adjusted,
        }

    # §4.5 — the conditional. Computed always; adjudication only if it fires.
    shares = {s: v["hallucination_share"] for s, v in stats["systems"].items()
              if v["hallucination_share"] is not None}
    threshold = float(config["fp_diagnosis"]["hallucination_share_threshold"])
    worst = max(shares.values()) if shares else 0.0
    stats["fp_diagnosis"] = {
        "hallucination_share_threshold": threshold,
        "max_hallucination_share": worst,
        "max_system": max(shares, key=shares.get) if shares else None,
        "adjudication_required": worst > threshold,
    }

    if adjudication:
        rows = adjudication_rates(adjudication)
        write_csv(results / "adjudication.csv", rows, list(rows[0]))
        stats["adjudication"] = {
            "pool_size": adjudication["pool_size"], "judged": adjudication["drawn"],
            "whole_pool": adjudication["whole_pool"], "seed": adjudication["seed"],
            "by_field": {r["field"]: {k: r[k] for k in
                                      ("judged", "correct", "incorrect", "unverifiable",
                                       "rate_correct")} for r in rows},
        }

    write_csv(results / "system_cells.csv", per_run_rows, list(per_run_rows[0]))
    write_csv(results / "system_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(results / "field_breakdown.csv", field_rows, list(field_rows[0]))
    write_csv(results / "cell_outcomes.csv", cell_rows, list(cell_rows[0]))
    write_csv(results / "attribution.csv", attr_rows, list(attr_rows[0]))
    write_csv(results / "ceilings.csv",
              [{"scope": s, **{k: (fmt(v) if isinstance(v, float) else v)
                               for k, v in c.items()}} for s, c in ceilings.items()],
              ["scope", "asserted", "tp", "precision", "recall", "recall_supported", "f1"])
    (results / "stats.json").write_text(json.dumps(stats, indent=2) + "\n",
                                        encoding="utf-8")

    # ------------------------------------------------------------------ console
    print(f"gold: {baseline['gold_values']} values "
          f"({n_supported} supported / {n_inference} inference-only), "
          f"{n_cells} cells\n")
    print(f"{'system':14s} {'scope':5s} {'pol':5s} {'asserted':>8s} {'P':>6s} {'R':>6s} "
          f"{'F1':>6s} {'R_sup':>6s} {'infer':>6s} {'exhP':>6s} {'p':>7s}")
    for row in summary_rows:
        print(f"{row['system']:14s} {row['evidence_scope']:5s} "
              f"{row['assertion_policy'][:5]:5s} {row['values_asserted_mean']:>8s} "
              f"{row['precision']:>6s} {row['recall']:>6s} {row['f1']:>6s} "
              f"{row['recall_supported']:>6s} {row['inference_yield']:>6s} "
              f"{row['exhaustive_precision']:>6s} {row['mcnemar_p'] or '—':>7s}")
    print(f"\nhallucination share: max {worst:.3f} "
          f"({stats['fp_diagnosis']['max_system']}), threshold {threshold} -> "
          f"adjudication {'REQUIRED' if worst > threshold else 'not required'}")
    if adjudication:
        a = stats["adjudication"]["by_field"]["ALL"]
        print(f"adjudication: {a['judged']} rows judged "
              f"({'whole pool' if adjudication['whole_pool'] else 'sample'}) — "
              f"{a['correct']} correct / {a['incorrect']} incorrect / "
              f"{a['unverifiable']} unverifiable")
        print(f"\n{'system':14s} {'strict P':>9s} {'adj lower':>10s} {'adj point':>10s} "
              f"{'adj upper':>10s}")
        for row in summary_rows:
            if not row["adjudicated_precision"]:
                continue
            print(f"{row['system']:14s} {row['precision']:>9s} "
                  f"{row['adjudicated_precision_lower']:>10s} "
                  f"{row['adjudicated_precision']:>10s} "
                  f"{row['adjudicated_precision_upper']:>10s}")
    print(f"wrote {len(list(results.glob('*.csv')))} CSVs + stats.json to "
          f"{results.relative_to(HERE)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
