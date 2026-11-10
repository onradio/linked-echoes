#!/usr/bin/env python3
"""Offline tests for score.py — the §6.1–§6.5 defects, each with the case that exposed it.

    pytest test_scoring.py

Every defect in the spec's blocking list gets a test that fails if it is reintroduced. The
two the superseded evaluator got wrong silently — the transposed P/R and the greedy matcher
— are the reason this file exists: both produced plausible-looking numbers.
"""
from __future__ import annotations

import pytest

import gold as G
import score as S

CONFIG = G.CONFIG
POLICIES = CONFIG["matching"]["policy"]


@pytest.fixture(scope="module")
def sheet():
    return G.Gold()


def norm(field, values):
    return [G.normalise(v, field) for v in values]


def dedupe(field, values):
    """§4.1 rule 1: duplicates collapse before matching, on the normalised form."""
    return list(dict.fromkeys(norm(field, values)))


def counts_for(field, preds, golds):
    matched, fps, fns = S.match_cell(field, dedupe(field, preds), dedupe(field, golds),
                                     POLICIES[field])
    return S.Counts(tp=len(matched), fp=len(fps), fn=len(fns))


def greedy_tp(field, preds, golds, threshold=85):
    """
    `process.extractOne` semantics: each prediction takes its single best gold, and is
    discarded rather than reassigned if that gold is already taken (§6.5).
    """
    used, tp = set(), 0
    for p in preds:
        best, bj = max((S.pair_score(field, p, g, POLICIES[field])[0], j)
                       for j, g in enumerate(golds))
        if best >= threshold and bj not in used:
            used.add(bj)
            tp += 1
    return tp


# ------------------------------------------------------- §6.5 the greedy-matching defect

def test_the_specs_rolling_stones_example_collapses_under_its_own_normalisation():
    """
    §9 asks for a test asserting `tp=2` on the spec's §6.5 example. It is unreachable, and
    not because of the matcher: `music_artists` is name-like, so §4.1 strips the leading
    article, and `Rolling Stones` and `The Rolling Stones` become the *same* normalised
    value. Under §4.1 rule 1 that is one assertion, not two, so the cell holds a single
    prediction and the best any algorithm can do is one true positive.

    The example was written from the raw fuzzy scores (95 / 90 / 100), which only arise
    without article stripping. The defect it illustrates is real; this instance of it is
    not. `test_optimal_assignment_beats_greedy` covers the real property instead.
    """
    assert (G.normalise("Rolling Stones", "music_artists")
            == G.normalise("The Rolling Stones", "music_artists") == "rolling stones")
    c = counts_for("music_artists",
                   ["Rolling Stones", "The Rolling Stones"],
                   ["The Rolling Stones", "Rolling Stones Fan Club"])
    assert (c.tp, c.fp, c.fn) == (1, 0, 1)


def test_optimal_assignment_beats_greedy():
    """
    A case where first-best matching loses a true positive it did not have to: `p2` matches
    `g1` exactly, but greedy lets `p1` take `g1` on a lower score first and then discards
    `p2` instead of reassigning.
    """
    golds = ["abc def ghi x", "abc def ghi xy"]
    preds = ["abc def ghi", "abc def ghi x"]
    matched, _, _ = S.match_cell("music_works", preds, golds, POLICIES["music_works"])
    assert greedy_tp("music_works", preds, golds) == 1
    assert len(matched) == 2


def test_optimal_assignment_is_never_worse_than_greedy_on_real_gold(sheet):
    """
    Over randomised cells drawn from the actual gold vocabulary, the optimal assignment can
    only tie or beat greedy. On this corpus it always *ties*: no two gold values within a
    field score even 70 against each other, so the assignment choice cannot move any
    reported number here. That is a measured null result, not a reason to keep greedy.
    """
    import random
    rng = random.Random(11)
    fuzzy_fields = [f for f, p in POLICIES.items() if p.get("fuzzy")]
    ties = 0
    for _ in range(2000):
        field = rng.choice(fuzzy_fields)
        vals = sorted({v for (f, v) in sheet.gold_vocabulary() if f == field})
        if len(vals) < 2:
            continue
        golds = rng.sample(vals, k=min(len(vals), rng.choice([1, 2, 3])))
        preds = list(dict.fromkeys(rng.choice(vals) for _ in range(rng.choice([1, 2, 3]))))
        opt = len(S.match_cell(field, preds, golds, POLICIES[field])[0])
        grd = greedy_tp(field, preds, golds)
        assert opt >= grd, (field, preds, golds, opt, grd)
        ties += (opt == grd)
    assert ties > 0


def test_matching_is_independent_of_list_order():
    forward = counts_for("music_works", ["abc def ghi", "abc def ghi x"],
                         ["abc def ghi x", "abc def ghi xy"])
    reverse = counts_for("music_works", ["abc def ghi x", "abc def ghi"],
                         ["abc def ghi xy", "abc def ghi x"])
    assert (forward.tp, forward.fp, forward.fn) == (reverse.tp, reverse.fp, reverse.fn)


def test_a_prediction_is_never_matched_twice():
    c = counts_for("music_artists", ["Queen"], ["Queen", "Queen"])
    assert (c.tp, c.fp) == (1, 0)


# ------------------------------------------------------- §6.1 the transposition defect

def test_precision_and_recall_are_not_transposed():
    """
    One prediction, right, against three gold values: precision 1.0, recall 1/3. A
    transposed implementation reports the reverse, and F₁ cannot tell you — the harmonic
    mean is symmetric, which is exactly why the defect survived 33 runs.
    """
    c = counts_for("music_artists", ["Queen"], ["Queen", "David Bowie", "Suzi Quatro"])
    assert (c.tp, c.fp, c.fn) == (1, 0, 2)
    assert c.precision == pytest.approx(1.0)
    assert c.recall == pytest.approx(1 / 3)
    assert c.precision != c.recall, "the asymmetric case is the whole point"


def test_asserting_into_an_empty_cell_cannot_raise_recall():
    """§0.6/§4.4.1: recall is only winnable on non-empty cells."""
    c = counts_for("music_genres", ["Punk", "Ska"], [])
    assert (c.tp, c.fp, c.fn) == (0, 2, 0)
    assert c.recall is None       # no gold here to recover
    assert c.precision == 0.0


# ------------------------------------------------------------ §4.1 the matching policy

def test_short_closed_classes_do_not_fuzzy_match():
    """`WRatio('rock','rock and roll')=90` would credit Rock with Rock and Roll."""
    c = counts_for("music_genres", ["Rock"], ["Rock and Roll"])
    assert (c.tp, c.fp, c.fn) == (0, 1, 1)


def test_name_like_fields_do_fuzzy_match_above_the_threshold():
    c = counts_for("music_works", ["Do They Know Its Christmas"],
                   ["Do They Know It's Christmas?"])
    assert c.tp == 1


def test_article_stripping_matches_by_normalisation_not_fuzziness():
    c = counts_for("music_artists", ["The Who"], ["Who"])
    assert c.tp == 1


def test_alias_table_is_honoured_and_scoped_to_its_field():
    assert counts_for("countries", ["UK"], ["United Kingdom"]).tp == 1
    # `places` has no such alias, and the fuzzy scorer must not rescue it either.
    assert counts_for("places", ["UK"], ["United Kingdom"]).tp == 0


def test_alias_does_not_beat_an_exact_match_for_the_same_slot():
    """Exact scores 100 and alias 99, so the assignment prefers the exact pairing."""
    matched, _, _ = S.match_cell(
        "countries", norm("countries", ["United Kingdom", "UK"]),
        norm("countries", ["United Kingdom"]), POLICIES["countries"])
    assert len(matched) == 1
    assert matched[0][3] == S.EXACT


def test_cells_are_scored_independently():
    """§4.1 rule 2: `London` under both places and venues is one TP and one FP."""
    good = counts_for("places", ["London"], ["London"])
    bad = counts_for("venues", ["London"], [])
    assert (good.tp, good.fp) == (1, 0)
    assert (bad.tp, bad.fp) == (0, 1)


# ------------------------------------------------------------------- years, §4.1 rule 3

def _year_record(values):
    return {"predicted": {f: ([{"value": v, "quote": None} for v in values]
                              if f == "years" else []) for f in G.SCORED}}


def test_a_year_returned_as_a_json_number_is_kept():
    values, discarded = S.prepare_predicted(_year_record([1973]), "years", CONFIG)
    assert values == ["1973"] and discarded == 0


def test_a_non_four_digit_year_is_discarded_and_counted():
    """`1970s` is a decade, not a year: stripping it to `1970` would invent precision the
    model declined to assert."""
    values, discarded = S.prepare_predicted(
        _year_record(["68", "1970s", "September 1973", "1973"]), "years", CONFIG)
    assert values == ["1973"]
    assert discarded == 3, "discarded values must be counted, not silently dropped"


def test_years_do_not_fuzzy_match():
    c = counts_for("years", ["1974"], ["1973"])
    assert (c.tp, c.fp, c.fn) == (0, 1, 1)


# ---------------------------------------------------------------- undefined ratios

def test_precision_is_undefined_rather_than_perfect_when_nothing_is_asserted():
    """§4.4.1: 1.000 would let a system that says nothing top a precision ranking."""
    c = S.Counts(tp=0, fp=0, fn=5)
    assert c.precision is None
    assert S.fmt(c.precision) == "—"
    assert c.recall == 0.0
    assert c.f1 is None


def test_f1_is_zero_when_both_components_are_defined_and_zero():
    c = S.Counts(tp=0, fp=3, fn=3)
    assert c.f1 == 0.0


# --------------------------------------------------------------------- attribution

def test_a_siblings_value_is_attributed_to_misattribution(sheet):
    oid = next(o["object_id"] for o in sheet.objects()
               if sheet.sibling_distractors(o["object_id"]))
    field, value = next(iter(sheet.sibling_distractors(oid)))
    cause = S.attribute_fp(sheet, oid, field, value, "E3", POLICIES[field])
    assert cause == "misattribution"


def test_a_value_recorded_under_another_field_is_wrong_field(sheet):
    oid = next(o["object_id"] for o in sheet.objects()
               if sheet.gold_values(o["object_id"], "music_works")
               and not sheet.gold_values(o["object_id"], "music_albums"))
    value = next(iter(sheet.gold_values(oid, "music_works")))
    # The same value offered as an album: recorded, but under a different field.
    cause = S.attribute_fp(sheet, oid, "music_albums", value, "E2",
                           POLICIES["music_albums"])
    assert cause in ("wrong_field", "misattribution")


def test_a_value_absent_from_the_evidence_is_a_hallucination(sheet):
    cause = S.attribute_fp(sheet, "OBJ-001", "venues",
                           G.normalise("Madison Square Garden", "venues"), "E1",
                           POLICIES["venues"])
    assert cause == "hallucination"


def test_an_inference_only_gold_value_is_out_of_scope_as_a_false_negative(sheet):
    found = None
    for o in sheet.objects():
        for f in G.SCORED:
            for v in sheet.gold_values(o["object_id"], f):
                if sheet.support_class(o["object_id"], f, v) == "inference_only":
                    found = (o["object_id"], f, v)
                    break
            if found:
                break
        if found:
            break
    oid, f, v = found
    assert S.attribute_fn(sheet, oid, f, v, "E3") == "out_of_scope"


# ------------------------------------------------------------------ cell outcomes

def test_the_five_cell_outcomes_are_exhaustive():
    from collections import Counter
    import predictions as P
    import json
    path = P.prediction_path(CONFIG, "M3", "google-gemini-2-5-flash", 1)
    if not path.exists():
        pytest.skip("M3 has not been run")
    sheet = G.Gold()
    run = S.score_run(sheet, CONFIG, P.read(path), system="M3", run=1, scope="E2",
                      policy_name="free", model="google-gemini-2-5-flash")
    assert sum(run.cells.values()) == 682, "every (object, field) gets exactly one outcome"
    assert set(run.cells) <= {"HIT", "WRONG", "MISS", "SPURIOUS", "ABSTAIN"}


# ------------------------------------------------------------------------ statistics

def test_bootstrap_returns_ordered_bounds(sheet):
    per_object = {o["object_id"]: S.Counts(tp=2, fp=1, fn=1) for o in sheet.objects()}
    ci = S.bootstrap_ci(per_object, sheet, resamples=200)
    for key in ("precision", "recall", "f1"):
        low, high = ci[key]
        assert low is not None and low <= high


def test_mcnemar_counts_discordant_cells_in_both_directions():
    system = {("a", "f"): True, ("b", "f"): False, ("c", "f"): True}
    reference = {("a", "f"): False, ("b", "f"): True, ("c", "f"): True}
    out = S.mcnemar(system, reference)
    assert (out["system_only"], out["reference_only"], out["discordant"]) == (1, 1, 2)
    assert 0.0 <= out["p_value"] <= 1.0


def test_majority_vote_takes_the_modal_outcome():
    runs = [type("R", (), {"per_cell_exact": {("a", "f"): True}})(),
            type("R", (), {"per_cell_exact": {("a", "f"): True}})(),
            type("R", (), {"per_cell_exact": {("a", "f"): False}})()]
    assert S.majority_vote(runs) == {("a", "f"): True}
    assert S.differing_cells(runs) == 1


# ------------------------------------------------- what the exhaustive extractor bounds

def test_recall_ceiling_binds_extractive_systems_only(sheet):
    """
    The recall column of §2.4 is a real bound, but only for systems that assert what they
    can see. Quote-policy cells and the two span taggers are extractive by construction and
    must respect it; a free-policy cell may exceed it by inferring, and `M1` does.
    """
    import csv as _csv
    import pathlib as _pathlib
    path = _pathlib.Path("data/results/system_summary.csv")
    if not path.exists():
        pytest.skip("scores have not been computed")
    ceilings = sheet.ceilings()
    breaches = []
    for row in _csv.DictReader(open(path, encoding="utf-8")):
        if row["system"] == "M3-zeroshot":
            continue
        recall = float(row["recall"])
        ceiling = ceilings[row["evidence_scope"]]["recall"]
        if row["assertion_policy"] in ("quote", "n/a"):
            assert recall <= ceiling + 1e-9, (
                f"{row['system']} is extractive but exceeds its scope's recall ceiling")
        elif recall > ceiling:
            breaches.append(row["system"])
    assert breaches, ("no free-policy system exceeded a recall ceiling; if that is now true, "
                      "the claim in §2.4 that inference can exceed it needs re-checking")


def test_exhaustive_precision_is_not_treated_as_a_bound(sheet):
    """
    Exceeding it is the finding, not an error. Pinned so nobody 'fixes' the scorer to clamp
    it: the exhaustive extractor maximises assertions, so a subset of them can be — and
    here is — more precise.
    """
    import csv as _csv
    import pathlib as _pathlib
    path = _pathlib.Path("data/results/system_summary.csv")
    if not path.exists():
        pytest.skip("scores have not been computed")
    rows = list(_csv.DictReader(open(path, encoding="utf-8")))
    beating = [r["system"] for r in rows if r["beats_exhaustive_precision"] == "yes"]
    assert beating, "at least one attributing cell should beat exhaustive precision"
    for r in rows:
        assert "ceiling_precision" not in r, "the column must not be named a ceiling"


# ------------------------------------------------------------------- adjudication

def _adj():
    return S.load_adjudication(CONFIG)


def test_adjudication_sheet_matches_the_frozen_sample():
    """A sheet that is not the sample it was drawn from is refused, not partially used."""
    adj = _adj()
    if adj is None:
        pytest.skip("no adjudication sheet")
    import json as _json
    import pathlib as _pathlib
    sample = _json.loads(
        (_pathlib.Path("data") / "adjudication_sample.json").read_text(encoding="utf-8"))
    assert set(adj["verdicts"]) == {tuple(k) for k in sample["keys"]}
    assert set(adj["verdicts"].values()) <= set(S.VERDICTS)


def test_every_hallucination_has_a_verdict(sheet):
    """
    The bracket is only exact if the pool was judged whole. If a hallucinated value had no
    verdict it would silently contribute zero, understating the correction.
    """
    adj = _adj()
    if adj is None:
        pytest.skip("no adjudication sheet")
    assert adj["whole_pool"], "pool was sampled; the estimator must extrapolate instead"
    import json as _json
    import pathlib as _pathlib
    import predictions as _P
    manifest = _json.loads(
        (_P.paths(CONFIG)["predictions"] / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["entries"].values():
        path = _P.prediction_path(CONFIG, entry["system"], entry["model_slug"],
                                  entry["run"])
        run = S.score_run(sheet, CONFIG, _P.read(path), system=entry["system"],
                          run=entry["run"], scope=entry["evidence_scope"],
                          policy_name=entry["assertion_policy"],
                          model=entry["model_slug"])
        for key in run.hallucinated:
            assert key in adj["verdicts"], f"{entry['system']}: {key} has no verdict"


def test_the_bracket_is_ordered_and_brackets_strict_precision():
    import csv as _csv
    import pathlib as _pathlib
    path = _pathlib.Path("data/results/system_summary.csv")
    if not path.exists():
        pytest.skip("scores have not been computed")
    for row in _csv.DictReader(open(path, encoding="utf-8")):
        if not row["adjudicated_precision"]:
            continue
        strict = float(row["precision"])
        lower = float(row["adjudicated_precision_lower"])
        point = float(row["adjudicated_precision"])
        upper = float(row["adjudicated_precision_upper"])
        assert lower <= point <= upper, row["system"]
        # Crediting false positives can only raise precision, never lower it.
        assert lower >= strict - 1e-9, row["system"]


def test_quote_policy_cells_are_unmoved_by_adjudication():
    """
    A value whose quote must occur in the input cannot be classed a hallucination, so there
    is nothing for the correction to credit. Their strict precision is the number itself.
    """
    import csv as _csv
    import pathlib as _pathlib
    path = _pathlib.Path("data/results/system_summary.csv")
    if not path.exists():
        pytest.skip("scores have not been computed")
    for row in _csv.DictReader(open(path, encoding="utf-8")):
        if row["assertion_policy"] != "quote" or not row["adjudicated_precision"]:
            continue
        assert row["adjudicated_precision_upper"] == row["precision"], row["system"]
