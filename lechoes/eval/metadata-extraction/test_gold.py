#!/usr/bin/env python3
"""Offline tests for gold.py — no API key, no network.

    pytest test_gold.py

Each test pins a defect that has actually occurred in this evaluation's history, so a
future edit that reintroduces one fails here rather than in a published number:

* `<infer>`-tagged gold that cannot match an untagged prediction (spec §6.2), including
  the two malformed closing tags the sheet shipped with — `,/infer>` and `</paper>`;
* commas treated as value separators, which would shred `Power, Corruption & Lies`;
* the ordering assumptions whose violation displaced all 62 rows of the superseded
  evaluator (§6.12, §6.13);
* possessive containment, where `Ken Colyer` appears only as `ken colyers jazzmen`.
"""
import csv
import shutil
from pathlib import Path

import pytest

import gold as G

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ normalisation

@pytest.mark.parametrize("raw, expected", [
    ("<infer> United Kingdom </infer>", "united kingdom"),
    ("<infer> Paper </paper>", "paper paper"),          # malformed: why the guard exists
    ("<infer> Power, Corruption & Lies,/infer>", "power corruption lies infer"),
    ("Do They Know It’s Christmas?", "do they know its christmas"),
    ("Do They Know It's Christmas", "do they know its christmas"),
])
def test_normalisation(raw, expected):
    assert G.normalise(raw) == expected


def test_tagged_gold_matches_untagged_prediction():
    """The §9 regression test: a tagged gold value must equal its untagged form."""
    tagged = G.split_cell("<infer> United Kingdom </infer>", "countries")
    plain = G.split_cell("United Kingdom", "countries")
    assert [v["value"] for v in tagged] == [v["value"] for v in plain]
    assert tagged[0]["tagged"] is True and plain[0]["tagged"] is False
    assert tagged[0]["surface"] == "United Kingdom"


def test_malformed_tags_would_not_match_without_the_fix():
    """Both malformed forms leave residue, which is precisely why they are blocking."""
    assert G.split_cell("<infer> Paper </paper>", "materials")[0]["value"] != "paper"
    assert (G.split_cell("<infer> Power, Corruption & Lies,/infer>",
                         "music_albums")[0]["value"] != "power corruption lies")


def test_article_stripping_applies_only_to_name_like_fields():
    assert G.normalise("The Who", "music_artists") == "who"
    assert G.normalise("The Rolling Stones", "music_artists") == "rolling stones"
    # `music_genres` is not name-like: a leading article is part of the term.
    assert G.normalise("The Blues", "music_genres") == "the blues"


def test_comma_is_not_a_value_separator():
    for cell, field in [("Power, Corruption & Lies", "music_albums"),
                        ("Metropolitan Club, Edge Road", "venues")]:
        assert len(G.split_cell(cell, field)) == 1, cell


def test_pipe_is_the_separator_and_duplicates_collapse():
    values = G.split_cell("Queen | Brian May | queen", "music_artists")
    assert [v["value"] for v in values] == ["queen", "brian may"]


# ------------------------------------------------------------------------ aliases

def test_alias_resolution_is_symmetric():
    assert G.canonicalise("countries", G.normalise("UK", "countries")) == \
           G.canonicalise("countries", G.normalise("United Kingdom", "countries"))


def test_alias_does_not_leak_across_fields():
    """`UK` is a country alias; it must not canonicalise a place of the same spelling."""
    assert G.canonicalise("places", G.normalise("UK", "places")) == "uk"


# -------------------------------------------------------------------- containment

def test_possessive_is_contained():
    """Why containment is naive substring: the name appears only in the genitive."""
    text = G.normalise("featuring Ken Colyer’s Jazzmen and Acker Bilk’s Paramount Jazz Band")
    assert G.contains(text, G.normalise("Ken Colyer", "music_artists"))
    assert not G.contains(text, G.normalise("Ken Colyer", "music_artists"), boundary=True)


# ------------------------------------------------------------- the sheet as frozen

@pytest.fixture(scope="module")
def sheet():
    return G.Gold()


def test_sheet_shape(sheet):
    assert len(sheet.objects()) == G.CONFIG["objects"]["n_expected"] == 62
    assert len(sheet.cases) == 21
    assert sorted(len(v) for v in sheet.cases.values()) == [2, 2] + [3] * 18 + [4]


def test_no_malformed_tags_survive(sheet):
    for o in sheet.objects():
        for f in G.SCORED:
            cell = sheet.raw_cell(o["object_id"], f)
            residue = G._WELL_FORMED_RE.sub("", cell)
            assert "<" not in residue and ">" not in residue, (o["object_id"], f, cell)


def test_ordering_assumptions_are_false(sheet):
    """The regression guard for §6.12 and §6.13."""
    cases = [o["case_number"] for o in sheet.objects()]
    assert cases != sorted(cases), "sheet must not be sorted by case"
    distinct = sorted(set(cases))
    assert distinct != list(range(min(distinct), max(distinct) + 1)), \
        "case numbers must not be contiguous — 18 and 22 are absent"
    assert 18 not in distinct and 22 not in distinct


def test_object_ids_are_stable_and_positional(sheet):
    ids = [o["object_id"] for o in sheet.objects()]
    assert ids[0] == "OBJ-001" and ids[-1] == "OBJ-062"
    assert len(set(ids)) == len(ids)


def test_held_out_examples_are_absent(sheet):
    held = {G.normalise(d) for d in G.CONFIG["objects"]["held_out_examples"]}
    assert not {G.normalise(o["item_description"]) for o in sheet.objects()} & held


def test_evidence_scopes_nest(sheet):
    for o in sheet.objects():
        oid = o["object_id"]
        e1, e2, e3 = (sheet.evidence(oid, s) for s in ("E1", "E2", "E3"))
        assert e1 in e2 and e2 in e3
        if not o["caption"]:
            assert e1 == e2, f"{oid} has no caption, so E1 and E2 must coincide"


def test_evidence_rejects_unknown_scope(sheet):
    with pytest.raises(ValueError):
        sheet.evidence("OBJ-001", "E2c")


def test_support_classes_split_as_frozen(sheet):
    counts = sheet.value_counts()
    assert counts["total"] == 209
    assert counts["supported"] == 173
    assert counts["total"] - counts["supported"] == 36


def test_distractors_match_the_frozen_baseline(sheet):
    dist = sheet.distractors()
    assert (dist["triples"], dist["objects"]) == (139, 49)
    # The three object-specific fields the section's argument rests on.
    for field, expected in [("music_artists", 45), ("music_works", 32),
                            ("music_albums", 13)]:
        assert dist["by_field"][field] == expected, field


def test_distractor_scope_is_case_wide_not_own_caption(sheet):
    """Own-caption-only yields 35/22 — a factor of four out. Pin the wider scope."""
    own_caption_only = set()
    for o in sheet.objects():
        oid = o["object_id"]
        text = G.normalise(sheet.evidence(oid, "E2"))
        mine = {(f, v) for f in G.SCORED for v in sheet.gold_values(oid, f)}
        for sib in o["siblings"]:
            for f in G.SCORED:
                for v in sheet.gold_values(sib, f):
                    if (f, v) not in mine and G.contains(text, v):
                        own_caption_only.add((oid, f, v))
    assert len(own_caption_only) == 35
    assert sheet.distractors()["triples"] > len(own_caption_only)


def test_ceilings_bound_every_extractive_system(sheet):
    ceil = sheet.ceilings()
    # Precision falls and recall rises monotonically as evidence widens: the trade-off
    # the section is built on, imposed by the corpus rather than by any prompt.
    assert ceil["E1"]["precision"] > ceil["E2"]["precision"] > ceil["E3"]["precision"]
    assert ceil["E1"]["recall"] < ceil["E2"]["recall"] < ceil["E3"]["recall"]
    assert ceil["E3"]["recall_supported"] == pytest.approx(1.0)
    assert ceil["E2"]["precision"] == pytest.approx(0.723, abs=0.001)


def test_events_holds_one_value_after_the_live_aid_correction(sheet):
    """§2: `Live Aid` was removed, not relabelled — the objects are Band Aid artefacts."""
    events = [v for o in sheet.objects()
              for v in sheet.gold_values(o["object_id"], "events")]
    assert events == ["new orleans joys"], events
    assert not any("live aid" in G.normalise(sheet.raw_cell(o["object_id"], "events"))
                   for o in sheet.objects())


def test_gold_vocabulary_covers_every_value(sheet):
    vocab = sheet.gold_vocabulary()
    for o in sheet.objects():
        for f in G.SCORED:
            for v in sheet.gold_values(o["object_id"], f):
                assert (f, v) in vocab


# ----------------------------------------------------------------- the verify gate

def _sheet_copy(tmp_path: Path, mutate) -> Path:
    rows = list(csv.DictReader(open(G.SHEET_PATH, newline="", encoding="utf-8")))
    mutate(rows)
    out = tmp_path / "mutated.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out


def test_verify_fails_on_a_reintroduced_malformed_tag(tmp_path, capsys):
    path = _sheet_copy(tmp_path, lambda rows: rows[1].update(
        {"materials": "<infer> Paper </paper>"}))
    shutil.copy(G.BASELINE_PATH, tmp_path / "keep.json")
    assert G.verify(G.Gold(path), freeze=False) == 1
    assert "malformed tag" in capsys.readouterr().out
    shutil.copy(tmp_path / "keep.json", G.BASELINE_PATH)


def test_verify_fails_when_the_sheet_is_sorted_by_case(tmp_path, capsys):
    path = _sheet_copy(tmp_path, lambda rows: rows.sort(key=lambda r: int(r["case"])))
    shutil.copy(G.BASELINE_PATH, tmp_path / "keep.json")
    assert G.verify(G.Gold(path), freeze=False) == 1
    assert "sorted-by-case" in capsys.readouterr().out
    shutil.copy(tmp_path / "keep.json", G.BASELINE_PATH)


def test_verify_warns_on_an_intra_cell_duplicate(tmp_path, capsys):
    path = _sheet_copy(tmp_path, lambda rows: rows[0].update(
        {"music_artists": "Suzi Quatro | suzi quatro"}))
    shutil.copy(G.BASELINE_PATH, tmp_path / "keep.json")
    G.verify(G.Gold(path), freeze=False)
    assert "duplicate value" in capsys.readouterr().out
    shutil.copy(tmp_path / "keep.json", G.BASELINE_PATH)


def test_verify_passes_on_the_frozen_sheet(capsys):
    assert G.verify(G.Gold(), freeze=False) == 0
    assert "all blocking checks pass" in capsys.readouterr().out
