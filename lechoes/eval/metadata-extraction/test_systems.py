#!/usr/bin/env python3
"""Offline tests for the Stage 2 runners — no API key, no network.

    pytest test_systems.py

The evidence check and the prompt assembly are where this stage can corrupt results
silently, so both are pinned here. Nothing in this file makes a call: `run_llm.py`'s
generation is a thin wrapper around `src/llm.py`, and what needs testing is the logic
either side of it.
"""
from __future__ import annotations

import re

import pytest

import gold as G
import predictions as P
import prompts as PR
import run_llm as R

CONFIG = G.CONFIG
SCORED = list(CONFIG["fields"]["scored"])


@pytest.fixture(scope="module")
def sheet():
    return G.Gold()


# ------------------------------------------------------------------ prompt assembly

def test_role_block_states_no_inference_policy():
    """§6.6: the superseded prompt forbade inference in one block and required it in
    another. The policy must live in exactly one place."""
    role = PR.ROLE_BLOCK.lower()
    for word in ("infer", "extrapolat", "invent", "creative"):
        assert word not in role, word


def test_the_two_policies_differ_only_in_their_last_paragraph():
    free = PR.INSTRUCTION_FREE.split("\n\n")
    quote = PR.INSTRUCTION_QUOTE.split("\n\n")
    assert free[:-1] == quote[:-1]
    assert "quote" in quote[-1].lower() and "quote" not in free[-1].lower()


def test_empty_is_presented_as_correct_not_as_an_escape_hatch():
    """§6.7: `You MUST populate all fields` was wrong for three cells in four."""
    head = PR.INSTRUCTION_FREE.lower()
    assert "empty list is the correct and expected" in head
    assert "must populate" not in head


def test_caption_is_omitted_rather_than_rendered_as_none():
    block = PR.input_block("A description", "", include_caption=True)
    assert "Caption:" not in block and "None" not in block
    block = PR.input_block("A description", "A caption", include_caption=False)
    assert "Caption:" not in block


def test_attribution_block_is_singular_for_one_sibling():
    one = PR.attribution_block(["Only sibling"])
    assert "The other object in this case is:" in one
    two = PR.attribution_block(["First", "Second"])
    assert "The other objects in this case are:" in two


def test_e1_prompt_carries_no_caption_and_e3_carries_siblings(sheet):
    captioned = next(o for o in sheet.objects() if o["caption"] and o["siblings"])
    oid = captioned["object_id"]
    e1 = PR.build_prompt(sheet, oid, scope="E1", policy="free", config=CONFIG)
    e2 = PR.build_prompt(sheet, oid, scope="E2", policy="free", config=CONFIG)
    e3 = PR.build_prompt(sheet, oid, scope="E3", policy="free", config=CONFIG)
    assert captioned["caption"][:40] not in e1
    assert captioned["caption"][:40] in e2
    assert "display case" in e3 and "display case" not in e2
    sibling = sheet.by_id[captioned["siblings"][0]]["item_description"]
    assert sibling[:30] in e3 and sibling[:30] not in e2


def test_zero_shot_drops_the_examples(sheet):
    few = PR.build_prompt(sheet, "OBJ-001", scope="E2", policy="free", config=CONFIG,
                          prompt_mode="few-shot")
    zero = PR.build_prompt(sheet, "OBJ-001", scope="E2", policy="free", config=CONFIG,
                           prompt_mode="zero-shot")
    assert "Example 1" in few and "Example 1" not in zero
    assert len(zero) < len(few)


def test_prompt_never_requests_item_description_back(sheet):
    """§6.8: the old schema asked for it, so the model rewrote its own input."""
    fields = R.metadata_fields(CONFIG, "free")
    assert "item_description" not in fields.names()
    assert "base_type" not in fields.names()          # §1.1: that is the category task
    assert len(fields.names()) == 14


# --------------------------------------------------------------------- few-shot block

def test_fewshot_demonstrates_the_empty_list():
    block = PR.fewshot_block(CONFIG, "free", include_caption=True)
    assert block.count(": []") >= 10, "the examples must show empty is correct"


def test_fewshot_quote_is_a_contiguous_span_of_one_block():
    """A window spanning the description/caption join is not a quotable span."""
    rows = PR.load_fewshot(CONFIG)
    block = PR.fewshot_block(CONFIG, "quote", include_caption=True)
    segments = [G.normalise(s) for row in rows
                for s in (row["item_description"], row.get("caption", ""))]
    quotes = re.findall(r'"quote": "((?:[^"\\]|\\.)*)"', block)
    assert quotes, "the quote-policy examples must carry quotes"
    for quote in quotes:
        norm_quote = G.normalise(quote)
        assert any(G.contains(seg, norm_quote) for seg in segments), quote


def test_fewshot_drops_a_value_it_cannot_quote():
    """`Germany` is an inference from `German`, so the quote policy declines it."""
    free = PR.fewshot_block(CONFIG, "free", include_caption=True)
    quoted = PR.fewshot_block(CONFIG, "quote", include_caption=True)
    assert "Germany" in free
    assert "Germany" not in quoted


def test_find_quote_respects_word_boundaries():
    text = "German promotional leaflet for Ken Colyer's Jazzmen"
    assert PR.find_quote(text, "Jazz") is None      # inside `Jazzmen`, not a match
    assert PR.find_quote(text, "Ken Colyer") is not None


# -------------------------------------------------------------------- response parsing

def test_years_arriving_as_json_numbers_survive():
    """§4.1 rule 3: the schema says string, models return 1973."""
    parsed = {"years": [1973, "1978"], "music_artists": ["Queen"]}
    predicted, _, ok = R.parse_response(parsed, CONFIG, "free")
    assert ok
    assert [i["value"] for i in predicted["years"]] == ["1973", "1978"]


def test_a_missing_field_becomes_an_empty_list_not_a_failure():
    predicted, _, ok = R.parse_response({"music_artists": ["Queen"]}, CONFIG, "free")
    assert ok
    assert predicted["music_genres"] == []
    assert set(predicted) == set(SCORED)


def test_a_non_object_response_is_a_parse_failure():
    _, _, ok = R.parse_response(["not", "an", "object"], CONFIG, "free")
    assert not ok


def test_unscored_fields_are_captured_separately():
    parsed = {"item_title": "A title", "specific_type": "Poster", "cardinality": 2}
    predicted, unscored, ok = R.parse_response(parsed, CONFIG, "quote")
    assert ok and unscored == parsed
    assert all(v == [] for v in predicted.values())


# --------------------------------------------------------------------- evidence check

SEGMENTS = ["Suzi Quatro Black leather jumpsuit",
            "For the 1973 performance of Can The Can on Top of the Pops"]


def _check(items):
    return R.evidence_check({f: items.get(f, []) for f in SCORED}, SEGMENTS, CONFIG)


def test_a_quoted_value_present_in_a_block_survives():
    kept, dropped, join = _check(
        {"music_artists": [{"value": "Suzi Quatro", "quote": "Suzi Quatro Black leather"}]})
    assert [i["value"] for i in kept["music_artists"]] == ["Suzi Quatro"]
    assert dropped == [] and join == 0


def test_a_value_with_no_quote_is_dropped():
    kept, dropped, _ = _check({"music_genres": [{"value": "Glam rock", "quote": None}]})
    assert kept["music_genres"] == []
    assert dropped[0]["reason"] == "no_quote"


def test_a_fabricated_quote_is_dropped():
    kept, dropped, _ = _check(
        {"venues": [{"value": "Wembley", "quote": "played at Wembley Stadium"}]})
    assert kept["venues"] == []
    assert dropped[0]["reason"] == "quote_not_in_text"


def test_a_real_quote_with_an_unrelated_value_is_dropped():
    """The failure where a model quotes a real sentence and attaches anything to it."""
    kept, dropped, _ = _check(
        {"music_artists": [{"value": "David Bowie", "quote": "Suzi Quatro Black leather"}]})
    assert kept["music_artists"] == []
    assert dropped[0]["reason"] == "value_not_in_quote"


def test_a_quote_spanning_two_blocks_is_dropped_and_counted_separately():
    """Stricter than the letter of §3.7, so its effect is measured, not assumed."""
    spanning = "leather jumpsuit For the 1973 performance"
    kept, dropped, join = _check(
        {"years": [{"value": "1973", "quote": spanning}]})
    assert kept["years"] == []
    assert join == 1 and dropped[0]["reason"] == "quote_spans_blocks"


def test_the_check_cannot_remove_a_siblings_value(sheet):
    """Stated in §3.7 and true by construction: the sibling's artist IS in the text."""
    oid = next(o["object_id"] for o in sheet.objects()
               if sheet.sibling_distractors(o["object_id"]))
    field, value = next(iter(sheet.sibling_distractors(oid)))
    segments = R.evidence_segments(sheet, oid, "E3", CONFIG)
    quote = next(s for s in segments if G.contains(G.normalise(s), value))
    kept, dropped, _ = R.evidence_check(
        {f: ([{"value": value, "quote": quote}] if f == field else []) for f in SCORED},
        segments, CONFIG)
    assert [i["value"] for i in kept[field]] == [value], dropped


# ------------------------------------------------------------------- record invariants

def test_every_scored_field_is_present_in_an_empty_prediction():
    empty = P.empty_prediction(SCORED)
    assert set(empty) == set(SCORED) and all(v == [] for v in empty.values())


def test_dedupe_collapses_on_the_normalised_form():
    items = [{"value": "The Beatles", "quote": "a"}, {"value": "Beatles", "quote": "b"},
             {"value": "Queen", "quote": "c"}]
    kept = P.dedupe(items, G.normalise, "music_artists")
    assert [i["value"] for i in kept] == ["The Beatles", "Queen"]
    assert kept[0]["quote"] == "a", "the first surface form and its quote are kept"


def test_a_failed_object_is_null_not_an_abstention():
    r = P.record(object_id="OBJ-001", case_number=1, system="M3", model="m", run=1,
                 evidence_scope="E2", assertion_policy="free", predicted=None,
                 error="APIError: boom")
    assert r["predicted"] is None and r["error"]
    assert r["dropped_no_quote"] == [] and r["unscored"] == {}


def test_baseline_records_declare_no_assertion_policy():
    path = P.prediction_path(CONFIG, "M0a", None, 1)
    if not path.exists():
        pytest.skip("M0a has not been run")
    for r in P.read(path):
        assert r["assertion_policy"] == "n/a"
        assert r["model"] is None
        assert set(r["predicted"]) == set(SCORED)
        for items in r["predicted"].values():
            for item in items:
                assert item["quote"], "a span tagger always has its span"


def test_prediction_filenames_carry_the_model_only_for_llm_cells():
    assert P.prediction_path(CONFIG, "M0a", None, 1).name == "M0a_run1.jsonl"
    assert (P.prediction_path(CONFIG, "M3", "google-gemini-2-5-flash", 2).name
            == "M3_google-gemini-2-5-flash_run2.jsonl")
