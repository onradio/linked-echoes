#!/usr/bin/env python3
"""Prompt assembly for the LLM systems (spec §3.5).

    python prompts.py --system M4 --object OBJ-003   # print one rendered prompt
    python prompts.py --system M6 --object OBJ-013 --show-fewshot

The blocks here replace `src/metadata_curation/templates/`'s `role_block` /
`instruction_block` pair, whose defects §6.6–§6.10 record. Three of those defects are
design faults rather than typos, and the structure below is what fixes them:

**The inference policy lives in exactly one block.** The superseded prompt forbade
inference in the role block (*"Avoid any creative reasoning or extrapolation"*) and
mandated it in the instruction block (*"Only infer genres, and ONLY if artists or
instruments suggest a clear genre"*). A model cannot obey both. The role block here states
who the model is and nothing else; the instruction block owns the policy, and the two
policies — `free` and `quote` — differ only in their final paragraph.

**Empty is the expected answer, stated first rather than as an escape hatch.** The old
prompt opened with *"You MUST populate all fields"* and mentioned empty values as an
afterthought. 73% of gold cells are empty, so that instruction was wrong for three cells in
four; §6.7 records it. The instruction here says so up front, and the few-shot block
demonstrates it — 19 of its 33 value cells are empty.

**`item_description` is not requested back.** The old schema asked for it, so the model
spent output rewriting its own input (§6.8).

The rendered prompt for a run is written to `predictions/manifest.json`, so what the model
actually saw is always on disk rather than reconstructed later from this file.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import gold as G

HERE = Path(__file__).resolve().parent

POLICY_FREE = "free"
POLICY_QUOTE = "quote"

ROLE_BLOCK = """You are a museum cataloguer working on a popular-music collection.
You extract structured metadata from the text supplied about an object."""

_INSTRUCTION_HEAD = """Return only values that this text supports for THIS object.

Most fields are empty for most objects. An empty list is the correct and expected
answer whenever the text does not support a value; it is not a failure to complete
the task."""

INSTRUCTION_FREE = _INSTRUCTION_HEAD + """

Where the text does not state a value but your knowledge of the object makes one
certain, you may supply it."""

INSTRUCTION_QUOTE = _INSTRUCTION_HEAD + """

For every value you return, also return `quote`: the exact words from the text that
the value comes from, copied character for character. If you cannot quote the text
for a value, do not return that value."""

ATTRIBUTION_BLOCK = """The caption below is displayed beside the whole display case. It describes SEVERAL
objects, not only this one. Names, works, years and places in it may belong to a
different object in the same case.

The other {object_word} in this case {is_are}:
{sibling_lines}

Return a value only if it belongs to THIS object. If something in the caption belongs
to one of the other objects listed above, leave it out."""


def instruction_block(policy: str) -> str:
    if policy == POLICY_FREE:
        return INSTRUCTION_FREE
    if policy == POLICY_QUOTE:
        return INSTRUCTION_QUOTE
    raise ValueError(f"unknown assertion policy {policy!r}")


def input_block(description: str, caption: str, *, include_caption: bool) -> str:
    """
    The object's text. The `Caption:` heading is **omitted entirely** when the scope
    excludes it or the caption is empty — not rendered as `None`, which is what the
    superseded template did and which teaches the model that a literal `None` is a
    thing that appears in museum records.
    """
    out = f"Object description:\n{description.strip()}"
    if include_caption and caption.strip():
        out += f"\n\nCaption:\n{caption.strip()}"
    return out


def attribution_block(sibling_descriptions: list) -> str:
    """
    `E3` only. Case sizes are 3 (×18), 2 (×2) and 4 (×1), so an object has 1, 2 or 3
    siblings and the singular has to be right — a prompt that says "the other objects"
    when there is one is a small thing that reads as carelessness to a model as much as
    to a person.
    """
    n = len(sibling_descriptions)
    lines = "\n".join(f"  - {d.strip()}" for d in sibling_descriptions)
    return ATTRIBUTION_BLOCK.format(
        object_word="object" if n == 1 else "objects",
        is_are="is" if n == 1 else "are",
        sibling_lines=lines)


# ----------------------------------------------------------------- few-shot examples

def load_fewshot(config: dict, path: Path | None = None) -> list:
    """The three held-out objects of §0.8, from this pack's own copy."""
    path = path or HERE / config["fewshot"]["sheet"]
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def find_quote(text: str, value: str, *, window: int = 60) -> str | None:
    """
    A quotable span for `value` in `text`, or None if the value does not occur in it.

    Used only for the few-shot examples, where the expected output has to model the
    `quote` policy. Two details matter, and both are about not teaching a bad habit:

    The value must occur on **word boundaries**. A naive substring search finds `Jazz`
    inside `Ken Colyer's Jazzmen`, which would produce an example quoting a fragment of
    another word to justify a genre — precisely the sloppiness the policy exists to
    prevent. With the boundary rule, `Jazz` is correctly unquotable there, and under
    `drop_unquotable_in_quote_policy` it disappears from the example, which is the honest
    lesson: that genre is an inference, so the quote policy declines it.

    Note this is deliberately *stricter* than the evidence check of §3.7, which follows
    the spec in using plain substring containment. The check governs what a system is
    allowed to assert; this governs what the prompt teaches, and there is no reason to
    teach the loosest thing the check would tolerate.

    The span is then widened to whole words, so an example never shows a quote starting
    mid-word.
    """
    escaped = re.escape(value)
    match = re.search(rf"(?<![0-9A-Za-z]){escaped}(?![0-9A-Za-z])", text,
                      flags=re.IGNORECASE)
    if match is None:
        return None
    i, j = match.start(), match.end()
    start, end = max(0, i - window // 2), min(len(text), j + window // 2)
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    return " ".join(text[start:end].split())


def fewshot_block(config: dict, policy: str, *, include_caption: bool,
                  examples: list | None = None) -> str:
    """
    The examples rendered in the shape the model must answer in.

    Under the `quote` policy a value with no literal quote in its example text is
    **dropped from the example**, because that is what the policy instructs the model to
    do: the block teaches the rule by obeying it rather than by contradicting it. That is
    why `Germany` disappears from the Ken Colyer example under `quote` — the leaflet is
    described as *German*, and no span of the text quotes `Germany`.
    """
    scored = list(config["fields"]["scored"])
    unscored = list(config["fields"]["extracted_unscored"])
    drop_unquotable = config["fewshot"].get("drop_unquotable_in_quote_policy", True)
    rows = examples if examples is not None else load_fewshot(config)

    chunks = []
    for n, row in enumerate(rows, 1):
        # Each block is searched separately, never a concatenation of them. A window
        # spanning the description/caption join would render a "quote" that appears
        # nowhere contiguously in the prompt, so the model would be shown a quote it
        # could not have copied — and the evidence check of §3.7 applies the same rule
        # for the same reason.
        segments = [row["item_description"]]
        if include_caption and row.get("caption"):
            segments.append(row["caption"])

        lines = [f"Example {n}", "",
                 input_block(row["item_description"], row.get("caption", ""),
                             include_caption=include_caption),
                 "", "Expected output:", "{"]
        body = []
        for field in scored:
            values = [v.strip() for v in (row.get(field) or "").split("|") if v.strip()]
            if policy == POLICY_QUOTE:
                items = []
                for v in values:
                    quote = next((q for q in (find_quote(seg, v) for seg in segments)
                                  if q), None)
                    if quote is None and drop_unquotable:
                        continue
                    items.append(f'{{"value": "{v}", "quote": "{quote}"}}')
                body.append(f'  "{field}": [{", ".join(items)}]')
            else:
                rendered = ", ".join(f'"{v}"' for v in values)
                body.append(f'  "{field}": [{rendered}]')
        for field in unscored:
            value = (row.get(field) or "").strip()
            if field == "cardinality":
                body.append(f'  "{field}": {value or 0}')
            else:
                body.append(f'  "{field}": "{value}"')
        lines.append(",\n".join(body))
        lines.append("}")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


# ------------------------------------------------------------------- full assembly

def build_prompt(gold: G.Gold, object_id: str, *, scope: str, policy: str,
                 config: dict, prompt_mode: str = "few-shot") -> str:
    """The whole prompt for one object under one system."""
    obj = gold.by_id[object_id]
    scopes = config["evidence_scopes"][scope]
    include_caption = bool(scopes.get("own_caption"))

    parts = [ROLE_BLOCK, instruction_block(policy)]

    if scopes.get("siblings"):
        parts.append(attribution_block(
            [gold.by_id[s]["item_description"] for s in obj["siblings"]]))

    if prompt_mode == "few-shot":
        parts.append(fewshot_block(config, policy, include_caption=include_caption))
    elif prompt_mode != "zero-shot":
        raise ValueError(f"unknown prompt mode {prompt_mode!r}")

    parts.append(input_block(obj["item_description"], obj["caption"],
                             include_caption=include_caption))
    return "\n\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", default="M3")
    ap.add_argument("--object", default="OBJ-001")
    ap.add_argument("--prompt-mode", default="few-shot",
                    choices=["few-shot", "zero-shot"])
    ap.add_argument("--show-fewshot", action="store_true",
                    help="print only the few-shot block")
    args = ap.parse_args()

    config = G.CONFIG
    systems = {s["id"]: s for s in config["systems"]}
    if args.system not in systems:
        raise SystemExit(f"unknown system {args.system!r}")
    system = systems[args.system]
    if system["kind"] != "llm":
        raise SystemExit(f"{args.system} is a {system['kind']} baseline, not an LLM cell")

    gold = G.Gold()
    if args.show_fewshot:
        include = bool(config["evidence_scopes"][system["scope"]].get("own_caption"))
        print(fewshot_block(config, system["policy"], include_caption=include))
        return 0

    prompt = build_prompt(gold, args.object, scope=system["scope"],
                          policy=system["policy"], config=config,
                          prompt_mode=args.prompt_mode)
    print(f"# {args.system}  scope={system['scope']}  policy={system['policy']}  "
          f"object={args.object}  chars={len(prompt)}")
    print("-" * 78)
    print(prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
