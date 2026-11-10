#!/usr/bin/env python3
"""Stage 1 — the gold standard, and the only module that parses the sheet (spec §2).

    python gold.py                  # the summary: values, support, distractors, ceilings
    python gold.py --verify         # the same, plus the data-quality gate; non-zero on failure
    python gold.py --verify --freeze # re-baseline data/gold_baseline.json after a re-annotation

Every other stage imports this module rather than reading `data/gold_objects.csv`, so
normalisation, value splitting, support classes and evidence assembly are defined once.
`run_llm.py`'s evidence check and `score.py`'s matching both call `normalise` from here:
two normalisations would mean a value could be "in the text" for one stage and not the
other, which is exactly the class of silent divergence this evaluation exists to avoid.

Three decisions are load-bearing and are recorded here rather than in a reader's memory:

**Support classes and distractors are computed at case-wide caption scope**
(`description_plus_case_captions`), not at the object's own caption. Measured on this
sheet, own-caption-only yields 35 sibling distractors over 22 objects where the case-wide
scope yields 139 over 49 — a factor of four on the number §0.1 calls the finding the
section is built on. Captions are written per display case, so a value "visible to" an
object is one visible anywhere in its case. `config.yaml: distractor_scope` names it.

**Containment is naive substring, on the normalised forms.** It is what §2.4 specifies and
what reproduces the spec's supported count of 173 exactly. A token-boundary variant is
computed alongside it and reported by `--verify`, because naive substring can in principle
credit `us` inside `museum`. Measured, it makes exactly two values differ and **naive
substring is right in both**: `Ken Colyer` and `Acker Bilk` appear in the text only as
possessives — `ken colyers jazzmen`, `acker bilks paramount jazz band` — which a boundary
rule would class inference-only even though the name is plainly there. Since apostrophes are
stripped before matching, the looser rule absorbs the genitive for free. The label never
changes a hit or a miss (§2.2) in any case.

**The counts are frozen on disk, not asserted as literals.** `data/gold_baseline.json`
records them with the sheet's SHA-256. `--verify` fails if the sheet has changed without
`--freeze`, which catches a silent edit; a deliberate re-annotation re-freezes and the new
counts flow through every stage. The spec's §9 previously hard-coded numbers that a single
annotation fix invalidated, and this is the replacement.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
BASELINE_PATH = HERE / "data" / "gold_baseline.json"

_TAG_RE = re.compile(r"<\s*/?\s*infer\s*>", re.IGNORECASE)
_WELL_FORMED_RE = re.compile(r"<infer>.*?</infer>", re.IGNORECASE | re.DOTALL)
_ANY_TAG_RE = re.compile(r"<[^>]*>|[^<\s][/]\s*\w+\s*>")
_YEAR_RE = re.compile(r"^\d{4}$")


# --------------------------------------------------------------------------- config

def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


CONFIG = load_config()
SCORED = list(CONFIG["fields"]["scored"])
UNSCORED = list(CONFIG["fields"]["extracted_unscored"])
NAME_LIKE = set(CONFIG["matching"]["strip_articles_in"])
SHEET_PATH = HERE / CONFIG["objects"]["sheet"]
ALIAS_PATH = HERE / CONFIG["matching"]["aliases"]
SCOPES = ("E1", "E2", "E3")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------- normalising

def normalise(text, field: str | None = None) -> str:
    """
    The §4.1 normalisation, applied to gold, predictions and evidence text alike.

    Strip `<infer>` tags, NFKD, drop combining marks, lowercase, remove apostrophes,
    replace every other non-alphanumeric with a space, collapse whitespace. For the
    name-like fields, additionally drop a leading article and a trailing parenthetical,
    which is what makes `The Who` == `Who` by normalisation rather than by fuzziness.
    """
    text = _TAG_RE.sub(" ", str(text))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if field in NAME_LIKE:
        text = re.sub(r"^(the|a|an)\s+", "", text)
        text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    return text


def contains(haystack: str, needle: str, *, boundary: bool = False) -> bool:
    """Whether a normalised value occurs in normalised text. See the module docstring."""
    if not needle:
        return False
    if boundary:
        return f" {needle} " in f" {haystack} "
    return needle in haystack


def split_cell(raw, field: str) -> list:
    """
    The values in one gold cell. `|` is the only separator: commas occur *inside*
    values — `Power, Corruption & Lies`, `Metropolitan Club, Edge Road` — so splitting
    on them would shred three albums and a venue. Deduplicated on the normalised form,
    because returning the same value twice is one assertion, not two (§4.1).
    """
    if raw is None:
        return []
    out, seen = [], set()
    for part in str(raw).split("|"):
        surface = _TAG_RE.sub("", part).strip()
        value = normalise(part, field)
        if value and value not in seen:
            seen.add(value)
            out.append({"value": value, "surface": surface,
                        "tagged": bool(_TAG_RE.search(part))})
    return out


# -------------------------------------------------------------------------- aliases

def load_aliases(path: Path = ALIAS_PATH) -> dict:
    """
    `(field, normalised variant) -> normalised canonical`, both directions collapsed to
    the canonical form so matching is symmetric (§2.5). Frozen before the systems run:
    `score.py` compares the recorded SHA-256 and fails if the table has moved, since
    growing it in response to a system's output would be tuning on the evaluation set.
    """
    table: dict = {}
    if not Path(path).exists():
        return table
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            field = row["field"].strip()
            canonical = normalise(row["canonical"], field)
            variant = normalise(row["variant"], field)
            if not canonical or not variant:
                continue
            table[(field, variant)] = canonical
            table.setdefault((field, canonical), canonical)
    return table


ALIASES = load_aliases()


def canonicalise(field: str, value: str) -> str:
    """The canonical form of an already-normalised value, or the value itself."""
    return ALIASES.get((field, value), value)


# ----------------------------------------------------------------------- the sheet

def _read_sheet(path: Path = SHEET_PATH) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class Gold:
    """The sheet, and everything derived from it."""

    def __init__(self, sheet_path: Path = SHEET_PATH):
        self.sheet_path = Path(sheet_path)
        rows = _read_sheet(self.sheet_path)
        self.rows = rows
        self._objects = []
        for i, row in enumerate(rows):
            self._objects.append({
                "object_id": f"OBJ-{i + 1:03d}",
                "row": i + 1,
                "case_number": int(row["case"]),
                "item_description": row["item_description"],
                "caption": row.get("caption", "") or "",
                "unscored": {f: row.get(f, "") or "" for f in UNSCORED},
            })

        self.by_id = {o["object_id"]: o for o in self._objects}
        self.cases = collections.defaultdict(list)
        for o in self._objects:
            self.cases[o["case_number"]].append(o["object_id"])
        for o in self._objects:
            o["siblings"] = [s for s in self.cases[o["case_number"]]
                             if s != o["object_id"]]

        # gold values, per (object, field)
        self._gold = {}
        for row, o in zip(rows, self._objects):
            for f in SCORED:
                self._gold[(o["object_id"], f)] = split_cell(row.get(f, ""), f)

        # evidence, per scope
        self._case_captions = {
            case: " ".join(sorted({self.by_id[m]["caption"]
                                   for m in members if self.by_id[m]["caption"]}))
            for case, members in self.cases.items()}
        self._evidence = {}
        for o in self._objects:
            sib_text = " ".join(
                f'{self.by_id[s]["item_description"]} {self.by_id[s]["caption"]}'.strip()
                for s in o["siblings"])
            self._evidence[o["object_id"]] = {
                "E1": o["item_description"],
                "E2": f'{o["item_description"]} {o["caption"]}'.strip(),
                "E3": f'{o["item_description"]} {o["caption"]} {sib_text}'.strip(),
                # The scope at which support classes and distractors are computed.
                "case": f'{o["item_description"]} '
                        f'{self._case_captions.get(o["case_number"], "")}'.strip(),
            }
        self._norm_evidence = {
            oid: {scope: normalise(text) for scope, text in scopes.items()}
            for oid, scopes in self._evidence.items()}

        # support classes
        self._support = {}
        self._support_boundary = {}
        for (oid, f), values in self._gold.items():
            text = self._norm_evidence[oid]["case"]
            for v in values:
                key = (oid, f, v["value"])
                self._support[key] = ("supported" if contains(text, v["value"])
                                     else "inference_only")
                self._support_boundary[key] = (
                    "supported" if contains(text, v["value"], boundary=True)
                    else "inference_only")

    # ------------------------------------------------------------ §1.4 interface

    def objects(self) -> list:
        """`{object_id, case_number, item_description, caption, siblings}`, sheet order."""
        return list(self._objects)

    def gold_values(self, object_id: str, field: str) -> set:
        return {v["value"] for v in self._gold[(object_id, field)]}

    def gold_surface(self, object_id: str, field: str) -> list:
        """The values as the annotators wrote them, for reports and adjudication sheets."""
        return [v["surface"] for v in self._gold[(object_id, field)]]

    def raw_cell(self, object_id: str, field: str) -> str:
        """The cell exactly as written in the sheet, for data-quality reporting."""
        return self.rows[self.by_id[object_id]["row"] - 1].get(field, "") or ""

    def support_class(self, object_id: str, field: str, value: str) -> str:
        return self._support[(object_id, field, value)]

    def gold_vocabulary(self) -> set:
        """`(field, value)` occurring anywhere in the gold — the ceilings' candidates."""
        return {(f, v["value"]) for (oid, f), values in self._gold.items()
                for v in values}

    def sibling_distractors(self, object_id: str) -> set:
        """
        `(field, value)` that is gold for a sibling, is not gold for this object, and is
        visible in this object's evidence at the configured scope (§2.4).
        """
        obj = self.by_id[object_id]
        mine = {(f, v) for f in SCORED for v in self.gold_values(object_id, f)}
        text = self._norm_evidence[object_id]["case"]
        out = set()
        for sib in obj["siblings"]:
            for f in SCORED:
                for v in self.gold_values(sib, f):
                    if (f, v) not in mine and contains(text, v):
                        out.add((f, v))
        return out

    def evidence(self, object_id: str, scope: str) -> str:
        """The exact text a system in scope E1/E2/E3 sees."""
        if scope not in SCOPES:
            raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPES}")
        return self._evidence[object_id][scope]

    def aliases(self) -> dict:
        return dict(ALIASES)

    # ------------------------------------------------------------------ summaries

    def value_counts(self) -> dict:
        per_field = {f: sum(len(self._gold[(o["object_id"], f)])
                            for o in self._objects) for f in SCORED}
        supported = collections.Counter(
            f for (oid, f, v), c in self._support.items() if c == "supported")
        return {"per_field": per_field,
                "total": sum(per_field.values()),
                "supported_per_field": dict(supported),
                "supported": sum(supported.values())}

    def cell_counts(self) -> dict:
        empty = sum(1 for key in self._gold if not self._gold[key])
        total = len(SCORED) * len(self._objects)
        return {"total": total, "empty": empty, "non_empty": total - empty}

    def distractors(self) -> dict:
        triples, by_field, objects_hit = set(), collections.Counter(), set()
        for o in self._objects:
            for (f, v) in self.sibling_distractors(o["object_id"]):
                triples.add((o["object_id"], f, v))
                by_field[f] += 1
                objects_hit.add(o["object_id"])
        return {"triples": len(triples), "objects": len(objects_hit),
                "by_field": dict(by_field.most_common())}

    def ceilings(self) -> dict:
        """
        The **exhaustive extractor** of §2.4: for each object, assert every gold-vocabulary
        pair whose value occurs in that object's evidence at the scope. It never
        hallucinates, never mis-normalises and never decides, and it needs no model and no
        API call.

        Be precise about what it bounds, because the word "ceiling" invites a stronger
        reading than the construction supports:

        **Recall is a ceiling, for extractive systems only.** Any system that asserts only
        values present in its evidence can score a true positive only on a gold value of
        that object; such a value is in the gold vocabulary and occurs in the evidence, so
        this oracle asserts it too. Its true positives therefore contain any extractive
        system's. A system that *infers* is not bound by it — measured, `M1` exceeds the E1
        recall ceiling by 0.080 while every quote-policy cell, which is extractive by
        construction, stays below its scope's.

        **Precision is not a ceiling in any direction.** This oracle maximises assertions,
        and precision is exactly what a well-chosen subset of them improves: a system
        asserting one correct value scores 1.000. Its precision is the *reference operating
        point of copying everything visible* — what you get when you decline to decide.
        Exceeding it is the finding the section is built on (§4.4.6), not an anomaly, and
        `report.py` calls it `exhaustive_precision` rather than a ceiling for that reason.
        """
        vocab = sorted(self.gold_vocabulary())
        counts = self.value_counts()
        out = {}
        for scope in SCOPES:
            tp = fp = tp_supported = 0
            for o in self._objects:
                oid = o["object_id"]
                text = self._norm_evidence[oid][scope]
                for (f, v) in vocab:
                    if not contains(text, v):
                        continue
                    if v in self.gold_values(oid, f):
                        tp += 1
                        if self._support[(oid, f, v)] == "supported":
                            tp_supported += 1
                    else:
                        fp += 1
            asserted = tp + fp
            precision = tp / asserted if asserted else None
            recall = tp / counts["total"]
            recall_sup = tp_supported / counts["supported"]
            f1 = (2 * precision * recall / (precision + recall)
                  if precision and (precision + recall) else None)
            out[scope] = {"asserted": asserted, "tp": tp, "precision": precision,
                          "recall": recall, "recall_supported": recall_sup, "f1": f1}
        return out

    def baseline(self) -> dict:
        """The counts worth freezing, for the checklist of spec §9."""
        counts = self.value_counts()
        cells = self.cell_counts()
        dist = self.distractors()
        ceil = self.ceilings()
        return {
            "sheet": self.sheet_path.name,
            "sheet_sha256": sha256(self.sheet_path),
            "aliases_sha256": sha256(ALIAS_PATH) if ALIAS_PATH.exists() else None,
            "objects": len(self._objects),
            "cases": len(self.cases),
            "gold_values": counts["total"],
            "gold_per_field": counts["per_field"],
            "supported": counts["supported"],
            "inference_only": counts["total"] - counts["supported"],
            "supported_per_field": counts["supported_per_field"],
            "cells": cells,
            "distractors": dist,
            "ceilings": {s: {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in c.items()} for s, c in ceil.items()},
        }


# ---------------------------------------------------------------------- the checks

def verify(gold: Gold, *, freeze: bool = False) -> int:
    """The checks of §2.6. Blocking failures return non-zero; the rest are warnings."""
    failures, warnings = [], []
    cfg = CONFIG["objects"]

    # 1 — malformed tags. The sheet has carried `,/infer>` and `</paper>`, either of
    # which leaves a stray token in the normalised value so the gold can never match.
    # The guard therefore rejects any tag that is not a well-formed <infer>…</infer>.
    for i, row in enumerate(gold.rows):
        for f in SCORED:
            cell = row.get(f, "") or ""
            residue = _WELL_FORMED_RE.sub("", cell)
            if _ANY_TAG_RE.search(residue) or "<" in residue or ">" in residue:
                failures.append(f"malformed tag in OBJ-{i + 1:03d} {f}: {cell!r}")

    # 2 — row count and object_id uniqueness.
    if len(gold.rows) != cfg["n_expected"]:
        failures.append(f"row count {len(gold.rows)}, expected {cfg['n_expected']}")
    ids = [o["object_id"] for o in gold.objects()]
    if len(set(ids)) != len(ids):
        failures.append("object_id collision")

    # 3 — the two ordering assumptions, asserted FALSE as a regression guard against
    # the defects of §6.12 and §6.13: sorting the gold while the predictions stay in
    # file order displaced all 62 rows of the superseded evaluator.
    case_seq = [o["case_number"] for o in gold.objects()]
    distinct = sorted(set(case_seq))
    is_sorted = case_seq == sorted(case_seq)
    is_contiguous = distinct == list(range(min(distinct), max(distinct) + 1))
    if is_sorted != cfg["assume_sorted_by_case"]:
        failures.append(f"sheet sorted-by-case is {is_sorted}, config expects "
                        f"{cfg['assume_sorted_by_case']}")
    if is_contiguous != cfg["assume_contiguous_cases"]:
        failures.append(f"contiguous cases is {is_contiguous}, config expects "
                        f"{cfg['assume_contiguous_cases']}")

    # 4 — the held-out examples must not be in the scored sheet (§0.8).
    held = {normalise(d) for d in cfg["held_out_examples"]}
    for o in gold.objects():
        if normalise(o["item_description"]) in held:
            failures.append(f"held-out example present as {o['object_id']}")

    # 5 — <infer> tags against the derived support class (§2.3). Advisory: the tags are
    # incomplete, the derived label is reproducible, so the label wins and the
    # disagreements are printed for the annotators to reconcile.
    disagreements = []
    for (oid, f), values in gold._gold.items():
        for v in values:
            derived = gold.support_class(oid, f, v["value"])
            if v["tagged"] != (derived == "inference_only"):
                disagreements.append((oid, f, v["surface"], "tagged" if v["tagged"]
                                      else "untagged", derived))

    # 6 — intra-cell duplicates, on the normalised form. `split_cell` already collapses
    # them, so the count of written parts against the count of distinct values is what
    # exposes one. Currently none, but the systems will produce them (§4.1 rule 1).
    for (oid, f), values in gold._gold.items():
        raw = gold.raw_cell(oid, f)
        written = [p for p in raw.split("|") if normalise(p, f)]
        if len(written) != len(values):
            warnings.append(f"duplicate value within {oid} {f}: {raw!r}")

    # 7 — years that are not four digits (§4.1 rule 3 discards these when scoring).
    for (oid, f), values in gold._gold.items():
        if f != "years":
            continue
        for v in values:
            if not _YEAR_RE.match(v["value"]):
                warnings.append(f"non-four-digit year in {oid}: {v['surface']!r}")

    # 8 — objects with no caption: E1 and E2 render identically for them, so the
    # E1-vs-E2 comparison must also be reported over the captioned objects alone (§3.4).
    uncaptioned = [o["object_id"] for o in gold.objects() if not o["caption"]]

    # 9 — containment rule sensitivity (see the module docstring).
    boundary_diff = [k for k in gold._support
                     if gold._support[k] != gold._support_boundary[k]]

    # ------------------------------------------------------------------- reporting
    counts = gold.value_counts()
    cells = gold.cell_counts()
    dist = gold.distractors()
    ceil = gold.ceilings()
    indicative = CONFIG["fields"]["indicative_below"]

    print(f"sheet        {gold.sheet_path.name}  sha256 {sha256(gold.sheet_path)[:16]}…")
    print(f"aliases      {ALIAS_PATH.name}  sha256 "
          f"{sha256(ALIAS_PATH)[:16] if ALIAS_PATH.exists() else '(missing)'}…  "
          f"{len(ALIASES)} entries")
    print(f"objects      {len(gold.objects())} in {len(gold.cases)} cases  "
          f"sizes {dict(collections.Counter(sorted(len(v) for v in gold.cases.values())))}")
    print(f"case order   {case_seq[:6]}…  sorted={is_sorted}  contiguous={is_contiguous}")
    print(f"captions     {len(gold.objects()) - len(uncaptioned)} captioned, "
          f"{len(uncaptioned)} without; "
          f"{len({o['caption'] for o in gold.objects() if o['caption']})} distinct")

    print(f"\ngold values  {counts['total']}  "
          f"({counts['supported']} supported / "
          f"{counts['total'] - counts['supported']} inference-only)")
    print(f"cells        {cells['total']}  "
          f"{cells['empty']} empty ({cells['empty'] / cells['total']:.1%}) / "
          f"{cells['non_empty']} non-empty")
    print(f"\n{'field':16s} {'gold':>5s} {'supp':>5s} {'infer':>6s} {'distract':>9s}  flag")
    for f in sorted(SCORED, key=lambda f: -counts["per_field"][f]):
        gold_n = counts["per_field"][f]
        sup_n = counts["supported_per_field"].get(f, 0)
        flag = "indicative" if gold_n < indicative else ""
        print(f"{f:16s} {gold_n:5d} {sup_n:5d} {gold_n - sup_n:6d} "
              f"{dist['by_field'].get(f, 0):9d}  {flag}")

    print(f"\ndistractors  {dist['triples']} triples over {dist['objects']} objects "
          f"({dist['triples'] / counts['total']:.2f} per gold value)")
    print(f"\n{'scope':6s} {'asserted':>9s} {'TP':>4s} {'P':>7s} {'R':>7s} "
          f"{'R_supp':>7s} {'F1':>7s}")
    for scope in SCOPES:
        c = ceil[scope]
        print(f"{scope:6s} {c['asserted']:9d} {c['tp']:4d} {c['precision']:7.3f} "
              f"{c['recall']:7.3f} {c['recall_supported']:7.3f} {c['f1']:7.3f}")

    if disagreements:
        print(f"\ntag/support disagreements: {len(disagreements)} "
              f"(tags are advisory, the derived label wins — §2.3)")
        tagged_but_supported = [d for d in disagreements if d[3] == "tagged"]
        for oid, f, surface, tag, derived in tagged_but_supported:
            print(f"  {oid} {f:14s} {surface!r} tagged <infer> but occurs in the text")
        untagged = collections.Counter(d[1] for d in disagreements if d[3] == "untagged")
        print(f"  untagged yet inference-only, by field: {dict(untagged.most_common())}")

    if boundary_diff:
        print(f"\ncontainment: {len(boundary_diff)} values change class under "
              f"token-boundary matching — review before relying on supported recall")
    else:
        print("\ncontainment: naive substring and token-boundary matching agree "
              "on every value")

    if uncaptioned:
        print(f"\nno caption ({len(uncaptioned)}): E1 and E2 are identical for these, so "
              f"§3.4's comparison is also reported over the {len(gold.objects()) - len(uncaptioned)} "
              f"captioned objects")

    for w in warnings:
        print(f"WARNING  {w}")

    # ------------------------------------------------------------------- baseline
    current = gold.baseline()
    if freeze or not BASELINE_PATH.exists():
        BASELINE_PATH.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"\nbaseline {'re-' if freeze else ''}frozen -> {BASELINE_PATH.name}")
    else:
        stored = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        drift = {k: (stored.get(k), current.get(k)) for k in current
                 if stored.get(k) != current.get(k)}
        if drift:
            failures.append(
                "sheet or derived counts have changed since the baseline was frozen: "
                + ", ".join(sorted(drift)) + " — re-run with --freeze if intended")
        else:
            print(f"\nbaseline matches {BASELINE_PATH.name}")

    if failures:
        print(f"\n{len(failures)} BLOCKING failure(s):")
        for f in failures:
            print(f"  FAIL  {f}")
        return 1
    print("\nall blocking checks pass")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", action="store_true",
                    help="run the data-quality gate and exit non-zero on failure")
    ap.add_argument("--freeze", action="store_true",
                    help="re-freeze data/gold_baseline.json after a deliberate re-annotation")
    ap.add_argument("--sheet", type=Path, default=SHEET_PATH)
    args = ap.parse_args()

    gold = Gold(args.sheet)
    status = verify(gold, freeze=args.freeze)
    return status if args.verify else 0


if __name__ == "__main__":
    sys.exit(main())
