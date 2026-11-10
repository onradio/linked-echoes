#!/usr/bin/env python3
"""The gold standard, read once and interpreted the same way by every stage.

    python gold.py            # summary of the completed annotation
    python gold.py --verify   # resolve every URI against the indexed vocabulary

`run_category.py`, `run_retrieval.py`, `run_selection.py` and `score.py` all import
`load_gold()` rather than parsing the sheet themselves, so the tier rules of §2.0
and the reachability rule of §2.1 are stated in exactly one place.

Tiers (spec §2.0):

    strong  type_gold_uri set                  the gold set is gold ∪ alt
    weak    only type_alt_uris set             no exact term exists; the recorded
                                               term is a defensible approximation,
                                               and abstention is equally correct
    nil     neither, type_nil_reason set       no usable term at all
"""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SHEET = DATA / "classification_annotation_sheet_done.csv"
VOCAB_CSV = HERE.parent.parent / "src" / "vocabularies" / "all_concepts.csv"

# Category -> vocabulary, mirroring VOCAB_MAP in src/classification.py. Repeated
# rather than imported so that the eval can be read on its own terms; the check
# in `verify` fails loudly if the two ever diverge.
VOCAB_MAP = {
    "Musical Instrument": "MIMO",
    "Clothing": "Europeana",
    "Audiovisual Media": "PBCore",
    "Textual & Graphic Work": "AAT",
    "Merchandise": "AAT",
    "Personal Effects": "AAT",
    "Award & Recognition": "AAT",
}

# URI namespace -> vocabulary. The sheet records full URIs; CURIEs are accepted
# too, since §2.1 was written in terms of prefixes.
NAMESPACES = {
    "http://vocab.getty.edu/aat/": "AAT",
    "aat:": "AAT",
    "http://www.mimo-db.eu/InstrumentsKeywords/": "MIMO",
    "mimo:": "MIMO",
    "http://thesaurus.europeanafashion.eu/thesaurus/": "Europeana",
    "euf:": "Europeana",
    "http://pbcore.org/": "PBCore",
    "pbcore:": "PBCore",
    "http://www.wikidata.org/entity/": "Wikidata",
    "wd:": "Wikidata",
    "http://radio.liv.ac.uk/bme/": "BME",
    "bme:": "BME",
}

# Vocabularies a search can actually reach: the routed one plus AAT, which the
# fallback always searches (spec §0.4).
ALWAYS_SEARCHED = "AAT"


def vocabulary_of(uri: str) -> str | None:
    """The vocabulary a term URI belongs to, or None if unrecognised."""
    for prefix, name in NAMESPACES.items():
        if uri.startswith(prefix):
            return name
    return None


def split_uris(cell: str) -> list:
    return [u.strip() for u in (cell or "").split("|") if u.strip()]


def load_gold(sheet: Path = SHEET) -> list:
    """
    Reads the completed annotation and returns one dict per object.

    Keys added on top of the sheet's own columns:

        tier            'strong' | 'weak' | 'nil'
        gold_uris       the set a prediction is scored against (empty on a nil row)
        routed_vocab    vocabulary the gold primary category routes to
        reachable       whether any gold term is searchable, per §2.1
        abstention_ok   True where abstaining is a correct outcome: weak and nil
                        rows, and any row whose gold is unreachable
        object_text     name + description, blanks skipped (§1); captions excluded
    """
    if not sheet.exists():
        raise FileNotFoundError(
            f"{sheet} not found. The annotator's completed sheet is the only input "
            "later stages read; copy the annotated CSV there."
        )
    with open(sheet, newline="", encoding="utf-8") as fh:
        rows = [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]

    objects = []
    for r in rows:
        gold_uris = split_uris(r["type_gold_uri"])
        alt_uris = split_uris(r["type_alt_uris"])

        if gold_uris:
            tier, scored_against = "strong", gold_uris + alt_uris
        elif alt_uris:
            tier, scored_against = "weak", alt_uris
        else:
            tier, scored_against = "nil", []

        routed = VOCAB_MAP.get(r["category_gold_primary"])
        searchable = {routed, ALWAYS_SEARCHED}
        reachable = any(vocabulary_of(u) in searchable for u in scored_against)

        objects.append({
            **r,
            "tier": tier,
            "gold_uris": scored_against,
            "routed_vocab": routed,
            "reachable": reachable,
            # Abstention is correct wherever no exact reachable term exists: NIL
            # rows, unreachable golds, and — per §2.0 — every weak row.
            "abstention_ok": tier in ("weak", "nil") or not reachable,
            "object_text": ". ".join(x for x in (r["name"], r["description"]) if x),
        })
    return objects


def score_outcome(obj: dict, predicted_uri: str | None, abstained: bool) -> str:
    """
    The §4.2 outcome for one prediction, in one place so that the runner and the
    scorer cannot drift apart.

        TP       reachable gold, predicted term is accepted
        WT       reachable gold, a term was predicted, not accepted
        FA       reachable gold, system abstained (strong rows only)
        TA       no reachable gold, system abstained
        TA_weak  weak row, system abstained — no exact term exists, so declining
                 to assert one is correct, and is credited but reported apart
                 from TA since it is not the same claim
        SA       no reachable gold, system asserted a term

    On a weak row the recorded approximation is the gold set: predicting it is a
    TP, because the annotator endorsed that term, and abstaining is TA_weak.
    Only a third term is wrong.
    """
    has_gold = bool(obj["gold_uris"]) and obj["reachable"]
    if abstained or predicted_uri is None:
        if not has_gold:
            return "TA"
        return "TA_weak" if obj["tier"] == "weak" else "FA"
    if not has_gold:
        return "SA"
    return "TP" if predicted_uri in obj["gold_uris"] else "WT"


def summarise(objects: list) -> None:
    from collections import Counter
    print(f"{len(objects)} objects from {SHEET.name}\n")

    tiers = Counter(o["tier"] for o in objects)
    print("tier      n   share")
    for t in ("strong", "weak", "nil"):
        print(f"  {t:7s} {tiers[t]:3d}   {100 * tiers[t] / len(objects):5.1f}%")

    print("\nprimary category            n   routed")
    cats = Counter(o["category_gold_primary"] for o in objects)
    for c, n in cats.most_common():
        print(f"  {c:25s} {n:3d}   {VOCAB_MAP.get(c)}")
    unattested = [c for c in VOCAB_MAP if c not in cats]
    print(f"  unattested as primary: {', '.join(unattested)}")

    n_sec = sum(1 for o in objects if o["category_gold_secondary"])
    print(f"\nsecondary category: {n_sec} objects "
          f"({100 * n_sec / len(objects):.1f}%)")
    print(f"unreachable gold:   {sum(1 for o in objects if not o['reachable'] and o['tier'] != 'nil')}")
    print(f"abstention is correct on: "
          f"{sum(1 for o in objects if o['abstention_ok'])} objects "
          f"(weak ∪ nil ∪ unreachable)")

    fallback = [o for o in objects
                if o["routed_vocab"] != "AAT"
                and any(vocabulary_of(u) == "AAT" for u in o["gold_uris"])]
    print(f"\nfallback-dependent golds ({len(fallback)}): "
          f"{', '.join(o['object_id'] for o in fallback)}")
    print("  retrieval must search the routed vocabulary ∪ AAT for these")

    cases = Counter(o["case_number"] for o in objects)
    print(f"\n{len(cases)} display cases; objects per case: "
          f"{sorted(cases.values(), reverse=True)}")


def verify(objects: list) -> int:
    """Resolves every gold URI against the indexed vocabulary, as §9 requires."""
    import pandas as pd
    vocab = pd.read_csv(VOCAB_CSV, dtype=str).fillna("")
    by_uri = {u: (t, s) for u, t, s in
              zip(vocab.uri, vocab.term, vocab.source_vocabulary)}

    problems = []
    for o in objects:
        for col, label_col in (("type_gold_uri", "type_gold_label"),
                               ("type_alt_uris", "type_alt_label")):
            labels = [x.strip() for x in (o[label_col] or "").split("|") if x.strip()]
            uris = split_uris(o[col])
            if labels and len(labels) != len(uris):
                problems.append(f"{o['object_id']}: {len(uris)} URIs but "
                                f"{len(labels)} labels in {col}")
            for i, u in enumerate(uris):
                if u not in by_uri:
                    problems.append(f"{o['object_id']}: {u} is not in the indexed vocabulary")
                    continue
                term, _ = by_uri[u]
                if i < len(labels) and labels[i].casefold() != term.casefold():
                    problems.append(f"{o['object_id']}: label {labels[i]!r} != "
                                    f"vocabulary term {term!r} for {u}")
        if o["tier"] == "nil" and not o["type_nil_reason"]:
            problems.append(f"{o['object_id']}: no gold and no type_nil_reason")
        if o["category_gold_primary"] not in VOCAB_MAP:
            problems.append(f"{o['object_id']}: unknown category "
                            f"{o['category_gold_primary']!r}")
        if o["vocab_hint"] and o["vocab_hint"] != o["routed_vocab"]:
            problems.append(f"{o['object_id']}: vocab_hint {o['vocab_hint']!r} is stale; "
                            f"the primary category routes to {o['routed_vocab']!r}")

    n_uris = sum(len(split_uris(o["type_gold_uri"])) + len(split_uris(o["type_alt_uris"]))
                 for o in objects)
    if problems:
        print(f"{len(problems)} problem(s) across {n_uris} URI mentions:\n")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"{n_uris} URI mentions: all resolve in the indexed vocabulary, "
          "all labels agree, routing consistent.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", action="store_true",
                    help="Check every URI against the indexed vocabulary")
    ap.add_argument("--sheet", default=str(SHEET))
    args = ap.parse_args()

    gold = load_gold(Path(args.sheet))
    if args.verify:
        sys.exit(verify(gold))
    summarise(gold)
