#!/usr/bin/env python3
"""Stage 2 — the two non-generative baselines (spec §3.2, §3.3). No LLM, no API key.

    python run_baselines.py                    # both, at scope E2
    python run_baselines.py --systems M0a      # spaCy only
    python run_baselines.py --limit 5          # smoke test over 5 objects
    python run_baselines.py --force            # overwrite existing predictions

| Id | System |
| -- | ------ |
| M0a | spaCy `en_core_web_trf`, fixed entity types mapped onto fields |
| M0b | GLiNER, the 11 field names supplied as zero-shot entity types |

Both read scope **E2** (description + own caption) — the scope whose ceiling F₁ is highest —
so they are compared at their most favourable setting rather than handicapped. Both are
deterministic and run once; their standard deviation is reported as `—`, never as 0.000.

Neither has an assertion policy: they are span taggers, so every value they return is by
construction a substring of the input and the evidence check of §3.7 could never drop
anything. Their records carry `assertion_policy: "n/a"` and put the tagged span in `quote`,
which keeps the error attribution of §4.4.7 computable for them on the same footing as for
the LLM cells.

`M0a` cannot reach four of the eleven fields at all — `music_albums`, `music_genres`,
`colours` and `materials` have no corresponding spaCy label — and that is the point of it:
a fixed-schema tagger is a floor, not a candidate. Its recall is bounded above by the 159
gold values in the seven fields it can reach, i.e. 0.761, before any error of its own.
`report.py` prints that bound beside its row so the figure is not misread as a failure of
the tagger.

Model weights are downloaded deliberately, not during a scored run:

    python -m spacy download en_core_web_trf
    python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_medium-v2.1')"
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import gold as G
import predictions as P

HERE = Path(__file__).resolve().parent
YEAR_RE = re.compile(r"\b(\d{4})\b")


def country_surface_forms(gold: G.Gold) -> set:
    """
    The forms that count as a country for `M0a`'s `GPE` decision.

    §3.2 routes `GPE` to `countries` when the surface form is a known country and to
    `places` otherwise. "Known" means the alias table's country entries plus the country
    values in the gold vocabulary — both fixed before the run, so no tuning is possible.
    """
    forms = {value for (field, value) in gold.aliases() if field == "countries"}
    forms |= {v for (field, v) in gold.gold_vocabulary() if field == "countries"}
    return forms


# --------------------------------------------------------------------------- M0a

def run_spacy(gold: G.Gold, config: dict, objects: list, model_name: str) -> list:
    import spacy

    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise SystemExit(
            f"spaCy model {model_name!r} is not installed — run\n"
            f"  python -m spacy download {model_name}\n({exc})") from exc

    label_map = config["spacy_label_map"]
    scored = list(config["fields"]["scored"])
    countries = country_surface_forms(gold)
    records = []

    for obj in objects:
        oid = obj["object_id"]
        text = gold.evidence(oid, "E2")
        started = time.perf_counter()
        doc = nlp(text)
        predicted = P.empty_prediction(scored)
        raw = []

        for ent in doc.ents:
            raw.append({"text": ent.text, "label": ent.label_,
                        "start": ent.start_char, "end": ent.end_char})
            field = label_map.get(ent.label_)
            if field is None:
                continue
            value = ent.text.strip()
            if field == "years":
                # DATE spans are prose ("the late 1970s/early 1980s", "23rd October
                # 1964"); only a four-digit year is a value this schema can hold, and a
                # span may carry more than one.
                for year in YEAR_RE.findall(value):
                    predicted["years"].append({"value": year, "quote": ent.text})
                continue
            if ent.label_ == "GPE":
                field = ("countries" if G.normalise(value, "countries") in countries
                         else "places")
            predicted[field].append({"value": value, "quote": ent.text})

        for field in scored:
            predicted[field] = P.dedupe(predicted[field], G.normalise, field)

        records.append(P.record(
            object_id=oid, case_number=obj["case_number"], system="M0a", model=None,
            run=1, evidence_scope="E2", assertion_policy="n/a", predicted=predicted,
            unscored={},  # a span tagger produces no title, type or cardinality
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw_response=raw))
    return records


# --------------------------------------------------------------------------- M0b

def run_gliner(gold: G.Gold, config: dict, objects: list, model_name: str,
               threshold: float) -> list:
    try:
        from gliner import GLiNER
    except ImportError as exc:                                  # pragma: no cover
        raise SystemExit(f"gliner is not installed ({exc})") from exc

    model = GLiNER.from_pretrained(model_name)
    labels = list(config["gliner_labels"])
    scored = list(config["fields"]["scored"])
    # Explicit in config, never derived from spelling: pluralising the label string gives
    # `country` -> `countrys`, which matches no field and drops `countries` on the floor.
    label_to_field = dict(config["gliner_label_map"])
    unmapped = set(labels) - set(label_to_field)
    if unmapped:
        raise SystemExit(f"gliner labels absent from gliner_label_map: {sorted(unmapped)}")
    unknown = {f for f in label_to_field.values() if f not in scored}
    if unknown:
        raise SystemExit(f"gliner_label_map targets unscored fields: {sorted(unknown)}")

    records = []
    for obj in objects:
        oid = obj["object_id"]
        text = gold.evidence(oid, "E2")
        started = time.perf_counter()
        entities = model.predict_entities(text, labels, threshold=threshold)
        predicted = P.empty_prediction(scored)
        raw = [{"text": e["text"], "label": e["label"],
                "score": round(float(e.get("score", 0.0)), 4)} for e in entities]

        for ent in entities:
            field = label_to_field.get(ent["label"])
            if field is None:
                continue
            value = ent["text"].strip()
            if field == "years":
                for year in YEAR_RE.findall(value):
                    predicted["years"].append({"value": year, "quote": ent["text"]})
                continue
            predicted[field].append({"value": value, "quote": ent["text"]})

        for field in scored:
            predicted[field] = P.dedupe(predicted[field], G.normalise, field)

        records.append(P.record(
            object_id=oid, case_number=obj["case_number"], system="M0b", model=None,
            run=1, evidence_scope="E2", assertion_policy="n/a", predicted=predicted,
            unscored={},
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw_response=raw))
    return records


# -------------------------------------------------------------------------- driver

def summarise(records: list, gold: G.Gold, config: dict) -> None:
    scored = list(config["fields"]["scored"])
    asserted = sum(len(r["predicted"][f]) for r in records for f in scored)
    cells = sum(1 for r in records for f in scored if r["predicted"][f])
    reachable = [f for f in scored if any(r["predicted"][f] for r in records)]
    print(f"  {len(records)} objects, {asserted} values asserted "
          f"({asserted / max(1, len(records)):.1f} per object), "
          f"{cells} non-empty cells")
    print(f"  fields reached: {len(reachable)}/{len(scored)} — "
          f"{', '.join(reachable)}")
    unreachable = [f for f in scored if f not in reachable]
    if unreachable:
        print(f"  fields never asserted: {', '.join(unreachable)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--systems", nargs="+", default=["M0a", "M0b"],
                    choices=["M0a", "M0b"])
    ap.add_argument("--limit", type=int, help="only the first N objects (smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing prediction files")
    args = ap.parse_args()

    config = G.CONFIG
    gold = G.Gold()
    objects = gold.objects()[:args.limit] if args.limit else gold.objects()
    systems = {s["id"]: s for s in config["systems"]}
    manifest = P.Manifest(config)

    for system_id in args.systems:
        spec = systems[system_id]
        path = P.prediction_path(config, system_id, None, 1)
        if path.exists() and not args.force:
            print(f"{system_id}: {path.name} exists — skipping (use --force)")
            continue

        print(f"{system_id}: {spec['kind']} {spec['model']} at scope {spec['scope']}")
        if spec["kind"] == "spacy":
            records = run_spacy(gold, config, objects, spec["model"])
        else:
            records = run_gliner(gold, config, objects, spec["model"],
                                 float(spec.get("threshold", 0.5)))

        P.write(records, path)
        summarise(records, gold, config)
        failures = sum(1 for r in records if r["error"])
        manifest.add(system=system_id, model_slug=None, model_id=spec["model"], run=1,
                     scope=spec["scope"], policy="n/a", prompt_mode=None,
                     objects=len(records), failures=failures, dropped=0,
                     cache_hits=0, cache_misses=0,
                     extra={"deterministic": True,
                            "threshold": spec.get("threshold")})
        print(f"  -> {path.name}")

    manifest.save(gold_baseline=gold.baseline())
    return 0


if __name__ == "__main__":
    sys.exit(main())
