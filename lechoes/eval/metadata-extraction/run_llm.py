#!/usr/bin/env python3
"""Stage 2 — the six LLM cells and the zero-shot ablation (spec §3.1, §3.5, §3.7).

This is the only stage that spends API budget.

    python run_llm.py --estimate                 # how many calls, no spending
    python run_llm.py --systems M3 --limit 3 --runs 1   # smoke test, 3 calls
    python run_llm.py                            # the full sweep: 6 cells x 62 x 3
    python run_llm.py --ablation                 # M3-zeroshot, prose only (§3.8)

| | free assertion | evidence-quote required |
| --- | --- | --- |
| **E1** description only | `M1` | `M2` |
| **E2** + own caption | `M3` (reference) | `M4` |
| **E3** + siblings, with attribution instruction | `M5` | `M6` |

One call per object in all three scopes. `E3` supplies the siblings as *disambiguating
context* rather than as further extraction targets, which is what removes the
response-count failure that killed three cells of the superseded grid (*"Only 2 responses
received, expected 3"*) and what makes attribution measurable: the model is told what the
distractors are.

**The evidence check** (§3.7) applies under the `quote` policy only, after generation and
before anything reaches `predicted`. A value survives if its quote occurs in the input and
the value occurs in the quote. Both tests run **per evidence block** — description, caption,
each sibling entry — never against their concatenation: a quote spanning the join between
two blocks is not "the exact words from the text", it is words from two texts joined, and
the model could not have copied it from anything it was shown. That is stricter than the
letter of §3.7, so the runner counts the drops it alone causes (`dropped_span_join`) and the
manifest records them, making the deviation's effect a measured quantity rather than an
assumption.

State plainly what the check cannot do: it removes values that are not in the text. It
cannot remove a *sibling's* artist, because that artist is in the text. Hallucination and
misattribution are different failures, and only the attribution instruction addresses the
second.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gold as G
import models as M
import predictions as P
import prompts as PR

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
sys.path.insert(0, str(SRC))

from llm import (  # noqa: E402
    SCHEMA_METADATA,
    CachingLLMClient,
    MetadataFields,
    build_client,
    cache_identity,
)


def metadata_fields(config: dict, policy: str) -> MetadataFields:
    """
    The 14 requested keys. `item_description` is **not** among them: the superseded schema
    asked for it back and the model spent its output rewriting its own input (§6.8).
    `base_type` is the category task and is not requested either (§1.1).
    """
    return MetadataFields(list_fields=list(config["fields"]["scored"]),
                          string_fields=[f for f in config["fields"]["extracted_unscored"]
                                         if f != "cardinality"],
                          int_fields=["cardinality"],
                          quoted=(policy == PR.POLICY_QUOTE))


def evidence_segments(gold: G.Gold, object_id: str, scope: str, config: dict) -> list:
    """
    The blocks a system sees, kept separate so a quote can be checked against the text it
    would have been copied from rather than against a concatenation of several.
    """
    obj = gold.by_id[object_id]
    scopes = config["evidence_scopes"][scope]
    segments = [obj["item_description"]]
    if scopes.get("own_caption") and obj["caption"]:
        segments.append(obj["caption"])
    if scopes.get("siblings"):
        for sib in obj["siblings"]:
            segments.append(gold.by_id[sib]["item_description"])
            if gold.by_id[sib]["caption"]:
                segments.append(gold.by_id[sib]["caption"])
    return segments


def _coerce(value) -> str:
    """
    Years arrive as the JSON number 1973 often enough to matter: the schema declares
    strings, but a model that returns an integer must not be discarded as a type error
    (§4.1 rule 3).
    """
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(int(value))
    return str(value or "").strip()


def parse_response(parsed, config: dict, policy: str) -> tuple:
    """
    The provider's object into `{field: [{value, quote}]}` plus the unscored fields.

    Tolerant by design: a field the model omits becomes an empty list, not a failure. A
    missing key and an empty list mean the same thing *to the model*, and collapsing the
    whole object over one absent key would cost 11 cells to save nothing.
    """
    scored = list(config["fields"]["scored"])
    predicted = P.empty_prediction(scored)
    if not isinstance(parsed, dict):
        return predicted, {}, False

    for field in scored:
        raw = parsed.get(field) or []
        if not isinstance(raw, list):
            raw = [raw]
        for item in raw:
            if isinstance(item, dict):
                value = _coerce(item.get("value"))
                quote = _coerce(item.get("quote")) or None
            else:
                value, quote = _coerce(item), None
            if value:
                predicted[field].append({"value": value, "quote": quote})

    unscored = {f: parsed.get(f) for f in config["fields"]["extracted_unscored"]}
    return predicted, unscored, True


def evidence_check(predicted: dict, segments: list, config: dict) -> tuple:
    """
    §3.7. Returns `(kept, dropped, join_only)` — the last being values dropped solely
    because their quote straddled the join between two evidence blocks.
    """
    norm_segments = [G.normalise(s) for s in segments]
    joined = G.normalise(" ".join(segments))
    kept = P.empty_prediction(list(config["fields"]["scored"]))
    dropped, join_only = [], 0

    for field, items in predicted.items():
        for item in items:
            value, quote = item["value"], item.get("quote")
            nvalue = G.normalise(value, field)
            if not quote:
                dropped.append({"field": field, "value": value, "quote": None,
                                "reason": "no_quote"})
                continue
            nquote = G.normalise(quote)
            in_segment = any(G.contains(seg, nquote) for seg in norm_segments)
            if not in_segment:
                # Distinguish "nowhere in the input" from "only across a block join":
                # the second is what this runner is stricter about than §3.7.
                if G.contains(joined, nquote):
                    join_only += 1
                    dropped.append({"field": field, "value": value, "quote": quote,
                                    "reason": "quote_spans_blocks"})
                else:
                    dropped.append({"field": field, "value": value, "quote": quote,
                                    "reason": "quote_not_in_text"})
                continue
            if not G.contains(nquote, nvalue):
                dropped.append({"field": field, "value": value, "quote": quote,
                                "reason": "value_not_in_quote"})
                continue
            kept[field].append(item)
    return kept, dropped, join_only


class _KeyStub:
    """Enough of an LLMClient for `CachingLLMClient._key`, so `--estimate` needs no key."""

    def __init__(self, provider: str, model: str):
        self.provider, self.model = provider, model


def run_cell(gold: G.Gold, config: dict, *, system: dict, model: dict, run: int,
             objects: list, api_key: str, prompt_mode: str,
             system_id: str | None = None) -> tuple:
    """One (system, model, run): every object, one call each."""
    system_id = system_id or system["id"]
    policy, scope = system["policy"], system["scope"]
    fields = metadata_fields(config, policy)
    scored = list(config["fields"]["scored"])

    cache_path = (P.paths(config)["llm_cache"] /
                  f"{system_id}_{model['slug']}_run{run}.json")
    client = CachingLLMClient(
        client=build_client(model["id"], api_key=api_key,
                            temperature=float(config["llm"]["temperature"]),
                            timeout_s=120),
        path=cache_path, run_index=run)

    records, failures, dropped_total, join_total = [], 0, 0, 0
    prompt_sample = None

    for obj in objects:
        oid = obj["object_id"]
        prompt = PR.build_prompt(gold, oid, scope=scope, policy=policy,
                                 config=config, prompt_mode=prompt_mode)
        prompt_sample = prompt_sample or prompt
        started = time.perf_counter()
        try:
            response = client.generate(prompt, SCHEMA_METADATA, fields=fields)
        except Exception as exc:                       # noqa: BLE001 - one object, not the run
            failures += 1
            records.append(P.record(
                object_id=oid, case_number=obj["case_number"], system=system_id,
                model=model["slug"], run=run, evidence_scope=scope,
                assertion_policy=policy, predicted=None,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}"))
            continue

        latency = int((time.perf_counter() - started) * 1000)
        predicted, unscored, ok = parse_response(response.parsed, config, policy)
        if not ok:
            failures += 1
            records.append(P.record(
                object_id=oid, case_number=obj["case_number"], system=system_id,
                model=model["slug"], run=run, evidence_scope=scope,
                assertion_policy=policy, predicted=None, latency_ms=latency,
                raw_response=response.text, error="parse: response was not an object"))
            continue

        dropped = []
        if policy == PR.POLICY_QUOTE:
            segments = evidence_segments(gold, oid, scope, config)
            predicted, dropped, join_only = evidence_check(predicted, segments, config)
            join_total += join_only
        dropped_total += len(dropped)

        for field in scored:
            predicted[field] = P.dedupe(predicted[field], G.normalise, field)

        records.append(P.record(
            object_id=oid, case_number=obj["case_number"], system=system_id,
            model=model["slug"], run=run, evidence_scope=scope,
            assertion_policy=policy, predicted=predicted, dropped_no_quote=dropped,
            unscored=unscored, latency_ms=latency, raw_response=response.text))

    client.save()
    return records, {"failures": failures, "dropped": dropped_total,
                     "join_only": join_total, "cache_hits": client.hits,
                     "cache_misses": client.misses, "prompt_sample": prompt_sample}


def estimate(gold: G.Gold, config: dict, cells: list, objects: list, runs: int,
             model: dict, prompt_mode_for) -> None:
    """Call count and how much of it the cache already covers. Spends nothing."""
    total = billable = 0
    print(f"{'cell':14s} {'calls':>6s} {'cached':>7s} {'to bill':>8s}")
    for system_id, system in cells:
        for run in range(1, runs + 1):
            cache_path = (P.paths(config)["llm_cache"] /
                          f"{system_id}_{model['slug']}_run{run}.json")
            stub = CachingLLMClient(
                client=_KeyStub(model["provider"],
                                cache_identity(model["id"]).split(":")[-1]),
                path=cache_path, run_index=run)
            fields = metadata_fields(config, system["policy"])
            cached = 0
            for obj in objects:
                prompt = PR.build_prompt(gold, obj["object_id"], scope=system["scope"],
                                         policy=system["policy"], config=config,
                                         prompt_mode=prompt_mode_for(system_id))
                if stub._key(prompt, SCHEMA_METADATA, None, fields) in stub._cache:
                    cached += 1
            n = len(objects)
            total += n
            billable += n - cached
            print(f"{system_id + f' run{run}':14s} {n:6d} {cached:7d} {n - cached:8d}")
    print(f"{'total':14s} {total:6d} {total - billable:7d} {billable:8d}")
    print(f"\n{billable} calls would be billed against {model['id']}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--systems", nargs="+", help="subset of M1..M6")
    ap.add_argument("--models", nargs="+", help="subset of the configured models")
    ap.add_argument("--runs", type=int, help="override the configured run count")
    ap.add_argument("--limit", type=int, help="only the first N objects (smoke test)")
    ap.add_argument("--api-key", default="",
                    help="otherwise the provider's environment variable is used")
    ap.add_argument("--estimate", action="store_true",
                    help="print the call count and exit without spending")
    ap.add_argument("--ablation", action="store_true",
                    help="run M3-zeroshot instead of the factorial (§3.8)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing prediction files")
    args = ap.parse_args()

    config = G.CONFIG
    gold = G.Gold()
    objects = gold.objects()[:args.limit] if args.limit else gold.objects()
    runs = args.runs or int(config["runs"])
    all_systems = {s["id"]: s for s in config["systems"] if s["kind"] == "llm"}

    if args.ablation:
        ab = config["ablation"]
        base = all_systems[ab["of"]]
        cells = [(ab["id"], base)]
        modes = {ab["id"]: ab["change"]["prompt_mode"]}
    else:
        wanted = args.systems or list(all_systems)
        unknown = [s for s in wanted if s not in all_systems]
        if unknown:
            raise SystemExit(f"unknown LLM system(s) {unknown}; "
                             f"configured: {sorted(all_systems)}")
        cells = [(s, all_systems[s]) for s in wanted]
        modes = {s: config["llm"]["prompt_mode"] for s in wanted}

    chosen = M.select_models(M.load_models(), args.models)

    if args.estimate:
        for model in chosen:
            print(f"\n== {model['id']} ==")
            estimate(gold, config, cells, objects, runs, model, modes.get)
        return 0

    manifest = P.Manifest(config)
    for model in chosen:
        if not (args.api_key or M.key_present(model)):
            raise SystemExit(f"no API key for {model['id']} — pass --api-key or set the "
                             f"provider's environment variable")
        for system_id, system in cells:
            for run in range(1, runs + 1):
                path = P.prediction_path(config, system_id, model["slug"], run)
                if path.exists() and not args.force:
                    print(f"{system_id} {model['slug']} run{run}: exists — skipping")
                    continue
                print(f"{system_id} {model['slug']} run{run}: scope {system['scope']}, "
                      f"policy {system['policy']}, {len(objects)} objects", flush=True)
                records, stats = run_cell(
                    gold, config, system=system, model=model, run=run, objects=objects,
                    api_key=args.api_key, prompt_mode=modes[system_id],
                    system_id=system_id)
                P.write(records, path)
                asserted = sum(len(r["predicted"][f]) for r in records
                               if r["predicted"] for f in config["fields"]["scored"])
                print(f"  {asserted} values asserted, {stats['dropped']} dropped by the "
                      f"evidence check ({stats['join_only']} for spanning blocks), "
                      f"{stats['failures']} failures, "
                      f"{stats['cache_hits']} cache hits / {stats['cache_misses']} calls")
                manifest.add(
                    system=system_id, model_slug=model["slug"], model_id=model["id"],
                    run=run, scope=system["scope"], policy=system["policy"],
                    prompt_mode=modes[system_id], objects=len(records),
                    failures=stats["failures"], dropped=stats["dropped"],
                    cache_hits=stats["cache_hits"], cache_misses=stats["cache_misses"],
                    prompt_sample=stats["prompt_sample"],
                    fewshot_block=(PR.fewshot_block(
                        config, system["policy"],
                        include_caption=bool(config["evidence_scopes"][system["scope"]]
                                             .get("own_caption")))
                        if modes[system_id] == "few-shot" else None),
                    extra={"dropped_span_join": stats["join_only"],
                           "values_asserted": asserted})
                manifest.save(gold_baseline=gold.baseline())
                print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
