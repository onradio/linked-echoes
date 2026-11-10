#!/usr/bin/env python3
"""Stage 5 — run the four systems over the frozen pools (spec §7).

    # offline systems, no API key needed
    python run_systems.py --systems S0 S1

    # LLM systems (key via --api-key, or the provider's environment variable)
    python run_systems.py --systems S2 S3
    python run_systems.py --systems S3 --model mistral:ministral-14b-2512

    python run_systems.py --systems S3 --k 10 --runs 1 --limit 5   # smoke test

Everything replays `pools_raw.jsonl` and `pools_valid_<eck_version>.jsonl`: no
retrieval and no SPARQL happen here, so the only live dependency is the LLM and
the only thing that varies between `S2` and `S3` is whether the ECK filter is
applied.

Predictions land in `data/predictions/`. The LLM systems carry the model in the
filename — `S3_<model-slug>_k<K>_run<R>.jsonl` — because a result produced by one
model says nothing about another; `S0` and `S1` call no model and so carry no
tag. `--model` takes `provider:model` (a bare name means Google).

| Id | Pipeline |
| -- | -------- |
| S0 | candidate at rank 1; never abstains; k-independent |
| S1 | highest-ranked ECK-valid candidate; abstain if none |
| S2 | LLM selects from the unfiltered top-k; may abstain |
| S3 | ECK filter, then LLM selects; may abstain |

LLM responses are cached on (system, mention, candidate ids, run), so the ~36% of
mentions whose pool is shallower than k cost one call across all four depths
rather than four, and an interrupted run resumes for free.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
sys.path.insert(0, str(SRC))

from linker import TypedRAGEntityLinker  # noqa: E402
from llm import API_KEY_ENV, cache_identity, parse_model_spec  # noqa: E402


def model_slug(spec: str) -> str:
    """Filesystem-safe model id, matching the classification evaluation's form."""
    provider, model = parse_model_spec(spec)
    return re.sub(r"[^a-z0-9]+", "-", f"{provider}:{model}".lower()).strip("-")

DATA = HERE / "data"
CACHE = DATA / ".cache"
PRED = DATA / "predictions"


def load_inputs(pools_path, valid_path, sheet_path):
    pools = [json.loads(l) for l in open(pools_path, encoding="utf-8")]
    valid = {}
    for line in open(valid_path, encoding="utf-8"):
        r = json.loads(line)
        v = r["eck_valid"]
        valid[(r["mention_id"], r["wikidata_id"])] = (
            None if v == "error_api" else bool(v))
    sheet = {r["mention_id"]: r for r in csv.DictReader(
        open(sheet_path, newline="", encoding="utf-8"))}
    return pools, valid, sheet


def gold_set(row):
    raw = (row["wikidata_qid"] + ";" + row["alt_qids"]).replace(",", ";")
    return {q.strip() for q in raw.split(";") if q.strip()}


def base_record(pool, sheet_row, system, k, run, eck_version, kept_ids,
                model=None):
    gold = gold_set(sheet_row) if sheet_row else set()
    retrieved = {c["wikidata_id"] for c in pool["candidates"][:k]}
    return {
        "mention_id": pool["mention_id"],
        "label": pool["label"],
        "entity_type": pool["entity_type"],
        "system": system, "k": k, "run": run, "eck_version": eck_version,
        # None for S0/S1, which reach no model at all.
        "model": model,
        "model_slug": model_slug(model) if model else None,
        "gold_in_retrieved": bool(gold & retrieved),
        "gold_in_validated": bool(gold & set(kept_ids)),
    }


def run_offline(system, pools, valid, sheet, k, eck_version):
    """S0 and S1 need no model — they read straight off the frozen artefacts."""
    out = []
    for p in pools:
        cands = p["candidates"][:k]
        kept = [c for c in cands
                if valid.get((p["mention_id"], c["wikidata_id"])) is not False]
        if system == "S0":
            pool_for_pick, unevaluable = cands, 0
        else:
            pool_for_pick = kept
            unevaluable = sum(
                1 for c in cands
                if valid.get((p["mention_id"], c["wikidata_id"])) is None)
        rec = base_record(p, sheet.get(p["mention_id"]), system, k, 1,
                          eck_version, [c["wikidata_id"] for c in kept])
        chosen = pool_for_pick[0] if pool_for_pick else None
        if not cands:
            outcome = "abstain_no_candidates"
        elif chosen is None:
            outcome = "abstain_all_filtered"
        else:
            outcome = "linked"
        rec.update({
            "n_retrieved": len(cands),
            "n_validated": len(cands) if system == "S0" else len(kept),
            "n_unevaluable": unevaluable,
            "predicted_qid": chosen["wikidata_id"] if chosen else None,
            "predicted_label": chosen["label"] if chosen else None,
            "outcome": outcome, "llm_raw_response": None, "latency_ms": 0,
        })
        out.append(rec)
    return out


def run_llm(system, linker, pools, valid, sheet, k, run, eck_version,
            llm_cache, cache_path, verbose=False):
    validate = system == "S3"
    out = []
    for i, p in enumerate(pools, start=1):
        mid = p["mention_id"]
        cands = p["candidates"][:k]
        validity = ({c["wikidata_id"]: valid.get((mid, c["wikidata_id"]))
                     for c in cands} if validate else None)
        kept = ([c for c in cands if validity.get(c["wikidata_id"]) is not False]
                if validate else cands)

        # Cache on everything that determines the answer — the model, its
        # temperature, and the exact candidate list shown — so identical inputs
        # across different k reuse one call, and changing the decoding settings
        # invalidates old entries instead of silently reusing them.
        # `cache_identity` renders a Google model by its bare name, which is how
        # the 1,443 entries predating the provider layer were keyed. Using
        # `generator_model` directly would spell the same model differently and
        # invalidate every one of them.
        key = hashlib.sha256(
            f"{cache_identity(linker.generator_model)}|{linker.temperature}|"
            f"{system}|{mid}|{run}|"
            f"{'|'.join(c['wikidata_id'] for c in kept)}".encode()).hexdigest()
        hit = llm_cache.get(key)
        if hit is None:
            row = sheet.get(mid)
            res = linker.link_instrumented(
                search_term=p["label"],
                context_query=(row or {}).get("object_refs", "")[:400],
                entity_type=p["entity_type"],
                candidates=cands,
                validity=validity,
                top_k=k,
                validate=validate,
                verbose=verbose,
            )
            # Never cache a failure: doing so would freeze a transient API
            # error into the results, and a re-run would keep serving it.
            if res["outcome"] != "error_api":
                llm_cache[key] = res
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(llm_cache), encoding="utf-8")
            hit = res
            marker = "" if res["outcome"] != "error_api" else "  !! not cached"
        else:
            marker = " (cached)"

        rec = base_record(p, sheet.get(mid), system, k, run, eck_version,
                          [c["wikidata_id"] for c in kept],
                          model=linker.generator_model)
        rec.update(hit)
        # The cache is keyed on the candidate list the model saw, which can be
        # identical across different k once filtering has been applied. The
        # model's decision is therefore reusable, but the per-cell counts are
        # not — recompute them for *this* k rather than inheriting the ones
        # stored with the cached decision.
        rec["n_retrieved"] = len(cands)
        rec["n_validated"] = len(kept)
        rec["n_unevaluable"] = (
            sum(1 for c in cands if validity.get(c["wikidata_id"]) is None)
            if validate else 0)
        out.append(rec)
        print(f"    [{i:>3}/{len(pools)}] {mid} {p['label'][:26]:<28}"
              f" {hit['outcome']:<22} {hit['predicted_qid'] or '-'}{marker}",
              flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pools", type=Path, default=DATA / "pools_raw.jsonl")
    ap.add_argument("--sheet", type=Path, default=DATA / "annotation_sheet.csv")
    ap.add_argument("--eck-version", default="v1")
    ap.add_argument("--valid", type=Path, default=None)
    ap.add_argument("--systems", nargs="+", default=["S0", "S1", "S2", "S3"],
                    choices=["S0", "S1", "S2", "S3"])
    ap.add_argument("--k", nargs="+", type=int, default=[5, 10, 25, 50])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="selection temperature (spec §11: 0.0)")
    ap.add_argument("--out-dir", type=Path, default=PRED)
    args = ap.parse_args(argv)

    valid_path = args.valid or DATA / f"pools_valid_{args.eck_version}.jsonl"
    pools, valid, sheet = load_inputs(args.pools, valid_path, args.sheet)
    if args.limit:
        pools = pools[:args.limit]
    print(f"{len(pools)} mentions | systems={args.systems} | k={args.k} "
          f"| runs={args.runs} | eck={args.eck_version}\n")

    linker = None
    if {"S2", "S3"} & set(args.systems):
        provider, _ = parse_model_spec(args.model)
        env_var = API_KEY_ENV[provider]
        key = args.api_key or os.environ.get(env_var, "")
        if not key:
            print(f"S2/S3 need a {provider} key: pass --api-key or set ${env_var}.")
            return 1
        linker = TypedRAGEntityLinker(
            generator_model=args.model, generator_api_key=key,
            strict_types=True, cache_path=None, temperature=args.temperature)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    llm_cache_path = CACHE / "llm_selections.json"
    llm_cache = {}
    if llm_cache_path.exists():
        try:
            llm_cache = json.loads(llm_cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  warning: could not read LLM cache: {e}")

    started = time.time()
    for system in args.systems:
        # S0 and S1 are deterministic, so one run is the whole story.
        runs = 1 if system in ("S0", "S1") else args.runs
        ks = [args.k[0]] if system == "S0" and len(args.k) > 1 else args.k
        if system == "S0" and len(args.k) > 1:
            print(f"{system}: k-independent (rank 1), computing once at k={ks[0]}")
        for k in ks:
            for run in range(1, runs + 1):
                # S0/S1 read straight off the frozen artefacts, so tagging them
                # with a model would imply a dependency they do not have.
                tag = "" if system in ("S0", "S1") else f"_{model_slug(args.model)}"
                out_file = args.out_dir / f"{system}{tag}_k{k}_run{run}.jsonl"
                if out_file.exists():
                    # Redo a cell that holds API failures; keep a clean one.
                    prior = [json.loads(l) for l in open(out_file, encoding="utf-8")]
                    bad = sum(1 for r in prior if r["outcome"] == "error_api")
                    if not bad:
                        print(f"  {out_file.name} exists — skipping")
                        continue
                    print(f"  {out_file.name} has {bad} error_api — redoing")
                print(f"\n>>> {system} k={k} run={run}", flush=True)
                if system in ("S0", "S1"):
                    recs = run_offline(system, pools, valid, sheet, k,
                                       args.eck_version)
                else:
                    recs = run_llm(system, linker, pools, valid, sheet, k, run,
                                   args.eck_version, llm_cache, llm_cache_path)
                with open(out_file, "w", encoding="utf-8") as fh:
                    for r in recs:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                counts = {}
                for r in recs:
                    counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
                print(f"  wrote {out_file.name}: "
                      + ", ".join(f"{k2}={v}" for k2, v in sorted(counts.items())))

    print(f"\ndone in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
