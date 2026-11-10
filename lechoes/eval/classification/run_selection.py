#!/usr/bin/env python3
"""Experiment C — type selection (spec §3.3). The only stage that spends API budget.

    python run_selection.py --estimate           # how many calls, no spending
    python run_selection.py --systems T0         # retrieval top-1: no LLM at all
    python run_selection.py --limit 3 --runs 1   # smoke test
    python run_selection.py --api-key AIza...    # the full sweep
    python run_selection.py --configuration term_hint    # one configuration only

| Id | Pipeline |
| -- | -------- |
| T0 | the top-ranked retrieved term; never abstains; deterministic |
| T1 | retrieve top-k, LLM selects one term or returns 0 to abstain (ours) |

Both systems are run over **both retrieval configurations** — `term_hint`, the
deployed pipeline, and `combine_name`, which retrieves best at large k — at every
k. Retrieval configuration is thus a reported factor rather than a choice made on
the evaluation set, and within a configuration k stays the only variable. T0 is
run per configuration too: its prediction is the top-ranked term of whichever pool
it is given, so it is not configuration-agnostic.

The selection prompt is identical across configurations, hint included: the two
differ only in retrieval.

Both systems run through `HierarchicalObjectClassifier` with the **gold primary
category** supplied, which skips categorisation (§0.5) and exercises the shipped
routing: the specific vocabulary first, AAT only if the selector abstains. The
LLM client is wrapped in a disk cache keyed by `sha1(model, prompt, run_index)`,
so an interrupted sweep resumes without re-billing completed cells.
"""
import argparse
import contextlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

from classification import HierarchicalObjectClassifier  # noqa: E402
from llm import CachingLLMClient, build_client            # noqa: E402
from gold import load_gold, score_outcome                # noqa: E402
from models import load_models, missing_keys, select_models  # noqa: E402
from build_indices import index_stem                     # noqa: E402
from run_retrieval import build_query                    # noqa: E402

def build_vocab_paths(indices_dir: Path, similarity: str, strategy: str) -> dict:
    """Points the classifier at the tagged evaluation builds, not the production ones."""
    paths = {}
    for vocab in ("AAT", "MIMO", "Europeana", "PBCore"):
        stem = index_stem(vocab, similarity, strategy)
        paths[vocab] = {"index": str(indices_dir / f"{stem}.index"),
                        "metadata": str(indices_dir / f"{stem}_metadata.pkl")}
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--api-key", default="", help="API key; otherwise the provider's environment variable is used")
    ap.add_argument("--systems", nargs="+", default=["T0", "T1"], choices=["T0", "T1"])
    ap.add_argument("--k", nargs="+", type=int, help="Subset of the selection k grid")
    ap.add_argument("--runs", type=int, help="Override the number of runs")
    ap.add_argument("--limit", type=int, help="Only the first N objects")
    ap.add_argument("--objects", nargs="+",
                    help="Specific object ids, e.g. for a pilot that covers every "
                         "vocabulary, tier and the fallback path (--limit takes the "
                         "first N, which are all AAT-routed)")
    ap.add_argument("--estimate", action="store_true",
                    help="Report how many LLM calls the sweep needs, then exit")
    ap.add_argument("--configuration", nargs="+",
                    help="Subset of the retrieval configurations, by id")
    ap.add_argument("--models", nargs="+",
                    help="Subset of the configured models, by id, slug or label. "
                         "T1 is run once per model; T0 uses no LLM.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing predictions")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    sel = cfg["selection"]
    llm = cfg["llm"]
    objects = load_gold(HERE / cfg["paths"]["annotations"])
    if args.objects:
        wanted = set(args.objects)
        objects = [o for o in objects if o["object_id"] in wanted]
        missing = wanted - {o["object_id"] for o in objects}
        if missing:
            print(f"error: unknown object ids {sorted(missing)}")
            return 1
    if args.limit:
        objects = objects[:args.limit]

    configurations = sel["configurations"]
    if args.configuration:
        configurations = [c for c in configurations if c["id"] in args.configuration]
        if not configurations:
            print(f"error: no configuration matches {args.configuration}")
            return 1
    k_grid = args.k or sel["k_grid"]

    similarity = cfg["retrieval"]["similarity"][0]
    runs = args.runs or cfg["runs"]
    out_dir = HERE / cfg["paths"]["predictions"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = HERE / "data" / "llm_cache"

    models = select_models(load_models(Path(args.config)), args.models)

    def out_path(config_id: str, system: str, k: int, run: int,
                 model_slug: str = "") -> Path:
        # T0 reads rank 1 and calls no model, so it carries no model tag.
        if system != "T1":
            return out_dir / f"selection_{system}_{config_id}_k{k}_run{run}.jsonl"
        return out_dir / (f"selection_{system}_{config_id}_{model_slug}"
                          f"_k{k}_run{run}.jsonl")

    # T0 is deterministic and calls nothing, so it runs once per (config, k).
    cells = []
    for c in configurations:
        for system in args.systems:
            for k in k_grid:
                if system == "T1":
                    cells += [(c["id"], system, k, r, m)
                              for m in models for r in range(1, runs + 1)]
                else:
                    cells.append((c["id"], system, k, 1, None))
    todo = [c for c in cells if args.force
            or not out_path(c[0], c[1], c[2], c[3],
                            c[4]["slug"] if c[4] else "").exists()]
    llm_cells = [c for c in todo if c[1] == "T1"]

    print(f"{len(objects)} objects | similarity {similarity} | k {k_grid} | {runs} runs")
    for c in configurations:
        print(f"  configuration {c['id']:14s} {c['index_strategy']} / {c['query']}")
    for m in models:
        print(f"  model         {m['slug']:34s} {m['id']}"
              + ("   (reference)" if m["reference"] else ""))
    print(f"cells: {len(cells)} total, {len(todo)} to run "
          f"({len(llm_cells)} of them LLM cells)")
    print(f"upper bound on LLM calls: {len(llm_cells) * len(objects)} "
          f"(plus one per abstention that triggers the AAT fallback), "
          f"minus anything already cached")
    if args.estimate:
        return 0
    if not todo:
        print("nothing to do (use --force to overwrite)")
        return 0

    absent = missing_keys([c[4] for c in todo if c[4]])
    if absent and not args.api_key:
        print("error: missing API key(s): "
              + ", ".join(f"{p} -> ${env}" for p, env in absent.items()))
        return 1

    written = []
    for config in configurations:
        config_id, strategy, query_id = config["id"], config["index_strategy"], config["query"]
        config_cells = [c for c in todo if c[0] == config_id]
        if not config_cells:
            continue

        print(f"\n--- {config_id}: loading {strategy} indices ---")
        vocab_paths = build_vocab_paths(
            HERE / cfg["paths"]["indices"], similarity, strategy)
        with contextlib.redirect_stdout(io.StringIO()):
            # Built retrieval-only; each T1 cell installs its own caching client
            # for the model under test, so one set of indices serves every model.
            classifier = HierarchicalObjectClassifier(
                embedding_model=cfg["retrieval"]["embedding_model"],
                vocab_paths=vocab_paths,
                temperature=llm["temperature"],
                require_generator=False,
                categoriser=object(),   # unused: the gold category is supplied
            )
        print("    " + ", ".join(f"{v}={c.index.ntotal}"
                                 for v, c in classifier.specific_classifiers.items()))
        written.extend(run_configuration(
            classifier, config_cells, objects, config_id, strategy, query_id,
            sel, cache_dir, out_path, llm, args.api_key))

    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_objects": len(objects), "systems": args.systems,
        "similarity": similarity,
        "configurations": configurations,
        "models": [{k: m[k] for k in ("id", "label", "slug", "reference")}
                   for m in models],
        "hint_in_prompt": sel.get("hint_in_prompt", "always"),
        "k_grid": k_grid, "runs": runs, "llm": llm,
        "files_written": written,
    }
    (out_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(written)} files written -> {out_dir}")
    return 0


def run_configuration(classifier, cells, objects, config_id, strategy, query_id,
                      sel, cache_dir, out_path, llm, api_key) -> list:
    """Runs every (system, k, run, model) cell for one retrieval configuration."""
    written = []
    for config_id_, system, k, run, model in cells:
        path = out_path(config_id_, system, k, run,
                        model["slug"] if model else "")
        caches = []
        if system == "T1":
            # One caching client per (model, run), shared by every vocabulary so
            # that a fallback to AAT bills and caches like any other call.
            cache = CachingLLMClient(
                client=build_client(model["id"], api_key=api_key,
                                    temperature=llm["temperature"]),
                path=cache_dir / f"selection_{model['slug']}_run{run}.json",
                run_index=run)
            for clf in classifier.specific_classifiers.values():
                clf.generator = cache
            caches.append(cache)

        records, counts = [], {}
        for obj in objects:
            query, hint_fallback = build_query(obj, query_id)
            if system == "T0":
                # Retrieval top-1 from the routed vocabulary. No LLM, no
                # abstention, and therefore no fallback: T0 is the pure
                # retrieval baseline the selector has to beat.
                clf = classifier.specific_classifiers[obj["routed_vocab"]]
                with contextlib.redirect_stdout(io.StringIO()):
                    top = clf.retrieve(query, top_k=1, return_scores=True)
                predicted = top[0][0]["uri"] if top else None
                result = {"vocab": obj["routed_vocab"], "fallback_triggered": False,
                          "abstained": False, "raw_response": None,
                          "outcome_detail": "retrieval_top1",
                          "predicted_rank": 1 if top else None, "attempts": []}
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    r = classifier.classify_instrumented(
                        description=obj["object_text"],
                        top_k=k,
                        prompt_type=sel["prompt"],
                        object_category=obj["category_gold_primary"],
                        predicted_type=obj["raw_type_hint"],
                        retrieval_query=query,
                    )
                idx = r["selected_indices"]
                predicted = r["candidates"][idx[0]]["uri"] if idx else None
                result = {"vocab": r["vocab"],
                          "fallback_triggered": r["fallback_triggered"],
                          "abstained": bool(r["attempts"] and not idx
                                            and r["outcome"] == "abstained"),
                          "raw_response": r["attempts"][-1]["raw_response"] if r["attempts"] else None,
                          "outcome_detail": r["outcome"],
                          # Rank of the chosen candidate: whether the selector
                          # ratifies retrieval's top-1 or overrides it is most of
                          # what distinguishes T1 from T0.
                          "predicted_rank": idx[0] + 1 if idx else None,
                          # One entry per vocabulary tried. The first attempt's
                          # response is the evidence for how often the specific
                          # vocabulary abstains, which the fallback share rests on.
                          "attempts": [{"vocab": a["vocab"], "outcome": a["outcome"],
                                        "raw_response": a["raw_response"],
                                        "n_candidates": len(a["candidates"])}
                                       for a in r["attempts"]]}

            outcome = ("error_malformed" if result["outcome_detail"] in
                       ("error_malformed", "error_api", "error_routing")
                       else score_outcome(obj, predicted, result["abstained"]))
            counts[outcome] = counts.get(outcome, 0) + 1

            records.append({
                "object_id": obj["object_id"], "case_number": obj["case_number"],
                "system": system, "configuration": config_id, "k": k, "run": run,
                "model": model["id"] if model else None,
                "model_slug": model["slug"] if model else None,
                "model_label": model["label"] if model else None,
                "index_strategy": strategy, "query": query_id,
                "hint_fallback": hint_fallback,
                "vocab": result["vocab"],
                "fallback_triggered": result["fallback_triggered"],
                "tier": obj["tier"], "gold_uris": obj["gold_uris"],
                "gold_reachable": obj["reachable"],
                "predicted_uri": predicted,
                "predicted_rank": result.get("predicted_rank"),
                "abstained": result["abstained"],
                "outcome": outcome,
                "outcome_detail": result["outcome_detail"],
                "raw_response": result["raw_response"],
                "attempts": result.get("attempts", []),
            })

        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for cache in caches:
            cache.save()
        written.append(path.name)
        cache_note = (f" | cache {caches[0].hits} hit / {caches[0].misses} new"
                      if caches else "")
        tag = model["slug"] if model else "(no LLM)"
        print(f"  {system} {tag:28s} k={k:<3} run={run}  "
              + " ".join(f"{o}={n}" for o, n in sorted(counts.items())) + cache_note)
    return written


if __name__ == "__main__":
    raise SystemExit(main())
