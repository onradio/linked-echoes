#!/usr/bin/env python3
"""Experiment A — category classification (spec §3.1).

    python run_category.py --systems C0 C1        # deterministic, no API key
    python run_category.py --estimate             # call count, no spending
    python run_category.py --models google:gemini-3.6-flash   # one model
    python run_category.py                        # every configured model

| Id | System |
| -- | ------ |
| C0 | majority class; always the same answer; the anchor every result is read against |
| C1 | nearest `CATEGORY_DEFINITIONS` by sentence embedding; deterministic, free |
| C2 | the LLM with `CATEGORY_ROLE` — the seven category names alone. **The system.** |
| C2-scoped | the same model with scope notes and few-shot examples: measured no better (p=1.0), and retained only because the article reports the comparison |

C0 and C1 are deterministic and run once; the LLM systems run three times at
temperature 0, like the selector. Nothing here uses the vocabulary indices: the
category decision precedes retrieval and routes it.

A prediction is correct if it matches the primary **or** the secondary gold
category (§4.1); `correct_strict` is against the primary alone, and the gap
between the two is the four composite objects.
"""
import argparse
import contextlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

from classification import (  # noqa: E402
    CATEGORY_DEFINITIONS,
    EmbeddingObjectCategoriser,
    ObjectCategoriser,
)
from gold import load_gold  # noqa: E402
from models import load_models, missing_keys, select_models  # noqa: E402

sys.path.insert(0, str(SRC))
from llm import CachingLLMClient, build_client  # noqa: E402

SYSTEMS = ("C0", "C1", "C2", "C2-scoped")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--api-key", default="")
    ap.add_argument("--systems", nargs="+", default=["C0", "C1", "C2"],
                    choices=list(SYSTEMS))
    ap.add_argument("--runs", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--models", nargs="+",
                    help="Subset of the configured models, by id, slug or label. "
                         "The LLM systems are run once per model.")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    llm = cfg["llm"]
    objects = load_gold(HERE / cfg["paths"]["annotations"])
    if args.limit:
        objects = objects[:args.limit]
    runs = args.runs or cfg["runs"]
    majority = cfg["category"]["majority_class"]
    out_dir = HERE / cfg["paths"]["predictions"]
    out_dir.mkdir(parents=True, exist_ok=True)

    models = select_models(load_models(Path(args.config)), args.models)

    def out_path(system: str, run: int, model_slug: str = "") -> Path:
        # C0 and C1 use no LLM, so they carry no model tag: tagging them would
        # duplicate identical files once per model and imply a dependency that
        # does not exist.
        if not system.startswith("C2"):
            return out_dir / f"category_{system}_run{run}.jsonl"
        return out_dir / f"category_{system}_{model_slug}_run{run}.jsonl"

    # Only the LLM systems vary between runs and across models.
    cells = []
    for system in args.systems:
        if system.startswith("C2"):
            cells += [(system, r, m) for m in models for r in range(1, runs + 1)]
        else:
            cells.append((system, 1, None))
    todo = [c for c in cells
            if args.force or not out_path(c[0], c[1], c[2]["slug"] if c[2] else "").exists()]
    llm_calls = sum(len(objects) for c in todo if c[0].startswith("C2"))

    print(f"{len(objects)} objects | systems {args.systems} | {runs} runs")
    for m in models:
        print(f"  model {m['slug']:34s} {m['id']}"
              + ("   (reference)" if m["reference"] else ""))
    print(f"cells: {len(cells)} total, {len(todo)} to run")
    print(f"upper bound on LLM calls: {llm_calls} (minus anything already cached)")
    if args.estimate:
        return 0
    if not todo:
        print("nothing to do (use --force to overwrite)")
        return 0

    llm_models = [c[2] for c in todo if c[2]]
    absent = missing_keys(llm_models)
    if absent and not args.api_key:
        print("error: missing API key(s): "
              + ", ".join(f"{p} -> ${env}" for p, env in absent.items()))
        return 1

    embedder = None
    if "C1" in {c[0] for c in todo}:
        print("loading the sentence-transformer for C1...")
        with contextlib.redirect_stdout(io.StringIO()):
            embedder = EmbeddingObjectCategoriser(
                embedding_model=cfg["retrieval"]["embedding_model"])

    written = []
    for system, run, model in todo:
        categoriser = None
        cache = None
        if system.startswith("C2"):
            variant = "scoped" if system.endswith("scoped") else "minimal"
            cache = CachingLLMClient(
                client=build_client(model["id"], api_key=args.api_key,
                                    temperature=llm["temperature"]),
                path=HERE / "data" / "llm_cache" /
                     f"category_{system}_{model['slug']}_run{run}.json",
                run_index=run)
            categoriser = ObjectCategoriser(
                temperature=llm["temperature"], prompt_variant=variant,
                generator=cache)

        records, correct, strict = [], 0, 0
        for obj in objects:
            primary = obj["category_gold_primary"]
            secondary = obj["category_gold_secondary"]
            raw = None
            if system == "C0":
                predicted = majority
            elif system == "C1":
                predicted = embedder.classify(obj["object_text"])
            else:
                result = categoriser.classify_instrumented(obj["object_text"])
                predicted, raw = result["predicted"], result["raw_response"]

            is_correct = predicted == primary or (bool(secondary) and predicted == secondary)
            is_strict = predicted == primary
            correct += is_correct
            strict += is_strict
            records.append({
                "object_id": obj["object_id"], "case_number": obj["case_number"],
                "system": system, "run": run,
                "model": model["id"] if model else None,
                "model_slug": model["slug"] if model else None,
                "model_label": model["label"] if model else None,
                "predicted": predicted,
                "gold_primary": primary, "gold_secondary": secondary,
                "correct": is_correct, "correct_strict": is_strict,
                "raw_response": raw,
            })

        path = out_path(system, run, model['slug'] if model else '')
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if cache:
            cache.save()
        written.append(path.name)
        note = f" | cache {cache.hits} hit / {cache.misses} new" if cache else ""
        tag = f"{model['slug']:28s}" if model else f"{'(no LLM)':28s}"
        print(f"  {system:10s}{tag} run={run}  "
              f"accuracy={correct / len(objects):.3f} "
              f"strict={strict / len(objects):.3f}{note}")

    (out_dir / "category_manifest.json").write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_objects": len(objects), "systems": args.systems, "runs": runs,
        "majority_class": majority, "temperature": llm["temperature"],
        "models": [{k: m[k] for k in ("id", "label", "slug", "reference")}
                   for m in models],
        "definitions": sorted(CATEGORY_DEFINITIONS),
        "files_written": written,
    }, indent=2))
    print(f"\n{len(written)} files written -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
