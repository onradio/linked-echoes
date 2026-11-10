#!/usr/bin/env python3
"""The model dimension, resolved in one place for every stage of this pack.

    python models.py           # what is configured, and whether its key is present
    python models.py --probe   # one live call per model, to check access before a sweep

`config.yaml` lists the models under comparison as `provider:model`. Everything else —
filenames, prediction records, cache keys, scoring groups — refers to a model by its
**slug**, a filesystem-safe form of the same id:

    google:gemini-2.5-flash      ->  google-gemini-2-5-flash

The first configured model is the **reference**: the one the pipeline deploys, the default
for a single-model run, and the baseline that paired comparisons are taken against. Adding
a model to the list must not change what the reference is, so it is defined by position
rather than by name.

This module is local to this pack by design (spec §1.2). The classification evaluation has
one with the same semantics; the two are deliberately not shared, so a change made for one
evaluation cannot move the other's published numbers.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"           # bmeapi/src
sys.path.insert(0, str(SRC))               # import bare, as the eval specs require

from llm import (  # noqa: E402
    API_KEY_ENV,
    SCHEMA_INT,
    build_client,
    parse_model_spec,
)


def slug(model_id: str) -> str:
    """Filesystem-safe form of a model id, stable across runs."""
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def load_models(config_path: Path = HERE / "config.yaml") -> list:
    """
    The configured models, in order, each as a dict with `id`, `label`, `slug`,
    `provider` and `reference`.
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    entries = cfg["llm"]["models"]
    models = []
    for position, entry in enumerate(entries):
        model_id = entry["id"] if isinstance(entry, dict) else entry
        label = entry.get("label", model_id) if isinstance(entry, dict) else model_id
        provider, _ = parse_model_spec(model_id)
        models.append({"id": model_id, "label": label, "slug": slug(model_id),
                       "provider": provider, "reference": position == 0})
    return models


def reference_model(models: list | None = None) -> dict:
    models = models or load_models()
    return models[0]


def select_models(models: list, wanted: list | None = None) -> list:
    """
    The subset named on the command line, matched on id or slug. An unknown name is an
    error rather than an empty sweep, which would otherwise look like a completed run
    that spent nothing.
    """
    if not wanted:
        return models
    by_name = {m["id"]: m for m in models} | {m["slug"]: m for m in models}
    chosen = []
    for name in wanted:
        if name not in by_name:
            raise SystemExit(f"unknown model {name!r}; configured: "
                             + ", ".join(m["id"] for m in models))
        chosen.append(by_name[name])
    return chosen


def key_present(model: dict) -> bool:
    return bool(os.environ.get(API_KEY_ENV[model["provider"]], ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", action="store_true",
                    help="make one trivial live call per model (spends a token or two)")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--models", nargs="+", help="subset, by id or slug")
    args = ap.parse_args()

    models = select_models(load_models(), args.models)
    print(f"{'model':32s} {'slug':30s} {'ref':4s} key")
    for m in models:
        env = API_KEY_ENV[m["provider"]]
        state = "set" if key_present(m) or args.api_key else f"MISSING ({env})"
        print(f"{m['id']:32s} {m['slug']:30s} {'yes' if m['reference'] else '':4s} {state}")

    if not args.probe:
        return 0

    print("\nprobing:")
    failures = 0
    for m in models:
        try:
            client = build_client(m["id"], api_key=args.api_key, temperature=0.0,
                                  timeout_s=30)
            response = client.generate("Reply with the integer 1 and nothing else.",
                                       SCHEMA_INT)
            ok = response.parsed == 1
            print(f"  {m['id']:32s} {'ok' if ok else f'unexpected: {response.parsed!r}'}")
            failures += 0 if ok else 1
        except Exception as exc:                      # noqa: BLE001 - report, don't crash
            print(f"  {m['id']:32s} FAIL {type(exc).__name__}: {str(exc)[:90]}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
