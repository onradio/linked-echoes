#!/usr/bin/env python3
"""The model dimension, resolved in one place for every stage.

    python models.py           # what is configured, and whether its key is present
    python models.py --probe   # one live call per model, to check access before a sweep

`config.yaml` lists the models under comparison as `provider:model`. Everything
else — filenames, prediction records, cache keys, scoring groups — refers to a
model by its **slug**, a filesystem-safe form of the same id:

    google:gemini-2.5-flash      ->  google-gemini-2-5-flash

The first configured model is the **reference**: the one the pipeline deploys,
the default for a single-model run, and the baseline that paired comparisons are
taken against. Adding a model to the list must not change what the reference is,
so it is defined by position rather than by name.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
sys.path.insert(0, str(SRC))

from llm import (  # noqa: E402
    API_KEY_ENV,
    SCHEMA_CATEGORY,
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
    cfg = yaml.safe_load(Path(config_path).read_text())
    entries = cfg["llm"]["models"]
    models = []
    for position, entry in enumerate(entries):
        model_id = entry["id"] if isinstance(entry, dict) else entry
        label = entry.get("label", model_id) if isinstance(entry, dict) else model_id
        provider, _ = parse_model_spec(model_id)
        models.append({"id": model_id, "label": label, "slug": slug(model_id),
                       "provider": provider, "reference": position == 0})
    return models


def select_models(models: list, wanted: list = None) -> list:
    """
    Filters by id, slug or label. Unknown names are an error rather than an empty
    sweep — a typo that silently ran nothing would be discovered only after the
    scoring stage found no predictions.
    """
    if not wanted:
        return models
    chosen, unknown = [], []
    for name in wanted:
        match = next((m for m in models
                      if name in (m["id"], m["slug"], m["label"])), None)
        (chosen.append(match) if match else unknown.append(name))
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}. Configured: "
                         + ", ".join(m["id"] for m in models))
    return chosen


def reference_model(models: list) -> dict:
    return next(m for m in models if m["reference"])


def missing_keys(models: list) -> dict:
    """Providers among the given models whose API key is absent from the environment."""
    return {m["provider"]: API_KEY_ENV[m["provider"]] for m in models
            if not os.environ.get(API_KEY_ENV[m["provider"]])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", action="store_true",
                    help="Make one live call per model to confirm access")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    args = ap.parse_args()

    models = load_models(Path(args.config))
    print(f"{len(models)} models configured\n")
    print(f"  {'slug':34s}{'id':32s}{'key':16s}{'role'}")
    for m in models:
        env = API_KEY_ENV[m["provider"]]
        key = "set" if os.environ.get(env) else f"MISSING {env}"
        print(f"  {m['slug']:34s}{m['id']:32s}{key:16s}"
              f"{'reference' if m['reference'] else ''}")

    if not args.probe:
        return 0

    print("\nprobing (one call each)...")
    failures = 0
    for m in models:
        try:
            client = build_client(m["id"], temperature=0.0)
            response = client.generate(
                "A Fender Stratocaster electric guitar.", SCHEMA_CATEGORY,
                system_instruction="Assign one broad category.",
                choices=["Musical Instrument", "Clothing", "Audiovisual Media",
                         "Textual & Graphic Work", "Merchandise",
                         "Personal Effects", "Award & Recognition"])
            print(f"  OK   {m['slug']:34s} -> {response.parsed!r}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL {m['slug']:34s} -> {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
