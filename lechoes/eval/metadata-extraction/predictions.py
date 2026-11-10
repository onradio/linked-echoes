#!/usr/bin/env python3
"""Prediction records and the run manifest (spec §1.3, §1.4).

Both runners write through this module, so `score.py` reads one shape rather than two.
The record is deliberately verbose: `predicted` holds what the system asserts and
`raw_response` what it produced before the evidence check, so the check's effect is
**measured** rather than inferred from a difference in totals.

Three invariants, each of which cost the superseded evaluation something:

* **A cell a system leaves empty is written as `[]`, never omitted.** 73% of gold cells are
  empty and abstention is a scored outcome (§4.3), so "asserted nothing" and "the key is
  missing" must stay distinguishable.
* **On failure, `predicted` is `null` and `error` carries the exception.** Such objects are
  excluded from that cell's metrics and counted, never scored as abstentions — the same
  rule the entity-linking spec applies to an `ASK` timeout.
* **`object_id` is the join key.** Nothing is ever aligned by list position; that is the
  defect that displaced all 62 rows of `eval_folder.py` (§6.12).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def paths(config: dict) -> dict:
    p = {k: HERE / v for k, v in config["paths"].items()}
    for directory in p.values():
        directory.mkdir(parents=True, exist_ok=True)
    return p


def prediction_path(config: dict, system: str, model_slug: str | None,
                    run: int) -> Path:
    """
    `<system>_<model-slug>_run<R>.jsonl`. The baselines call no model and so carry no
    slug: a result produced by one model says nothing about another, but a span tagger's
    output is not a model's opinion in that sense.
    """
    stem = f"{system}_{model_slug}_run{run}" if model_slug else f"{system}_run{run}"
    return paths(config)["predictions"] / f"{stem}.jsonl"


def record(*, object_id: str, case_number: int, system: str, model: str | None,
           run: int, evidence_scope: str, assertion_policy: str,
           predicted: dict | None, dropped_no_quote: list | None = None,
           unscored: dict | None = None, latency_ms: int | None = None,
           raw_response=None, error: str | None = None) -> dict:
    """One prediction record. Field order is fixed so the JSONL diffs cleanly."""
    return {
        "object_id": object_id,
        "case_number": case_number,
        "system": system,
        "model": model,
        "run": run,
        "evidence_scope": evidence_scope,
        "assertion_policy": assertion_policy,
        "predicted": predicted,
        "dropped_no_quote": dropped_no_quote or [],
        "unscored": unscored or {},
        "latency_ms": latency_ms,
        "raw_response": raw_response,
        "error": error,
    }


def empty_prediction(fields: list) -> dict:
    """Every scored field present and empty — the shape a record must always have."""
    return {field: [] for field in fields}


def dedupe(values: list, normalise, field: str) -> list:
    """
    Collapse values whose normalised forms agree, keeping the first surface form and its
    quote. Returning the same artist twice is one assertion, not two (§4.1 rule 1); the
    gold has no intra-cell duplicates but systems produce them freely.
    """
    out, seen = [], set()
    for item in values:
        key = normalise(item["value"], field)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def write(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read(path: Path) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def config_hash(config: dict) -> str:
    """Fingerprint of the configuration a run was produced under."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class Manifest:
    """
    One entry per `(system, model, run)`. `score.py` refuses to run if an expected cell is
    missing from it, so a partial sweep cannot be silently scored as a complete one.
    """

    def __init__(self, config: dict):
        self.path = paths(config)["predictions"] / "manifest.json"
        self.config = config
        self.entries = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8")).get(
                "entries", {})

    @staticmethod
    def key(system: str, model_slug: str | None, run: int) -> str:
        return f"{system}|{model_slug or '-'}|{run}"

    def add(self, *, system: str, model_slug: str | None, model_id: str | None,
            run: int, scope: str, policy: str, prompt_mode: str | None,
            objects: int, failures: int, dropped: int, cache_hits: int,
            cache_misses: int, prompt_sample: str | None = None,
            fewshot_block: str | None = None, extra: dict | None = None) -> None:
        self.entries[self.key(system, model_slug, run)] = {
            "system": system,
            "model": model_id,
            "model_slug": model_slug,
            "run": run,
            "evidence_scope": scope,
            "assertion_policy": policy,
            "prompt_mode": prompt_mode,
            "objects": objects,
            "failures": failures,
            "dropped_no_quote": dropped,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "config_hash": config_hash(self.config),
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # What the model actually saw, once per cell rather than per object.
            "prompt_sample": prompt_sample,
            "fewshot_block": fewshot_block,
            **(extra or {}),
        }

    def save(self, *, gold_baseline: dict | None = None) -> None:
        body = {"config_hash": config_hash(self.config), "entries": self.entries}
        if gold_baseline:
            # The alias-table and sheet hashes the run was scored against, so score.py
            # can refuse to mix predictions with a gold that has since moved (§2.5).
            body["gold"] = {k: gold_baseline.get(k) for k in
                            ("sheet_sha256", "aliases_sha256", "gold_values",
                             "supported", "inference_only")}
        self.path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")

    def has(self, system: str, model_slug: str | None, run: int) -> bool:
        return self.key(system, model_slug, run) in self.entries
