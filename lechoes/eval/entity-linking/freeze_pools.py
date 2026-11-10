#!/usr/bin/env python3
"""Stage 2 — freeze the raw candidate pools (spec §4).

Retrieves the top `k_max` Wikidata candidates for every mention in the annotation
sheet and writes them verbatim to `pools_raw.jsonl`, together with each
candidate's P31 classes. Wikidata's search ranking drifts, so this file is the
immutable basis of every later stage: truncation to a smaller `k` is a slice of
this list, never a re-query.

    python freeze_pools.py                    # writes data/pools_raw.jsonl
    python freeze_pools.py --limit 5          # dry run over the first 5 mentions
    python freeze_pools.py --k-max 50 --force # overwrite an existing pool file

Integrity rules, both deliberate:

* **A failed request is never frozen as an empty pool.** `wikidata_entity_lookup`
  returns None on failure and `[]` when Wikidata genuinely knows nothing by that
  name; conflating them would turn an outage into a permanent retrieval miss.
  If any mention fails, nothing is written and the run exits non-zero — re-run
  and the on-disk cache will replay the successes for free.
* **An existing pool file is not overwritten** without `--force`, because
  everything downstream is keyed to it.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"          # bmeapi/src
sys.path.insert(0, str(SRC))

from linker import (  # noqa: E402
    wikidata_entity_lookup, wikidata_get_claims, truthy_values,
)

DATA = HERE / "data"
CACHE = DATA / ".cache"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  warning: could not read {path.name}: {e}")
    return default


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def read_mentions(sheet: Path) -> list:
    with open(sheet, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    missing = [r for r in rows if not (r.get("label") or "").strip()]
    if missing:
        raise SystemExit(f"{len(missing)} sheet rows have an empty label; fix the sheet first")
    return rows


def fetch_pools(mentions, k_max, search_cache, cache_path):
    """Returns (pools, failures). Cache hits do not touch the network."""
    pools, failures = {}, []
    for i, m in enumerate(mentions, start=1):
        mid, label = m["mention_id"], m["label"]
        key = f"{label}||{k_max}"
        hit = search_cache.get(key)
        if hit is None:
            candidates = wikidata_entity_lookup(label, num_results=k_max)
            if candidates is None:            # request failed, not "no results"
                print(f"  [{i:>3}/{len(mentions)}] {mid} {label!r}: REQUEST FAILED")
                failures.append((mid, label))
                continue
            hit = {"fetched_at": utcnow(), "candidates": candidates}
            search_cache[key] = hit
            save_json(cache_path, search_cache)   # checkpoint every mention
            marker = ""
        else:
            marker = " (cached)"
        pools[mid] = hit
        print(f"  [{i:>3}/{len(mentions)}] {mid} {label[:38]:<40} "
              f"{len(hit['candidates']):>2} candidates{marker}", flush=True)
    return pools, failures


def fetch_claims(pools, claims_cache, cache_path):
    """Batch-fetch P31 for every candidate not already cached."""
    wanted = {c["wikidata_id"] for p in pools.values() for c in p["candidates"]}
    todo = sorted(wanted - set(claims_cache))
    print(f"\nP31 for {len(wanted)} distinct candidates "
          f"({len(todo)} to fetch, {len(wanted) - len(todo)} cached)")
    for start in range(0, len(todo), 50):
        chunk = todo[start:start + 50]
        got = wikidata_get_claims(chunk, properties=("P31",))
        if got is None:
            return False
        for qid, record in got.items():
            claims_cache[qid] = record["P31"]
        # Ids that came back under a different key (redirects) still need an entry.
        for qid in chunk:
            claims_cache.setdefault(qid, [])
        save_json(cache_path, claims_cache)
        print(f"  {min(start + 50, len(todo)):>4}/{len(todo)}", flush=True)
    return True


def build_records(mentions, pools, claims_cache, k_max):
    records = []
    for m in mentions:
        mid = m["mention_id"]
        if mid not in pools:
            continue
        pool = pools[mid]
        candidates = []
        for rank, c in enumerate(pool["candidates"], start=1):
            statements = claims_cache.get(c["wikidata_id"], [])
            candidates.append({
                "rank": rank,
                "wikidata_id": c["wikidata_id"],
                "label": c.get("label"),
                "description": c.get("description"),
                # what wdt:P31 exposes, i.e. what the ECKs actually match on
                "p31": truthy_values(statements),
                # every statement with its rank, for eck_report.py diagnostics
                "p31_all": statements,
            })
        records.append({
            "mention_id": mid,
            "label": m["label"],
            "entity_type": m["entity_type"],
            "retrieved_at": pool["fetched_at"],
            "k_max": k_max,
            "n_candidates": len(candidates),
            "candidates": candidates,
        })
    return records


def summarise(records, k_sweep=(5, 10, 25, 50)) -> None:
    sizes = [r["n_candidates"] for r in records]
    if not sizes:
        return
    print(f"\n{'=' * 70}\nPOOL SUMMARY\n{'=' * 70}")
    print(f"mentions          : {len(sizes)}")
    print(f"mean / median     : {sum(sizes) / len(sizes):.1f} / {sorted(sizes)[len(sizes) // 2]}")
    print(f"empty pools       : {sum(1 for s in sizes if s == 0)}")
    for k in k_sweep:
        n = sum(1 for s in sizes if s >= k)
        print(f"pools with >= {k:>2}  : {n:>4} / {len(sizes)}  ({100 * n / len(sizes):.0f}%)")

    levels = Counter(len({min(k, s) for k in k_sweep}) for s in sizes)
    print("\ndistinct pools across the k sweep (1 = every k identical):")
    for n_levels in sorted(levels):
        print(f"  {n_levels}: {levels[n_levels]:>4} mentions")

    by_type = {}
    for r in records:
        by_type.setdefault(r["entity_type"], []).append(r["n_candidates"])
    print(f"\n{'entity type':<14}{'n':>4}{'mean':>7}{'empty':>7}")
    print("-" * 32)
    for etype, vals in by_type.items():
        print(f"{etype:<14}{len(vals):>4}{sum(vals) / len(vals):>7.1f}"
              f"{sum(1 for v in vals if v == 0):>7}")

    empty = [r for r in records if r["n_candidates"] == 0]
    if empty:
        print(f"\nempty pools (guaranteed retrieval miss for every system):")
        for r in empty:
            print(f"  {r['mention_id']}  {r['entity_type']:<14} {r['label']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheet", type=Path, default=DATA / "annotation_sheet.csv")
    ap.add_argument("--out", type=Path, default=DATA / "pools_raw.jsonl")
    ap.add_argument("--k-max", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N mentions (dry run)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing pool file")
    args = ap.parse_args(argv)

    if args.out.exists() and not args.force and args.limit is None:
        print(f"{args.out} already exists and the frozen pool is immutable.\n"
              f"Pass --force only if you intend to re-freeze and invalidate every\n"
              f"downstream artefact keyed to it.")
        return 1

    mentions = read_mentions(args.sheet)
    if args.limit:
        mentions = mentions[:args.limit]
    print(f"{len(mentions)} mentions from {args.sheet.name}, k_max={args.k_max}\n")

    search_cache_path = CACHE / f"search_k{args.k_max}.json"
    claims_cache_path = CACHE / "claims_p31.json"
    search_cache = load_json(search_cache_path, {})
    claims_cache = load_json(claims_cache_path, {})

    pools, failures = fetch_pools(mentions, args.k_max, search_cache, search_cache_path)

    if failures:
        print(f"\n{len(failures)} mention(s) could not be retrieved:")
        for mid, label in failures:
            print(f"  {mid}  {label}")
        print("\nNothing written — the frozen pool must be complete.\n"
              "Re-run to retry; successful lookups are cached and will not be repeated.")
        return 1

    if not fetch_claims(pools, claims_cache, claims_cache_path):
        print("\nP31 fetch failed; nothing written. Re-run to resume from cache.")
        return 1

    records = build_records(mentions, pools, claims_cache, args.k_max)

    if args.limit:
        print(f"\n--limit given: dry run, {len(records)} records not written")
        summarise(records)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(records)} pools to {args.out}")
    summarise(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
