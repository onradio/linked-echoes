#!/usr/bin/env python3
"""Stage 3 — validation cache (spec §5).

Evaluates every candidate in `pools_raw.jsonl` against its mention's ECK and
writes `pools_valid_<eck_version>.jsonl`, one record per (mention, candidate):

    {"mention_id": "M-003", "wikidata_id": "Q…", "eck_valid": true, "eck_version": "v1"}

`eck_valid` is three-valued: `true`, `false`, or `"error_api"`. A query that
could not be run is **never** recorded as `false` — that would score a network
failure as a validator rejection.

    python validate_pools.py                     # writes data/pools_valid_v1.jsonl
    python validate_pools.py --eck-version v2    # after an ECK revision

Re-running is cheap: verdicts are cached per (entity_type, qid). The cache
records a hash of each rendered ASK query and drops any verdict whose query has
since changed, so editing `entichecks.py` without bumping `--eck-version`
cannot silently serve stale results.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"          # bmeapi/src
sys.path.insert(0, str(SRC))

from entichecks import get_eck, normalise_type      # noqa: E402
from linker import validate_entities_batch          # noqa: E402

DATA = HERE / "data"
CACHE = DATA / ".cache"


def query_hash(entity_type: str) -> str:
    eck = get_eck(entity_type)
    return hashlib.sha256(eck["wd"].encode("utf-8")).hexdigest()[:16] if eck else ""


def load_cache(path: Path, types) -> dict:
    """Load cached verdicts, dropping any whose ECK query has changed."""
    blank = {"queries": {}, "verdicts": {}}
    if not path.exists():
        return blank
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  warning: could not read cache ({e}); starting fresh")
        return blank
    cache.setdefault("queries", {})
    cache.setdefault("verdicts", {})
    stale = {t for t in types if cache["queries"].get(t) not in (None, query_hash(t))}
    if stale:
        print(f"  ECK query changed for {sorted(stale)} — dropping their cached verdicts")
        cache["verdicts"] = {k: v for k, v in cache["verdicts"].items()
                             if k.split("||", 1)[0] not in stale}
    return cache


def save_cache(path: Path, cache: dict, types) -> None:
    for t in types:
        cache["queries"][t] = query_hash(t)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pools", type=Path, default=DATA / "pools_raw.jsonl")
    ap.add_argument("--sheet", type=Path, default=DATA / "annotation_sheet.csv")
    ap.add_argument("--eck-version", default="v1")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--chunk", type=int, default=4,
                    help="candidates per SPARQL request (default 4)")
    args = ap.parse_args(argv)
    out_path = args.out or DATA / f"pools_valid_{args.eck_version}.jsonl"

    pools = [json.loads(l) for l in open(args.pools, encoding="utf-8")]
    types = sorted({p["entity_type"] for p in pools})

    unresolved = [t for t in types if get_eck(t) is None]
    if unresolved:
        print(f"No ECK for entity type(s): {unresolved}")
        return 1
    print(f"{len(pools)} pools, {sum(p['n_candidates'] for p in pools)} candidates, "
          f"{len(types)} types, eck_version={args.eck_version}\n")

    cache_path = CACHE / f"eck_{args.eck_version}.json"
    cache = load_cache(cache_path, types)
    verdicts = cache["verdicts"]

    # Work out what still needs asking, grouped by type so one ECK covers a batch.
    todo = defaultdict(list)
    for p in pools:
        for c in p["candidates"]:
            key = f"{p['entity_type']}||{c['wikidata_id']}"
            if key not in verdicts or verdicts[key] == "error_api":
                todo[p["entity_type"]].append(c["wikidata_id"])
    n_todo = sum(len(set(v)) for v in todo.values())
    print(f"{n_todo} (type, entity) pairs to check "
          f"({len(verdicts)} cached)\n")

    # WDQS latency varies by more than an order of magnitude for the same query,
    # so there is no chunk size that is reliably best; a moderate one plus a
    # resumable cache beats trying to tune against the noise.
    started, done_total = time.time(), 0
    for etype, qids in todo.items():
        eck = get_eck(etype)
        uniq = sorted(set(qids))
        print(f"\n  {etype:<14} {len(uniq):>4} to check", flush=True)
        for start in range(0, len(uniq), args.chunk):
            batch = uniq[start:start + args.chunk]
            got = validate_entities_batch(batch, eck, chunk=args.chunk)
            for qid in batch:
                value = "error_api" if got is None else got.get(qid, "error_api")
                verdicts[f"{etype}||{qid}"] = value
            save_cache(cache_path, cache, types)   # checkpoint every batch
            done_total += len(batch)
            elapsed = time.time() - started
            rate = done_total / elapsed
            eta = (n_todo - done_total) / rate / 60 if rate else 0
            print(f"      {min(start + args.chunk, len(uniq)):>4}/{len(uniq)}"
                  f"   [{done_total}/{n_todo} overall, {rate:.1f}/s, ~{eta:.0f} min left]",
                  flush=True)

    # --- write the cache file -------------------------------------------------
    n_err = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for p in pools:
            for c in p["candidates"]:
                value = verdicts.get(f"{p['entity_type']}||{c['wikidata_id']}", "error_api")
                if value == "error_api":
                    n_err += 1
                fh.write(json.dumps({
                    "mention_id": p["mention_id"],
                    "wikidata_id": c["wikidata_id"],
                    "eck_valid": value,
                    "eck_version": args.eck_version,
                }, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path}")
    if n_err:
        print(f"  !! {n_err} candidate(s) recorded as error_api — re-run to retry them")

    report(pools, verdicts, args.sheet, args.eck_version)
    return 0


def report(pools, verdicts, sheet_path, version) -> None:
    def verdict(etype, qid):
        return verdicts.get(f"{etype}||{qid}", "error_api")

    print(f"\n{'=' * 76}\nPRUNING BY ENTITY TYPE (eck_version={version})\n{'=' * 76}")
    print(f"{'entity type':<14}{'mentions':>9}{'N':>7}{'M':>7}{'mean N':>8}{'mean M':>8}{'err':>6}")
    print("-" * 59)
    tot_n = tot_m = tot_e = 0
    by_type = defaultdict(lambda: [0, 0, 0, 0])
    for p in pools:
        row = by_type[p["entity_type"]]
        row[0] += 1
        for c in p["candidates"]:
            v = verdict(p["entity_type"], c["wikidata_id"])
            row[1] += 1
            row[2] += 1 if v is True else 0
            row[3] += 1 if v == "error_api" else 0
    for etype, (n_m, n, m, e) in by_type.items():
        tot_n += n; tot_m += m; tot_e += e
        print(f"{etype:<14}{n_m:>9}{n:>7}{m:>7}{n / n_m:>8.1f}{m / n_m:>8.1f}{e:>6}")
    print("-" * 59)
    n_mentions = len(pools)
    print(f"{'ALL':<14}{n_mentions:>9}{tot_n:>7}{tot_m:>7}"
          f"{tot_n / n_mentions:>8.1f}{tot_m / n_mentions:>8.1f}{tot_e:>6}")

    if not sheet_path.exists():
        return
    rows = {r["mention_id"]: r for r in csv.DictReader(
        open(sheet_path, newline="", encoding="utf-8"))}
    if not any(r["wikidata_qid"].strip() or r["nil_reason"].strip() for r in rows.values()):
        return

    # --- validator false rejection, spec §8.2 metric 5 ------------------------
    print(f"\n{'=' * 76}\nVALIDATOR FALSE REJECTION (gold in pool but ECK-invalid)\n{'=' * 76}")
    print(f"{'entity type':<14}{'gold in pool':>13}{'rejected':>10}{'FRR':>8}")
    print("-" * 45)
    rejected_rows = []
    frr_tot = [0, 0]
    for p in pools:
        r = rows.get(p["mention_id"])
        if not r:
            continue
        gold = {q.strip() for q in
                (r["wikidata_qid"] + ";" + r["alt_qids"]).replace(",", ";").split(";")
                if q.strip()}
        if not gold:
            continue
        in_pool = [c for c in p["candidates"] if c["wikidata_id"] in gold]
        if not in_pool:
            continue
        for c in in_pool:
            v = verdict(p["entity_type"], c["wikidata_id"])
            frr_tot[0] += 1
            if v is False:
                frr_tot[1] += 1
                rejected_rows.append((p, r, c))
    by_t = defaultdict(lambda: [0, 0])
    for p in pools:
        r = rows.get(p["mention_id"])
        if not r:
            continue
        gold = {q.strip() for q in
                (r["wikidata_qid"] + ";" + r["alt_qids"]).replace(",", ";").split(";")
                if q.strip()}
        for c in p["candidates"]:
            if c["wikidata_id"] in gold:
                by_t[p["entity_type"]][0] += 1
                if verdict(p["entity_type"], c["wikidata_id"]) is False:
                    by_t[p["entity_type"]][1] += 1
    for etype, (n, bad) in by_t.items():
        print(f"{etype:<14}{n:>13}{bad:>10}{(bad / n if n else 0):>8.2f}")
    print("-" * 45)
    print(f"{'ALL':<14}{frr_tot[0]:>13}{frr_tot[1]:>10}"
          f"{(frr_tot[1] / frr_tot[0] if frr_tot[0] else 0):>8.2f}")

    if rejected_rows:
        print(f"\nGold entities their own ECK rejects — each is a real cost of the "
              f"symbolic step:")
        for p, r, c in rejected_rows:
            print(f"  {p['mention_id']}  {p['entity_type']:<13} {r['label'][:26]:<28}"
                  f" gold {c['wikidata_id']:<11} P31={c['p31']}")
            if r["notes"].strip():
                print(f"       note: {r['notes'][:88]}")


if __name__ == "__main__":
    sys.exit(main())
