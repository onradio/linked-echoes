#!/usr/bin/env python3
"""Experiment B — type retrieval over the full grid (spec §3.2). No LLM, no API key.

    python run_retrieval.py                       # the full grid
    python run_retrieval.py --limit 5             # smoke test over 5 objects
    python run_retrieval.py --similarity cosine_normalised --query hint_description
    python run_retrieval.py --force               # overwrite existing predictions

Grid: 2 similarities x 2 index strategies x 3 query formulations = 12 configurations,
each retrieved **once at k_max** and written out at every k in the grid, so a smaller
k is always a slice of the same pool rather than a fresh query.

Two pools are recorded per object, because the deployed system does not search one
flat index:

  `retrieved`           the vocabulary the gold primary category routes to
  `retrieved_fallback`  Getty AAT, which the fallback searches when the routed
                        vocabulary yields no usable term (null when already AAT)

`hit` is therefore reported twice — against the routed pool alone, and against the
two pools together (`hit_with_fallback`). Six objects have an AAT term in their
accepted set under a non-AAT routing, three of them with no other gold at all, so
collapsing the two would either flatter or penalise the retriever depending on
which single pool were chosen. Recording both leaves that decision to `report.py`.
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
sys.path.insert(0, str(SRC))      # bare import: bmeapi/__init__ needs the serving deps
sys.path.insert(0, str(HERE))

from classification import RAGObjectClassifier, SIMILARITY_COSINE  # noqa: E402
from llm import API_KEY_ENV                                        # noqa: E402
from gold import load_gold, vocabulary_of                          # noqa: E402
from build_indices import index_stem                               # noqa: E402


def build_query(obj: dict, formulation: str) -> tuple[str, bool]:
    """
    Renders one of the three query formulations of §3.2, verbatim.

    Returns the query and whether the hint fallback was applied — OBJ-024 has no
    type hint, so `hint_description` degrades to `description` there and that
    object must be excluded from any claim about the hint.
    """
    name, description = obj["name"], obj["description"]
    hint = obj["raw_type_hint"]
    if formulation == "description":
        return description, False
    if formulation == "name_description":
        return f"{name}. {description}", False
    if formulation == "hint_description":
        if not hint:
            return description, True
        return f"{hint}: {description}", False
    raise ValueError(f"unknown query formulation: {formulation}")


def rank_of_gold(pool: list, gold_uris: list) -> int | None:
    """1-based rank of the best-ranked accepted term, or None if absent."""
    ranks = [i for i, c in enumerate(pool, 1) if c["uri"] in gold_uris]
    return min(ranks) if ranks else None


@contextlib.contextmanager
def no_llm_key():
    """
    Hides every provider API key for the duration.

    `RAGObjectClassifier` falls back to the environment for its key, so on a
    developer machine it would build a live client even in retrieval-only mode.
    Experiment B must not be able to make an API call by accident, so the keys
    are removed rather than merely unused — every provider's, since hiding only
    Google's would still leave a Mistral spec able to reach the network.
    """
    stashed = {env: os.environ.pop(env, None) for env in API_KEY_ENV.values()}
    try:
        yield
    finally:
        for env, value in stashed.items():
            if value is not None:
                os.environ[env] = value


class IndexSet:
    """The four vocabulary indices for one (similarity, index_strategy) cell."""

    def __init__(self, indices_dir: Path, similarity: str, strategy: str,
                 embedding_model: str):
        self.similarity = similarity
        self.strategy = strategy
        self.classifiers = {}
        for vocab in ("AAT", "MIMO", "Europeana", "PBCore"):
            stem = index_stem(vocab, similarity, strategy)
            # The loader is chatty and says the same five things per index; its
            # failures raise rather than print, so only the notes are lost.
            with no_llm_key(), contextlib.redirect_stdout(io.StringIO()):
                self.classifiers[vocab] = RAGObjectClassifier(
                    index_path=str(indices_dir / f"{stem}.index"),
                    metadata_path=str(indices_dir / f"{stem}_metadata.pkl"),
                    embedding_model=embedding_model,
                    require_generator=False,   # Experiment B never calls an LLM
                )
            assert self.classifiers[vocab].generator is None, \
                "retrieval-only classifier must not hold an LLM client"
        sizes = ", ".join(f"{v}={c.index.ntotal}"
                          for v, c in self.classifiers.items())
        print(f"    indices loaded ({sizes})")

    def search(self, vocab: str, query: str, k: int) -> list:
        scored = self.classifiers[vocab].retrieve(query, top_k=k, return_scores=True)
        return [{"rank": i, "uri": c["uri"], "term": c["term"], "score": round(s, 6)}
                for i, (c, s) in enumerate(scored, 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--similarity", nargs="+", help="Subset of the similarity grid")
    ap.add_argument("--index-strategy", nargs="+", help="Subset of the index-strategy grid")
    ap.add_argument("--query", nargs="+", help="Subset of the query formulations")
    ap.add_argument("--limit", type=int, help="Only the first N objects (smoke test)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing predictions")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    r_cfg = cfg["retrieval"]
    similarities = args.similarity or r_cfg["similarity"]
    strategies = args.index_strategy or r_cfg["index_strategies"]
    queries = args.query or r_cfg["queries"]
    k_grid = sorted(r_cfg["k_grid"])
    k_max = max(r_cfg["k_max"], max(k_grid))
    embedding_model = r_cfg["embedding_model"]

    indices_dir = HERE / cfg["paths"]["indices"]
    out_dir = HERE / cfg["paths"]["predictions"]
    out_dir.mkdir(parents=True, exist_ok=True)

    objects = load_gold(HERE / cfg["paths"]["annotations"])
    if args.limit:
        objects = objects[:args.limit]

    n_cells = len(similarities) * len(strategies) * len(queries)
    print(f"{len(objects)} objects x {n_cells} configurations, "
          f"retrieved at k_max={k_max} and written at k in {k_grid}\n")

    written, skipped, summary = [], [], []
    for similarity in similarities:
        for strategy in strategies:
            targets = {(similarity, strategy, q, k)
                       for q in queries for k in k_grid}
            existing = {t for t in targets
                        if (out_dir / f"retrieval_{t[0]}_{t[1]}_{t[2]}_k{t[3]}.jsonl").exists()}
            if existing == targets and not args.force:
                print(f"= {similarity}/{strategy}: all outputs exist, skipping")
                skipped.extend(sorted(existing))
                continue

            print(f"--- loading indices: {similarity} / {strategy} ---")
            index_set = IndexSet(indices_dir, similarity, strategy, embedding_model)

            for query_id in queries:
                records = []
                for obj in objects:
                    query, hint_fallback = build_query(obj, query_id)
                    routed = obj["routed_vocab"]
                    pool = index_set.search(routed, query, k_max)
                    fallback_pool = (index_set.search("AAT", query, k_max)
                                     if routed != "AAT" else None)

                    records.append({
                        "object_id": obj["object_id"],
                        "case_number": obj["case_number"],
                        "similarity": similarity,
                        "index_strategy": strategy,
                        "query": query_id,
                        "query_text": query,
                        "hint_fallback": hint_fallback,
                        "vocab": routed,
                        "tier": obj["tier"],
                        "gold_uris": obj["gold_uris"],
                        "gold_reachable": obj["reachable"],
                        "pool": pool,
                        "fallback_pool": fallback_pool,
                    })

                for k in k_grid:
                    path = out_dir / f"retrieval_{similarity}_{strategy}_{query_id}_k{k}.jsonl"
                    if path.exists() and not args.force:
                        skipped.append(path.name)
                        continue
                    hits = considered = 0
                    with open(path, "w", encoding="utf-8") as fh:
                        for rec in records:
                            pool = rec["pool"][:k]
                            fpool = rec["fallback_pool"][:k] if rec["fallback_pool"] else None
                            gold = rec["gold_uris"]
                            rank = rank_of_gold(pool, gold) if gold else None
                            frank = rank_of_gold(fpool, gold) if (gold and fpool) else None
                            best = min([r for r in (rank, frank) if r], default=None)

                            fh.write(json.dumps({
                                "object_id": rec["object_id"],
                                "case_number": rec["case_number"],
                                "similarity": similarity,
                                "index_strategy": strategy,
                                "query": query_id,
                                "query_text": rec["query_text"],
                                "hint_fallback": rec["hint_fallback"],
                                "k": k,
                                "vocab": rec["vocab"],
                                "tier": rec["tier"],
                                "retrieved": pool,
                                "retrieved_fallback": fpool,
                                "gold_uris": gold,
                                # A NIL row has nothing to retrieve: hit is null
                                # rather than false, so it is excluded from hit@k
                                # instead of counted as a miss (§3.2).
                                "gold_reachable": rec["gold_reachable"],
                                "hit": None if not gold else rank is not None,
                                "gold_rank": rank,
                                "hit_with_fallback": None if not gold else best is not None,
                                "gold_rank_with_fallback": best,
                            }, ensure_ascii=False) + "\n")

                            if gold:
                                considered += 1
                                hits += 1 if best is not None else 0
                    written.append(path.name)
                    summary.append((similarity, strategy, query_id, k,
                                    hits, considered))
                print(f"  {query_id}: {len(records)} objects -> "
                      f"{len(k_grid)} files")

    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_objects": len(objects),
        "similarities": similarities,
        "index_strategies": strategies,
        "queries": queries,
        "k_grid": k_grid,
        "k_max": k_max,
        "embedding_model": embedding_model,
        "note": "files at k < k_max are slices of the same retrieval, not re-queries",
        "files_written": len(written),
        "files_skipped": len(skipped),
    }
    (out_dir / "retrieval_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{'-' * 66}")
    print(f"{len(written)} files written, {len(skipped)} skipped -> {out_dir}")
    if summary:
        # Run feedback only. score.py computes the reported numbers, with the
        # clustered CIs and the routed-vs-fallback distinction this table elides.
        cells = {}
        for sim, strat, q, k, hits, n in summary:
            cells.setdefault((sim, strat, q), {})[k] = (hits, n)
        n_scored = max(n for *_, n in summary)
        print(f"\nhit@k with fallback, {n_scored} objects with a reachable gold "
              f"(indicative; score.py is authoritative)")
        header = "".join(f"{'@' + str(k):>9}" for k in k_grid)
        print(f"  {'similarity':<20}{'index':<10}{'query':<20}{header}")
        for (sim, strat, q), by_k in cells.items():
            row = "".join(f"{100 * by_k[k][0] / by_k[k][1]:>8.1f}%"
                          if k in by_k else f"{'-':>9}" for k in k_grid)
            print(f"  {sim:<20}{strat:<10}{q:<20}{row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
