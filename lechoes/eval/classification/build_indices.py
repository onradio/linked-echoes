#!/usr/bin/env python3
"""Stage 0 — build the tagged index grid the retrieval experiment sweeps (spec §3.2).

    python build_indices.py                     # the full grid, 4 vocabs x 2 x 2
    python build_indices.py --vocab MIMO PBCore # a subset, for a smoke test
    python build_indices.py --force             # rebuild what already exists

Every combination of `similarity` x `index_strategy` is built once per vocabulary
and written to `data/indices/<vocab>_<similarity>_<index_strategy>.index`, with
its metadata and a build manifest beside it. The tag is in the filename because
the four builds of a vocabulary are not interchangeable and nothing downstream
should have to infer which is which.

Production indices under `vocabularies/<name>/vectordb/` are never touched: the
evaluation must be free to rebuild at will without disturbing the pipeline.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"           # bmeapi/src
sys.path.insert(0, str(SRC))               # import bare, as the eval specs require

from classification import (  # noqa: E402
    INDEX_STRATEGIES,
    SIMILARITIES,
    build_vector_database,
    manifest_path_for,
)

DEFAULT_VOCAB_CSV = SRC / "vocabularies" / "all_concepts.csv"
DEFAULT_OUTPUT_DIR = HERE / "data" / "indices"

# `source_vocabulary` values in all_concepts.csv, keyed by the vocabulary name
# used in VOCAB_MAP and in the annotation sheet's `vocab_hint`.
VOCAB_SUBSETS = {
    "AAT": ["AAT"],
    "MIMO": ["MIMO"],
    "Europeana": ["Europeana"],
    "PBCore": ["PBCore-AssetType", "PBCore-Instantiation"],
}


def index_stem(vocab: str, similarity: str, index_strategy: str) -> str:
    """The tagged basename of one build: <vocab>_<similarity>_<index_strategy>."""
    return f"{vocab.lower()}_{similarity}_{index_strategy}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vocabulary", default=str(DEFAULT_VOCAB_CSV),
                    help="Harmonised vocabulary CSV (default: src/vocabularies/all_concepts.csv)")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                    help="Where the tagged builds are written (default: data/indices)")
    ap.add_argument("--vocab", nargs="+", choices=list(VOCAB_SUBSETS),
                    default=list(VOCAB_SUBSETS), help="Vocabularies to build")
    ap.add_argument("--similarity", nargs="+", choices=list(SIMILARITIES),
                    default=list(SIMILARITIES), help="Similarity metrics to build")
    ap.add_argument("--index-strategy", nargs="+", choices=list(INDEX_STRATEGIES),
                    default=list(INDEX_STRATEGIES), help="Index strategies to build")
    ap.add_argument("--model-name", default="all-MiniLM-L6-v2",
                    help="Sentence-transformer shared by every build")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild indices that already exist")
    args = ap.parse_args()

    vocab_df = pd.read_csv(args.vocabulary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    planned = [(v, s, i) for v in args.vocab
               for s in args.similarity
               for i in args.index_strategy]
    print(f"{len(planned)} builds planned into {output_dir}\n")

    built, skipped, failed = [], [], []
    for vocab, similarity, strategy in planned:
        stem = index_stem(vocab, similarity, strategy)
        index_path = output_dir / f"{stem}.index"
        if index_path.exists() and not args.force:
            print(f"= {stem}: exists, skipping (use --force to rebuild)")
            skipped.append(stem)
            continue

        print(f"\n--- Building {stem} ---")
        try:
            ok = build_vector_database(
                vocab_df,
                model_name=args.model_name,
                index_strategy=strategy,
                similarity=similarity,
                output_dir=str(output_dir),
                vocabulary_name=stem,
                subset=VOCAB_SUBSETS[vocab],
            )
        except Exception as exc:  # a failed build must not abort the rest
            print(f"! {stem}: {type(exc).__name__}: {exc}")
            failed.append(stem)
            continue
        (built if ok else failed).append(stem)

    print(f"\n{'-' * 60}")
    print(f"built: {len(built)}   skipped: {len(skipped)}   failed: {len(failed)}")
    for stem in built:
        manifest = manifest_path_for(str(output_dir / f'{stem}.index'))
        print(f"  {stem}  ->  {Path(manifest).name}")
    if failed:
        print(f"failed builds: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
