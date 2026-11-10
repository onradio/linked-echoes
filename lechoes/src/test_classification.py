"""
Test suite for the retrieval and prompting behaviour of ``classification.py``.

Run it as a script — it needs neither an LLM key nor the network::

    python -m lechoes.src.test_classification
    python -m lechoes.src.test_classification --keep    # keep the scratch indices

Everything here is built from a tiny synthetic vocabulary in a temporary
directory, so the suite is independent of which production indices happen to
exist on the machine.

The cases correspond to the defects listed in §5 of
``eval/classification/classification_evaluation_spec.md``. Each one is a silent
corruption rather than a crash — a phantom candidate, a meaningless ranking, the
string 'nan' in a prompt, a category presented as a predicted type — so they are
exactly the failures a run would carry all the way into published numbers
without anything looking wrong.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:  # package import and bare import (notebooks, eval scripts) both work
    from .classification import (
        SIMILARITY_COSINE,
        SIMILARITY_L2,
        CATEGORY_DEFINITIONS,
        EmbeddingObjectCategoriser,
        RAGObjectClassifier,
        build_hint_block,
        build_vector_database,
        load_build_manifest,
    )
except ImportError:  # pragma: no cover
    from classification import (
        SIMILARITY_COSINE,
        SIMILARITY_L2,
        CATEGORY_DEFINITIONS,
        EmbeddingObjectCategoriser,
        RAGObjectClassifier,
        build_hint_block,
        build_vector_database,
        load_build_manifest,
    )

# A vocabulary small enough that k > ntotal is reachable with k=10, which is the
# condition that used to inject the last row of the vocabulary as a candidate.
FIXTURE = pd.DataFrame([
    {"term": "electric guitars", "definition": "Guitars amplified electrically.",
     "uri": "http://example.org/1", "source_vocabulary": "TEST"},
    {"term": "kimonos", "definition": "Loose, wide-sleeved Japanese garments.",
     "uri": "http://example.org/2", "source_vocabulary": "TEST"},
    {"term": "concert posters", "definition": np.nan,   # deliberately missing
     "uri": "http://example.org/3", "source_vocabulary": "TEST"},
    {"term": "vinyl records", "definition": "Analogue sound carriers.",
     "uri": "http://example.org/4", "source_vocabulary": "TEST"},
    {"term": "trophies", "definition": "Objects awarded for achievement.",
     "uri": "http://example.org/5", "source_vocabulary": "TEST"},
])

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append((name, detail))
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def build_fixture_indices(workdir: Path) -> dict:
    """Builds the synthetic vocabulary under both similarity metrics."""
    paths = {}
    for similarity in (SIMILARITY_L2, SIMILARITY_COSINE):
        name = f"test_{similarity}"
        build_vector_database(
            FIXTURE.copy(),
            model_name="all-MiniLM-L6-v2",
            index_strategy="combine",
            similarity=similarity,
            output_dir=str(workdir),
            vocabulary_name=name,
        )
        paths[similarity] = {
            "index": str(workdir / f"{name}.index"),
            "metadata": str(workdir / f"{name}_metadata.pkl"),
        }
    return paths


def load(paths: dict, built_as: str, **kwargs) -> RAGObjectClassifier:
    """Loads the fixture index built under `built_as`, in retrieval-only mode."""
    return RAGObjectClassifier(
        index_path=paths[built_as]["index"],
        metadata_path=paths[built_as]["metadata"],
        require_generator=False,
        **kwargs,
    )


def test_short_index_padding(paths: dict) -> None:
    """Defect 1 — FAISS pads with -1 when k exceeds the number of vectors."""
    print("\n[1] k > ntotal returns no phantom candidates")
    clf = load(paths, SIMILARITY_L2)
    candidates = clf.retrieve("a guitar", top_k=10)
    check("returns ntotal candidates, not k",
          len(candidates) == len(FIXTURE),
          f"got {len(candidates)}, expected {len(FIXTURE)}")
    terms = [c["term"] for c in candidates]
    check("no duplicated candidates",
          len(terms) == len(set(terms)), f"got {terms}")

    scored = clf.retrieve("a guitar", top_k=10, return_scores=True)
    check("scores are returned alongside candidates",
          len(scored) == len(FIXTURE) and all(
              isinstance(score, float) for _, score in scored))


def test_similarity_round_trip(paths: dict) -> None:
    """Defect 2 — the query must be encoded the way the index was built."""
    print("\n[2] similarity is recovered from the index and applied to queries")
    l2 = load(paths, SIMILARITY_L2)
    cosine = load(paths, SIMILARITY_COSINE)
    check("L2 index reports l2_unnormalised", l2.similarity == SIMILARITY_L2,
          l2.similarity)
    check("IP index reports cosine_normalised", cosine.similarity == SIMILARITY_COSINE,
          cosine.similarity)

    manifest = load_build_manifest(paths[SIMILARITY_COSINE]["index"])
    check("build manifest records the similarity",
          manifest is not None and manifest["similarity"] == SIMILARITY_COSINE,
          str(manifest))
    check("build manifest records the FAISS index class",
          manifest is not None and manifest["faiss_index"] == "IndexFlatIP",
          str(manifest))

    # Unit-norm queries: an inner product in [-1, 1] is cosine, anything larger
    # means the query was not normalised to match the index.
    _, top_score = cosine.retrieve("an amplified guitar", top_k=1, return_scores=True)[0]
    check("cosine scores lie in [-1, 1]", -1.0001 <= top_score <= 1.0001,
          f"got {top_score}")
    check("cosine retrieval is sane",
          cosine.retrieve("an amplified guitar", top_k=1)[0]["term"] == "electric guitars")
    check("l2 retrieval is sane",
          l2.retrieve("an amplified guitar", top_k=1)[0]["term"] == "electric guitars")

    # Scores must be ordered the way the metric implies: L2 distances ascend,
    # inner products descend. A mismatched query encoding breaks this first.
    l2_scores = [s for _, s in l2.retrieve("an amplified guitar", top_k=5, return_scores=True)]
    ip_scores = [s for _, s in cosine.retrieve("an amplified guitar", top_k=5, return_scores=True)]
    check("l2 scores ascend", all(a <= b for a, b in zip(l2_scores, l2_scores[1:])),
          str(l2_scores))
    check("inner products descend", all(a >= b for a, b in zip(ip_scores, ip_scores[1:])),
          str(ip_scores))


def test_similarity_conflict(paths: dict) -> None:
    """A declared metric that contradicts the index must fail loudly."""
    print("\n[3] a contradictory similarity argument is rejected")
    try:
        load(paths, SIMILARITY_COSINE, similarity=SIMILARITY_L2)
        check("mismatched similarity raises", False, "no exception raised")
    except ValueError as e:
        check("mismatched similarity raises", "contradicts" in str(e), str(e))


def test_nan_definitions(paths: dict) -> None:
    """Defect 3 — a missing definition must not reach the prompt as 'nan'."""
    print("\n[4] missing definitions never render as 'nan'")
    clf = load(paths, SIMILARITY_L2)
    candidates = clf.retrieve("a poster for a concert", top_k=5)
    poster = next(c for c in candidates if c["term"] == "concert posters")
    check("missing definition is an empty string", poster["definition"] == "",
          repr(poster["definition"]))
    prompt = clf.generate_prompt("A concert poster", candidates)
    check("prompt contains no 'nan' definition", "Definition: nan" not in prompt)


def test_hint_fields(paths: dict) -> None:
    """Defect 4 — a category must never be presented as a predicted type."""
    print("\n[5] type hint and category are distinct prompt fields")
    clf = load(paths, SIMILARITY_L2)
    candidates = clf.retrieve("a concert poster", top_k=3)

    with_type = clf.generate_prompt("A poster", candidates, predicted_type="posters")
    check("type hint renders as PREDICTED TYPE",
          "PREDICTED TYPE: posters" in with_type)
    check("type hint does not render a category field",
          "OBJECT CATEGORY:" not in with_type)

    with_category = clf.generate_prompt("A poster", candidates,
                                        object_category="Merchandise")
    check("category renders as OBJECT CATEGORY",
          "OBJECT CATEGORY: Merchandise" in with_category)
    check("category is not labelled as a predicted type",
          "PREDICTED TYPE: Merchandise" not in with_category)

    neither = clf.generate_prompt("A poster", candidates)
    check("no hint renders neither field",
          "PREDICTED TYPE:" not in neither and "OBJECT CATEGORY:" not in neither)
    check("no-hint prompt still carries the description and rule",
          "ITEM DESCRIPTION:" in neither and "STRICT RULE:" in neither)

    both = build_hint_block("posters", "Merchandise")
    check("type hint takes precedence over category", both[1] == "type_hint")
    check("hint source is reported for a category-only call",
          build_hint_block("", "Merchandise")[1] == "category")
    check("hint source is reported when there is no hint",
          build_hint_block("", "")[1] == "none")


def test_retrieval_query_override(paths: dict) -> None:
    """Experiment B varies the query formulation independently of the hint."""
    print("\n[6] the retrieval query is composable and overridable")
    clf = load(paths, SIMILARITY_L2)
    check("default query is the description alone",
          clf.build_retrieval_query("A red guitar") == "A red guitar")
    check("a type hint is prepended",
          clf.build_retrieval_query("A red guitar", "electric guitars")
          == "electric guitars: A red guitar")


def test_index_metadata_mismatch(paths: dict, workdir: Path) -> None:
    """Misaligned metadata makes every candidate lookup wrong; fail at load."""
    print("\n[7] an index/metadata mismatch is caught at load time")
    truncated = workdir / "truncated_metadata.pkl"
    FIXTURE.iloc[:3].to_pickle(truncated)
    try:
        RAGObjectClassifier(
            index_path=paths[SIMILARITY_L2]["index"],
            metadata_path=str(truncated),
            require_generator=False,
        )
        check("mismatch raises", False, "no exception raised")
    except ValueError as e:
        check("mismatch raises", "mismatch" in str(e).lower(), str(e))


def test_retrieval_without_key(paths: dict) -> None:
    """Experiment B runs with no LLM key at all."""
    print("\n[8] retrieval works without an LLM key, selection refuses")
    # The constructor falls back to GEMINI_API_KEY, which is often set on a
    # developer machine; hide it so this exercises the keyless path.
    stashed = os.environ.pop("GEMINI_API_KEY", None)
    try:
        clf = load(paths, SIMILARITY_L2)
        check("constructed without a generator", clf.generator is None)
        check("retrieval still works", len(clf.retrieve("a trophy", top_k=2)) == 2)
        try:
            clf.classify_instrumented("a trophy", top_k=2)
            check("selection raises without a key", False, "no exception raised")
        except RuntimeError as e:
            check("selection raises without a key", "require_generator" in str(e), str(e))

        try:
            RAGObjectClassifier(
                index_path=paths[SIMILARITY_L2]["index"],
                metadata_path=paths[SIMILARITY_L2]["metadata"],
            )
            check("require_generator=True still fails fast", False,
                  "no exception raised")
        except ValueError as e:
            check("require_generator=True still fails fast", "API key" in str(e), str(e))
    finally:
        if stashed is not None:
            os.environ["GEMINI_API_KEY"] = stashed


def test_embedding_categoriser() -> None:
    """The C1 baseline: nearest category definition, no LLM."""
    print("\n[9] embedding categoriser (C1)")
    categoriser = EmbeddingObjectCategoriser()
    check("covers the seven categories",
          len(categoriser.categories) == len(CATEGORY_DEFINITIONS))
    result = categoriser.classify_instrumented(
        "A Fender Stratocaster electric guitar played on stage")
    check("returns a known category", result["predicted"] in CATEGORY_DEFINITIONS,
          result["predicted"])
    check("scores every category", len(result["scores"]) == len(CATEGORY_DEFINITIONS))
    check("ranking is ordered by score",
          result["ranking"][0] == result["predicted"])
    check("is deterministic",
          categoriser.classify("A vinyl LP of My Generation")
          == categoriser.classify("A vinyl LP of My Generation"))
    # Not asserted as correctness: C1 is a baseline, and what it gets right is
    # a result of the experiment, not a property the suite should pin down.
    print(f"        (guitar -> {result['predicted']}, "
          f"score {result['score']:.3f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--keep", action="store_true",
                    help="Keep the scratch index directory for inspection")
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="test_classification_"))
    print(f"Scratch indices in {workdir}\n")
    try:
        paths = build_fixture_indices(workdir)
        test_short_index_padding(paths)
        test_similarity_round_trip(paths)
        test_similarity_conflict(paths)
        test_nan_definitions(paths)
        test_hint_fields(paths)
        test_retrieval_query_override(paths)
        test_index_metadata_mismatch(paths, workdir)
        test_retrieval_without_key(paths)
        test_embedding_categoriser()
    finally:
        if args.keep:
            print(f"\nScratch directory kept: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for name, detail in FAILED:
        print(f"  FAILED: {name} — {detail}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
