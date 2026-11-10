import os
import enum
import json
import argparse
import logging
from datetime import datetime, timezone

import faiss
import pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer

try:  # package import (lechoes.src.classification) and bare import both work
    from .llm import SCHEMA_CATEGORY, SCHEMA_INT_LIST, build_client
except ImportError:  # pragma: no cover
    from llm import SCHEMA_CATEGORY, SCHEMA_INT_LIST, build_client

logger = logging.getLogger(__name__)

# --- Retrieval configuration ---------------------------------------------
# Two similarity metrics are supported, and the choice is a property of the
# index rather than of a query: an index built with one and searched with the
# other silently yields a third, meaningless metric. `RAGObjectClassifier`
# therefore recovers the metric from the index it loads (build manifest first,
# FAISS metric type as a fallback) and normalises queries to match.
SIMILARITY_L2 = "l2_unnormalised"       # raw embeddings + IndexFlatL2 (legacy behaviour)
SIMILARITY_COSINE = "cosine_normalised"  # unit-norm embeddings + IndexFlatIP
SIMILARITIES = (SIMILARITY_L2, SIMILARITY_COSINE)

INDEX_STRATEGIES = ("term", "combine")

class ObjectCategory(enum.Enum):
    MUSICAL_INSTRUMENT = "Musical Instrument"
    CLOTHING = "Clothing"
    AUDIOVISUAL_MEDIA = "Audiovisual Media"
    TEXTUAL_GRAPHIC_WORK = "Textual & Graphic Work"
    MERCHANDISE = "Merchandise"
    PERSONAL_EFFECTS = "Personal Effects"
    AWARD_RECOGNITION = "Award & Recognition"

VOCAB_MAP = {
    ObjectCategory.MUSICAL_INSTRUMENT.value: "MIMO",
    ObjectCategory.CLOTHING.value: "Europeana",
    ObjectCategory.AUDIOVISUAL_MEDIA.value: "PBCore",
    ObjectCategory.TEXTUAL_GRAPHIC_WORK.value: "AAT",
    ObjectCategory.MERCHANDISE.value: "AAT",
    ObjectCategory.PERSONAL_EFFECTS.value: "AAT",
    ObjectCategory.AWARD_RECOGNITION.value: "AAT"
}

VOCAB_PATHS = {
    "AAT": {
        "index": "vocabularies/aat/vectordb/aat.index",
        "metadata": "vocabularies/aat/vectordb/aat_metadata.pkl"
    },
    "MIMO": {
        "index": "vocabularies/mimo/vectordb/mimo.index",
        "metadata": "vocabularies/mimo/vectordb/mimo_metadata.pkl"
    },
    "Europeana": {
        "index": "vocabularies/fashion/vectordb/fashion.index",
        "metadata": "vocabularies/fashion/vectordb/fashion_metadata.pkl"
    },
    "PBCore": {
        "index": "vocabularies/pbcore/vectordb/pbcore.index",
        "metadata": "vocabularies/pbcore/vectordb/pbcore_metadata.pkl"
    }
}

CATEGORY_ROLE_SCOPED = """
You are a specialist AI assistant for museum cataloging, tasked with classifying
objects in a music museum according to a strict, pre-defined framework. Your
goal is to assign a single broad category to a museum object based on its
description. You must choose only one from the seven official categories listed
below. Your response must adhere strictly to the provided scope notes.

---
**OFFICIAL BROAD CATEGORIES AND DEFINITIONS**

1.  **Musical Instrument**: Objects created or adapted to make musical sounds.
Includes acoustic, electric, and electronic instruments, as well as their
essential functional components (e.g., bows, drum skins, mutes).

2.  **Clothing**: Items worn on the body, including garments, footwear, headwear,
and costume accessories. Includes both everyday wear and specialized stage costumes.

3.  **Audiovisual Media**: Physical carriers of recorded sound and/or moving
images. This category is for the physical object (e.g., vinyl record, cassette
tape, CD) itself, not the intellectual content.

4.  **Textual & Graphic Work**: Two-dimensional objects whose primary function
is to convey information or artistic expression through text, notation, or
imagery on a substrate (typically paper). This includes documents, posters,
sheet music, handwritten lyrics, magazines, books, tickets, programmes, and flyers.

5.  **Merchandise**: Commercially produced items, other than clothing or
audiovisual media, created for sale to fans, often featuring an artist's name,
image, or logo. This typically includes three-dimensional objects like dolls,
badges, lunchboxes, and keychains.

6.  **Personal Effects**: Objects personally owned, used, or modified by an
individual that are not classified as instruments, clothing, or documents.
Includes accessories (e.g., glasses), equipment cases, instrument accessories
(e.g., drumsticks, plectrums), and other unique personal items.

7.  **Award & Recognition**: Official items conferred to recognise achievement
or commemorate participation, such as a sales award, a trophy, or a
commemorative object.
---
**EXAMPLES OF CORRECT CLASSIFICATION**

* Input: "The Clash - 'Pretty Vacant' lyrics" -> Output: Textual & Graphic Work
* Input: "A handwritten set-list for the Ziggy Stardust tour" -> Output: Textual & Graphic Work
* Input: "Take That dolls x 5" -> Output: Merchandise
* Input: "Take That dolls" -> Output: Merchandise
* Input: "Acker Bilk - leather clarinet case" -> Output: Personal Effects
* Input: "Ziggy Stardust Kimono with Carp Design" -> Output: Clothing
* Input: "The Who - My Generation LP" -> Output: Audiovisual Media
* Input: "Commemorative watch" -> Output: Award & Recognition
---

Your final output for any given task must consist **ONLY** of the single, most
appropriate category name from the list. Do not add any additional text.
"""

# The deployed system instruction: the seven category names and nothing else.
#
# It replaced the scope-note-and-examples variant above after the evaluation
# measured them side by side on the 65-object gold standard (eval/classification,
# §3.1): 98.0% vs 100.0% lenient accuracy, one discordant object, McNemar p=1.0.
# The extra prompt material bought no measurable accuracy, so the simpler
# instruction is preferred. `CATEGORY_ROLE_SCOPED` is retained because the
# comparison is reported in the article and the released BME KG was built with
# it; see the caveat in the evaluation spec about the three categories the
# sample does not attest, which are the ones the scope notes were written to
# disambiguate.
CATEGORY_ROLE = """
You are a specialist AI assistant for museum cataloging. Assign a single broad
category to a museum object based on its description. You must choose only one
from the seven official categories listed below.

---
**OFFICIAL BROAD CATEGORIES**

1.  Musical Instrument
2.  Clothing
3.  Audiovisual Media
4.  Textual & Graphic Work
5.  Merchandise
6.  Personal Effects
7.  Award & Recognition
---

Your final output for any given task must consist **ONLY** of the single, most
appropriate category name from the list. Do not add any additional text.
"""

CATEGORY_ROLES = {
    "minimal": CATEGORY_ROLE,        # deployed
    "scoped": CATEGORY_ROLE_SCOPED,  # the ablation: scope notes + few-shot
}

CATEGORY_PROMPT = """
Based on the item description below, choose the most appropriate broad category:
{0}
"""

CATEGORY_DEFINITIONS = {
    "Musical Instrument":   "Objects created or adapted to make musical sounds. Includes acoustic, electric, and electronic instruments, as well as their essential functional components (e.g., bows, drum skins, mutes).",
    "Clothing":             "Items worn on the body, including garments, footwear, headwear, and costume accessories. Includes both everyday wear and specialized stage costumes.",
    "Audiovisual Media":    "Physical carriers of recorded sound and/or moving images. This category is for the physical object (e.g., vinyl record, cassette tape, CD) itself, not the intellectual content.",
    "Textual & Graphic Work": "Two-dimensional objects whose primary function is to convey information or artistic expression through text, notation, or imagery on a substrate (typically paper). This includes documents, posters, sheet music, handwritten lyrics, magazines, books, tickets, programmes, and flyers.",
    "Merchandise":          "Commercially produced items, other than clothing or audiovisual media, created for sale to fans, often featuring an artist's name, image, or logo. This typically includes three-dimensional objects like dolls, badges, lunchboxes, and keychains.",
    "Personal Effects":     "Objects personally owned, used, or modified by an individual that are not classified as instruments, clothing, or documents. Includes accessories (e.g., glasses), equipment cases, instrument accessories (e.g., drumsticks, plectrums), and other unique personal items.",
    "Award & Recognition":  "Official items conferred to recognise achievement or commemorate participation, such as a sales award, a trophy, or a commemorative object.",
}

# The hint given to the selector is either an automatically predicted *type* or
# the broad *category* the object was routed by — never both, and they are not
# interchangeable. The previous revision interpolated whichever was available
# into a single "PREDICTED TYPE" field, so a category name (Merchandise) could
# be presented to the model as a proposed object type. They are separate blocks
# now, and `hint_source` records which one a call actually used.
TYPE_HINT_BLOCK = """
PREDICTED TYPE: {type_hint}
This is an indicative suggestion of what the object type might be, extracted
automatically from the item description. Use it as a hint to guide your selection,
but do not treat it as definitive — always base your final answer on the full
item description and the vocabulary list.
"""

CATEGORY_HINT_BLOCK = """
OBJECT CATEGORY: {category}
This is the broad category the object has already been assigned. Use it to judge
which kinds of term are plausible, but do not treat it as a proposed type —
always base your final answer on the full item description and the vocabulary list.
"""

SINGLE_CLASS_PROMPT = """
You are an expert museum cataloguer. Based on the item description below, choose
the single most appropriate specific type from the 'Controlled Vocabulary List'.
Your response must be ONLY the number of the best matching term.
{hint_block}
STRICT RULE: If no term is a genuinely accurate match for the object described,
you MUST respond with 0. Do not select the closest or most similar term —
only select a term if it is a correct and precise match. When in doubt, respond with 0.

ITEM DESCRIPTION:
{description}

CONTROLLED VOCABULARY LIST:
"""

MULTI_CLASS_PROMPT = """
You are an expert museum cataloguer. Based on the item description below, choose
the most appropriate specific type from the 'Controlled Vocabulary List'. If the
item is a combination of multiple objects, select ALL relevant types. Your
response must only include the list of numbers of the best matching terms.
{hint_block}
STRICT RULE: If no term is a genuinely accurate match for the object described,
you MUST respond with 0. Do not select the closest or most similar term —
only select a term if it is a correct and precise match. When in doubt, respond with 0.

ITEM DESCRIPTION:
{description}

CONTROLLED VOCABULARY LIST:
"""


def build_hint_block(predicted_type: str = "", object_category: str = "") -> tuple[str, str]:
    """
    Renders the hint paragraph of a selection prompt.

    A predicted type takes precedence over the category: it is the more specific
    signal, and the two are never shown together.

    Returns:
        tuple[str, str]: the rendered block (possibly empty) and the hint source,
            one of 'type_hint', 'category' or 'none'.
    """
    if predicted_type:
        return TYPE_HINT_BLOCK.format(type_hint=predicted_type), "type_hint"
    if object_category:
        return CATEGORY_HINT_BLOCK.format(category=object_category), "category"
    return "", "none"


def manifest_path_for(index_path: str) -> str:
    """Path of the build manifest that accompanies a FAISS index file."""
    base = index_path[:-len('.index')] if index_path.endswith('.index') else index_path
    return f"{base}_build.json"


def load_build_manifest(index_path: str) -> dict | None:
    """
    Reads the build manifest written alongside an index, if it exists.

    Indices built before manifests were introduced simply have none; callers
    fall back to inspecting the FAISS metric type.
    """
    path = manifest_path_for(index_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"[classification] Could not read build manifest {path}: {e}")
        return None


def build_vector_database(vocab_df: pd.DataFrame,
                          model_name: str = 'all-MiniLM-L6-v2',
                          index_strategy: str = 'combine',
                          similarity: str = SIMILARITY_L2,
                          output_dir: str = 'vocabularies/vectordb',
                          vocabulary_name: str = 'vocabulary',
                          subset: list = None) -> bool:
    """
    Builds a FAISS vector database from a list of vocabulary CSV files.

    Args:
        vocab_df (pd.DataFrame): A DataFrame containing vocabulary terms,
            definitions, and URIs.
        model_name (str): The name of the sentence-transformer model to use.
        index_strategy (str): 'term' to use only the term, 'combine' to use
            a concatenation of term + definition.
        similarity (str): 'l2_unnormalised' embeds without normalisation and
            indexes with IndexFlatL2; 'cosine_normalised' unit-normalises the
            embeddings and indexes with IndexFlatIP, so that inner product is
            cosine similarity. Queries must be encoded the same way, which
            `RAGObjectClassifier` enforces from the build manifest.
        output_dir (str): Directory to save the FAISS index and metadata.
        vocabulary_name (str): Name to use for the saved index file.
        subset (list): A list of vocabulary prefixes to filter by.

    Returns:
        bool: True if successful, False otherwise.
    """
    if similarity not in SIMILARITIES:
        raise ValueError(f"similarity must be one of {SIMILARITIES}")

    # Filter the DataFrame based on the provided subset prefixes
    if subset:
        print(f"Filtering vocabulary to include only: {', '.join(subset)}")
        # Create a boolean mask where True indicates a row to keep
        filter_mask = vocab_df['source_vocabulary'].isin(subset)
        vocab_df = vocab_df[filter_mask].copy()

    # Check if the filtered DataFrame is empty
    if vocab_df.empty:
        print("Warning: The filtered DataFrame is empty. No index is built.")
        return False

    if vocab_df['definition'].isna().sum() > 0:
        print(f"Found {vocab_df['definition'].isna().sum()} NaN definitions.")
        vocab_df['definition'] = vocab_df['definition'].fillna('')

    if index_strategy == 'term':
        texts_to_embed = vocab_df['term'].tolist()
    elif index_strategy == 'combine':
        texts_to_embed = (vocab_df['term'] + ": " + vocab_df['definition']).tolist()
    else:  # invalid index embedding strategy
        raise ValueError("index_strategy must be 'term' or 'combine'")

    print(f"Prepared {len(texts_to_embed)} entries to embed")
    print(f"Index construction strategy: {index_strategy}, similarity: {similarity}.")

    normalise = similarity == SIMILARITY_COSINE

    print(f"\n--- Generating Embeddings with '{model_name}' ---")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts_to_embed, show_progress_bar=True,
                              normalize_embeddings=normalise)
    # Ensure embeddings are float32, as required by FAISS
    embeddings = np.array(embeddings).astype('float32')
    print(f"Generated {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}.")

    print("\n--- Building and Storing the FAISS Index ---")
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension) if normalise \
        else faiss.IndexFlatL2(embedding_dimension)
    index.add(embeddings)
    print(f"FAISS index built. Total entries in index: {index.ntotal}")

    os.makedirs(output_dir, exist_ok=True)
    # Save the FAISS index and the metadata using the vocabulary name
    index_path = f"{output_dir}/{vocabulary_name}.index"
    faiss.write_index(index, index_path)
    vocab_df.to_pickle(f"{output_dir}/{vocabulary_name}_metadata.pkl")

    # The manifest travels with the index so that retrieval can reproduce the
    # build-time encoding rather than guess at it.
    manifest = {
        "vocabulary_name": vocabulary_name,
        "model_name": model_name,
        "index_strategy": index_strategy,
        "similarity": similarity,
        "normalise_embeddings": normalise,
        "faiss_index": "IndexFlatIP" if normalise else "IndexFlatL2",
        "embedding_dimension": int(embedding_dimension),
        "n_terms": int(index.ntotal),
        "subset": list(subset) if subset else None,
        "built_at": datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    with open(manifest_path_for(index_path), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nVector DB saved in '{output_dir}'.")

    return True



def build_bm25_index(vocab_df: pd.DataFrame,
                     output_dir: str = 'vocabularies/vectordb',
                     file_name: str = 'bm25_index.pkl') -> bool:
    """
    Builds a BM25 index from a vocabulary DataFrame and saves it to a file.

    The function combines the 'term' and 'definition' columns to create a corpus
    for indexing. It handles potential missing values and stores the entire
    BM25 index object to disk using the pickle library.

    Args:
        vocab_df (pd.DataFrame): DataFrame containing vocabulary data. Expected
                                 to have 'term' and 'definition' columns.
        output_dir (str): Directory to save the pickled index file.
        file_name (str): The name for the saved pickle file.

    Returns:
        bool: True if the index was built and saved successfully.
    """
    # Check if the required columns exist in the DataFrame
    if 'term' not in vocab_df.columns or 'definition' not in vocab_df.columns:
        raise ValueError("DataFrame must contain 'term' and 'definition' columns.")
    # Fill any potential NaN values with empty strings to avoid errors
    print("Pre-processing vocabulary data for BM25 indexing...")
    vocab_df['term'] = vocab_df['term'].fillna('')
    # Combine the 'term' and 'definition' into a single corpus string
    # for more comprehensive keyword searching.
    corpus = (vocab_df['term'] + ' ' + vocab_df['definition']).tolist()

    # Tokenize the corpus. BM25 requires a list of tokenized documents.
    # The simplest way is to split by space, but we should ideally
    # consider more advanced tokenizers (e.g., from NLTK or SpaCy)
    # to handle punctuation, stemming, and stopwords.
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    print(f"Tokenized {len(tokenized_corpus)} documents.")

    # Build the BM25 index from the tokenized corpus.
    print("Building BM25 index...")
    bm25_index = BM25Okapi(tokenized_corpus)
    print("BM25 index built successfully.")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, file_name)
    # Use pickle to serialize and save the entire BM25 object to disk.
    print(f"Dumping BM25 index to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 index successfully saved.")

    return True


class RAGObjectClassifier:
    def __init__(self,
                 index_path: str,
                 metadata_path: str,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 generator_model: str = 'gemini-2.5-flash',
                 generator_api_key: str = '',
                 similarity: str = None,
                 temperature: float = 0.0,
                 require_generator: bool = True,
                 generator: object = None):
        """
        Initialises the RAGObjectClassifier with a FAISS index and metadata.
        Remember to export your API key for LLM access — `GEMINI_API_KEY` or
        `MISTRAL_API_KEY`, depending on the provider named by `generator_model`
        — otherwise the key should be provided at construction time.

        Args:
            index_path (str): Path to the FAISS index file.
            metadata_path (str): Path to the metadata pickle file.
            embedding_model (str): The name of the sentence-transformer model to use.
            generator_model (str): The name of the LLM model to use for generation.
            generator_api_key (str): API key for the LLM service. If empty, it
                                     is read from the environment variable of
                                     the provider named by `generator_model`.
            similarity (str): Overrides the similarity metric recovered from the
                index. Only needed for indices built before build manifests
                existed; a value that contradicts the index is an error.
            temperature (float): Sampling temperature for the selector LLM.
                Defaults to 0.0 so that repeated runs are comparable.
            require_generator (bool): When False, the classifier is usable
                without an LLM key for retrieval-only work; `classify` then
                raises rather than failing at construction time.
        """
        print(f"\n--- Loading FAISS Index from '{index_path}' ---")
        self.index = faiss.read_index(index_path)
        print(f"Index loaded. Total entries in index: {self.index.ntotal}")

        self.build_manifest = load_build_manifest(index_path)
        self.similarity = self._resolve_similarity(similarity)
        print(f"Similarity metric: {self.similarity}")

        print(f"--- Loading Metadata from '{metadata_path}' ---")
        self.metadata_df = pd.read_pickle(metadata_path)
        # A missing definition would otherwise reach the prompt as the string
        # 'nan' — the build fills these at index time, so the metadata must
        # agree with it at load time.
        if 'definition' in self.metadata_df.columns:
            self.metadata_df['definition'] = self.metadata_df['definition'].fillna('')
        print(f"Metadata loaded. Total entries in metadata: {len(self.metadata_df)}")

        if len(self.metadata_df) != self.index.ntotal:
            raise ValueError(
                f"Index/metadata mismatch: {self.index.ntotal} vectors in "
                f"'{index_path}' but {len(self.metadata_df)} rows in "
                f"'{metadata_path}'. Candidate lookups would be misaligned."
            )

        print(f"--- Loading SentenceTransformer Model '{embedding_model}' ---")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("Model loaded successfully.")

        self.generator_model = generator_model
        self.temperature = temperature
        # `generator` is an llm.LLMClient (or a caching wrapper around one), so
        # the selector is model-agnostic: 'gemini-2.5-flash' means Google for
        # backwards compatibility, 'mistral:ministral-14b-2512' names another
        # provider. A caller may also inject a ready-made client.
        if generator is not None:
            self.generator = generator
            self.generator_model = getattr(generator, "spec", generator_model)
        else:
            try:
                self.generator = build_client(
                    generator_model, api_key=generator_api_key,
                    temperature=temperature)
            except ValueError:
                if require_generator:
                    raise
                self.generator = None
        if self.generator is None:
            print("No LLM key provided: retrieval-only mode.")
        else:
            print(f"LLM client initialised: {self.generator.spec}")

    def _resolve_similarity(self, similarity: str = None) -> str:
        """
        Determines how queries must be encoded for the loaded index.

        The build manifest is authoritative; without one the FAISS metric type
        is a reliable proxy, since the two supported builds use different index
        classes. An explicit argument that contradicts either is rejected: a
        cosine index searched with unnormalised queries returns a ranking that
        is neither cosine nor L2, and nothing downstream would reveal it.
        """
        detected = None
        if self.build_manifest and self.build_manifest.get("similarity") in SIMILARITIES:
            detected = self.build_manifest["similarity"]
        elif self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
            detected = SIMILARITY_COSINE
        elif self.index.metric_type == faiss.METRIC_L2:
            detected = SIMILARITY_L2

        if similarity is None:
            if detected is None:
                raise ValueError(
                    "Could not determine the similarity metric of the index; "
                    f"pass similarity=<one of {SIMILARITIES}> explicitly."
                )
            return detected

        if similarity not in SIMILARITIES:
            raise ValueError(f"similarity must be one of {SIMILARITIES}")
        if detected is not None and similarity != detected:
            raise ValueError(
                f"similarity='{similarity}' contradicts the loaded index, which "
                f"was built as '{detected}'. Searching an index with the wrong "
                "metric silently produces a meaningless ranking."
            )
        return similarity

    def encode_query(self, text: str) -> np.ndarray:
        """Encodes a query with the same normalisation the index was built with."""
        return self.embedding_model.encode(
            [text],
            normalize_embeddings=self.similarity == SIMILARITY_COSINE,
        ).astype('float32')

    def retrieve(self,
                 description: str,
                 top_k: int = 10,
                 verbose: bool = False,
                 return_scores: bool = False) -> list:
        """
        Retrieves top_k candidate terms from the vocabulary vector database.

        Args:
            description (str): The description of the object to classify.
            top_k (int): The number of top candidates to retrieve.
            verbose (bool): Whether to print the retrieved candidates.
            return_scores (bool): When True, returns `(candidate, score)` pairs.
                The score is a squared L2 distance (lower is better) for
                `l2_unnormalised` indices and an inner product, i.e. cosine
                similarity (higher is better), for `cosine_normalised` ones —
                so it is comparable within a configuration, not across them.
                Rank order always follows FAISS's own ordering.

        Returns:
            list: Up to top_k candidate terms with their metadata, optionally
                paired with their scores. Fewer than top_k are returned when the
                vocabulary holds fewer terms than requested.
        """
        query_vector = self.encode_query(description)
        distances, indices = self.index.search(query_vector, k=top_k)
        # Retrieve term candidates, in order of similarity. FAISS pads the
        # result with -1 when the index holds fewer than k vectors (PBCore has
        # 93 terms, so any k near that value is padded); passing -1 to .iloc
        # would silently return the last row of the vocabulary as a candidate.
        retrieved_candidates, retrieved_scores = [], []
        for i in range(indices.shape[1]):
            idx = int(indices[0][i])
            if idx < 0:
                continue
            retrieved_candidates.append(self.metadata_df.iloc[idx])
            retrieved_scores.append(float(distances[0][i]))

        if len(retrieved_candidates) < top_k:
            logger.debug(
                f"[classification] Requested top_k={top_k} but the index holds "
                f"{self.index.ntotal} terms; returning {len(retrieved_candidates)}."
            )

        if verbose:  # print candidate terms to console
            print(f"Top {len(retrieved_candidates)} Retrieved Candidates:")
            for i, candidate in enumerate(retrieved_candidates):
                print(f"{i+1}. Label: {candidate['term']},"\
                      f" URI: {candidate['uri']},"\
                      f" vocab: {candidate['source_vocabulary']},"\
                      f" score: {retrieved_scores[i]:.4f}")

        if return_scores:
            return list(zip(retrieved_candidates, retrieved_scores))
        return retrieved_candidates

    def generate_prompt(self,
                        description: str,
                        retrieved_candidates: list,
                        classification_prompt: str = SINGLE_CLASS_PROMPT,
                        object_category: str = "",
                        predicted_type: str = "") -> str:
        """
        Generates a prompt for LLM classification based on the object description
        and retrieved candidate terms.

        Args:
            description (str): The description of the object to classify.
            retrieved_candidates (list): List of candidate terms with metadata.
            classification_prompt (str): The prompt template to use for the task.
            object_category (str): The broad category already assigned (e.g.
                'Merchandise') — included in the prompt so the model can judge
                specific type in context.
            predicted_type (str): An automatically predicted type, if available.
                It takes precedence over the category as the prompt's hint.

        Returns:
            str: The generated prompt for the LLM.
        """
        hint_block, _ = build_hint_block(predicted_type, object_category)
        prompt = classification_prompt.format(
            description=description, hint_block=hint_block)
        for i, candidate in enumerate(retrieved_candidates):
            prompt += f"({i+1}) Term: {candidate['term']},"\
                      f" Definition: {candidate['definition']},"\
                      f" URI: {candidate['uri']}\n"

        return prompt

    @staticmethod
    def build_retrieval_query(description: str, predicted_type: str = "") -> str:
        """
        Composes the retrieval query used by `classify`.

        The type hint steers FAISS toward type-relevant candidates while the
        description supplies context. Experiment B varies this formulation, so
        it is exposed rather than inlined.
        """
        return f"{predicted_type}: {description}" if predicted_type else description

    def classify(self,
                 description: str,
                 top_k: int = 10,
                 classification_prompt: str = 'single',
                 object_category: str = "",
                 predicted_type: str = "",
                 retrieval_query: str = None) -> tuple[list, list]:
        """
        Classifies an object based on its description by retrieving candidate
        terms and generating a prompt for LLM classification.

        Args:
            description (str): The description of the object to classify.
            top_k (int): The number of top candidates to retrieve.
            classification_prompt (str): 'single' or 'multi'.
            object_category (str): The broad category already assigned.
            predicted_type (str): An automatically predicted type, if available.
            retrieval_query (str): Overrides the query sent to the retriever,
                leaving the prompt unchanged. Used by the retrieval experiments
                to vary query formulation independently of the hint.

        Returns:
            tuple[list, list]: the selected (0-based) candidate positions and
                the retrieved candidates.
        """
        outcome = self.classify_instrumented(
            description=description,
            top_k=top_k,
            classification_prompt=classification_prompt,
            object_category=object_category,
            predicted_type=predicted_type,
            retrieval_query=retrieval_query,
        )
        return outcome["selected_indices"], outcome["candidates"]

    def classify_instrumented(self,
                              description: str,
                              top_k: int = 10,
                              classification_prompt: str = 'single',
                              object_category: str = "",
                              predicted_type: str = "",
                              retrieval_query: str = None) -> dict:
        """
        `classify` with the full record of what happened, for evaluation and
        error attribution.

        Returns a dict with the retrieval query, candidates and their scores,
        the exact prompt sent, the raw and parsed LLM responses, which hint the
        prompt carried, and an outcome in {'selected', 'abstained',
        'error_malformed', 'error_api'}. An abstention (the model returning 0
        under the STRICT RULE) is reported separately from a failure to parse,
        since the two mean opposite things when scoring.
        """
        assert classification_prompt in ['single', 'multi'], \
            "classification_prompt must be 'single' or 'multi'"
        prompt_template = SINGLE_CLASS_PROMPT \
            if classification_prompt == 'single' else MULTI_CLASS_PROMPT

        query = retrieval_query if retrieval_query is not None \
            else self.build_retrieval_query(description, predicted_type)
        scored = self.retrieve(query, top_k=top_k, return_scores=True)
        retrieved_candidates = [candidate for candidate, _ in scored]
        scores = [score for _, score in scored]

        _, hint_source = build_hint_block(predicted_type, object_category)
        prompt = self.generate_prompt(
            description, retrieved_candidates, prompt_template,
            object_category, predicted_type)

        result = {
            "selected_indices": [],
            "candidates": retrieved_candidates,
            "scores": scores,
            "retrieval_query": query,
            "prompt": prompt,
            "hint_source": hint_source,
            "raw_response": None,
            "parsed_response": None,
            "abstained": False,
            "outcome": None,
            "error": None,
        }

        if self.generator is None:
            raise RuntimeError(
                "This classifier was constructed without an LLM key "
                "(require_generator=False) and cannot run selection."
            )

        logger.info(f"[classification] prompt sent:\n{prompt}")

        try:
            response = self.generator.generate(prompt, SCHEMA_INT_LIST)
        except Exception as e:  # transport, quota, safety — all non-recoverable here
            logger.warning(f"[classification] LLM call failed: {e}")
            result["outcome"] = "error_api"
            result["error"] = f"{type(e).__name__}: {e}"
            return result

        result["raw_response"] = response.text
        result["model"] = self.generator.spec
        parsed_response = response.parsed
        result["parsed_response"] = parsed_response
        logger.info(f"[classification] response received: {parsed_response}")

        if not isinstance(parsed_response, list) or not parsed_response:
            logger.warning(
                f"[classification] Unusable LLM response: {parsed_response!r}")
            result["outcome"] = "error_malformed"
            result["error"] = f"expected a non-empty list, got {parsed_response!r}"
            return result

        # 0 is the abstention signal of the STRICT RULE. A response mixing it
        # with real positions is contradictory; the earlier revision returned
        # [] for it, silently discarding the valid selections it also carried.
        positions = [i for i in parsed_response if i != 0]
        if not positions:
            logger.info("[classification] LLM returned 0 — no suitable term.")
            result["abstained"] = True
            result["outcome"] = "abstained"
            return result
        if len(positions) != len(parsed_response):
            logger.warning(
                f"[classification] Response {parsed_response} mixes abstention "
                "with selections; keeping the selections.")

        # Convert (1-based) positions to (0-based) indexes, discarding anything
        # outside the candidate list rather than raising on the lookup.
        selected_indices = [i - 1 for i in positions
                            if 1 <= i <= len(retrieved_candidates)]
        if not selected_indices:
            logger.warning(
                f"[classification] Response {parsed_response} is out of range "
                f"for {len(retrieved_candidates)} candidates.")
            result["outcome"] = "error_malformed"
            result["error"] = (f"positions {positions} out of range for "
                               f"{len(retrieved_candidates)} candidates")
            return result

        selected_terms = [retrieved_candidates[i]["term"] for i in selected_indices]
        logger.info(f"[classification] selected terms: {selected_terms}")
        result["selected_indices"] = selected_indices
        result["outcome"] = "selected"
        return result


class ObjectCategoriser:
    """
    Classifies an object into one of the 7 pre-defined categories using an LLM
    with structured output.
    """
    def __init__(self, generator_api_key: str = '',
                 generator_model: str = 'gemini-2.5-flash',
                 temperature: float = 0.0,
                 prompt_variant: str = 'minimal',
                 generator: object = None):
        """
        Initialises the ObjectCategoriser with an LLM client.

        Args:
            generator_api_key (str): API key for the LLM service; falls back to
                the provider's environment variable.
            generator_model (str): The LLM to use, as `provider:model`. A bare
                name means Google, so 'gemini-2.5-flash' keeps working.
            temperature (float): Sampling temperature, 0.0 by default so that
                repeated runs are comparable.
            prompt_variant (str): 'minimal' (default, deployed) uses
                CATEGORY_ROLE — the seven names alone; 'scoped' uses
                CATEGORY_ROLE_SCOPED, with scope notes and few-shot examples,
                which the evaluation found no better (see the constant).
            generator (object): A ready-made `llm.LLMClient`, or a caching
                wrapper around one, used in preference to building a client.
        """
        if prompt_variant not in CATEGORY_ROLES:
            raise ValueError(f"prompt_variant must be one of {tuple(CATEGORY_ROLES)}")
        self.generator = generator or build_client(
            generator_model, api_key=generator_api_key, temperature=temperature)
        self.generator_model = getattr(self.generator, "spec", generator_model)
        self.temperature = temperature
        self.prompt_variant = prompt_variant
        self.system_instruction = CATEGORY_ROLES[prompt_variant]

    def classify(self, description: str) -> str:
        """
        Classifies the object description into one of the 7 categories.

        Args:
            description (str): The description of the object.

        Returns:
            str: The predicted category name.
        """
        return self.classify_instrumented(description)["predicted"]

    def classify_instrumented(self, description: str) -> dict:
        """
        `classify` with the raw response retained, plus a `valid` flag telling
        whether the answer is one of the seven categories. The enum schema makes
        an invalid answer unlikely, not impossible, and a silent miss would be
        scored as a wrong prediction rather than as an API defect.
        """
        prompt = CATEGORY_PROMPT.format(description)
        logger.info(f"[categoriser] prompt sent:\n{prompt}")
        result = {
            "predicted": None,
            "raw_response": None,
            "prompt": prompt,
            "prompt_variant": self.prompt_variant,
            "model": self.generator_model,
            "valid": False,
            "error": None,
        }
        try:
            response = self.generator.generate(
                prompt, SCHEMA_CATEGORY,
                system_instruction=self.system_instruction,
                choices=[c.value for c in ObjectCategory])
        except Exception as e:
            logger.warning(f"[categoriser] LLM call failed: {e}")
            result["error"] = f"{type(e).__name__}: {e}"
            return result

        logger.info(f"[categoriser] response received: {response.text!r}")
        result["raw_response"] = response.text
        # The provider layer normalises the enum to a bare string; Mistral's
        # structured output arrives wrapped in a JSON object, Gemini's does not.
        predicted = response.parsed if isinstance(response.parsed, str) else ""
        result["predicted"] = predicted
        result["valid"] = predicted in {c.value for c in ObjectCategory}
        if not result["valid"]:
            logger.warning(f"[categoriser] Unknown category returned: {raw!r}")
        return result


class EmbeddingObjectCategoriser:
    """
    Non-LLM category baseline (`C1`): embeds the object text and the seven
    CATEGORY_DEFINITIONS with the same sentence-transformer and returns the
    nearest category.

    Deterministic and free, which is what makes it the honest comparison for the
    LLM categoriser: a 7-way decision over well-separated definitions is not
    obviously a task that needs a generative model.
    """
    def __init__(self,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 definitions: dict = None,
                 include_label: bool = False):
        """
        Args:
            embedding_model (str): Sentence-transformer shared with retrieval.
            definitions (dict): category -> scope note. Defaults to
                CATEGORY_DEFINITIONS.
            include_label (bool): When True the category name is prepended to
                its definition before embedding. Off by default so the baseline
                is exactly "nearest definition".
        """
        self.definitions = definitions or CATEGORY_DEFINITIONS
        self.include_label = include_label
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.categories = list(self.definitions.keys())
        texts = [f"{c}: {self.definitions[c]}" if include_label else self.definitions[c]
                 for c in self.categories]
        # Cosine over unit-norm vectors: this component has no legacy index to
        # stay compatible with, so it uses the metric appropriate to sentence
        # embeddings rather than raw L2.
        self.category_vectors = self.embedding_model.encode(
            texts, normalize_embeddings=True).astype('float32')

    def classify(self, description: str) -> str:
        """Returns the nearest category to the given object text."""
        return self.classify_instrumented(description)["predicted"]

    def classify_instrumented(self, description: str) -> dict:
        """`classify` plus the cosine similarity of every category."""
        query = self.embedding_model.encode(
            [description], normalize_embeddings=True).astype('float32')
        similarities = (self.category_vectors @ query[0]).tolist()
        ranked = sorted(zip(self.categories, similarities),
                        key=lambda pair: pair[1], reverse=True)
        return {
            "predicted": ranked[0][0],
            "score": ranked[0][1],
            "scores": dict(zip(self.categories, similarities)),
            "ranking": [c for c, _ in ranked],
            "valid": True,
        }


class HierarchicalObjectClassifier:
    """
    Orchestrates a two-step classification process:
    1. Determines the broad object category (e.g., 'Musical Instrument').
    2. Uses a category-specific RAG classifier to find the specific term.
    Includes fallback to AAT for robustness.
    """
    def __init__(self, generator_api_key: str = '',
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 generator_model: str = 'gemini-2.5-flash',
                 vocab_paths: dict = None,
                 temperature: float = 0.0,
                 require_generator: bool = True,
                 categoriser: object = None,
                 generator: object = None):
        """
        Args:
            vocab_paths (dict): Overrides VOCAB_PATHS, so the evaluation can
                point the same orchestration at its tagged index builds without
                touching the production ones.
            temperature (float): Passed to both stages.
            require_generator (bool): See `RAGObjectClassifier`.
            categoriser (object): Any object exposing `classify(description)`,
                e.g. an `EmbeddingObjectCategoriser`. Defaults to the LLM
                `ObjectCategoriser`.
            generator (object): An `llm.LLMClient` (or caching wrapper) shared by
                every vocabulary classifier. Sharing one client means a
                multi-model sweep needs one cache per run rather than one per
                vocabulary, and the model in force is unambiguous.
        """
        self.vocab_paths = vocab_paths or VOCAB_PATHS
        # Initialise the broad categoriser
        self.categoriser = categoriser or ObjectCategoriser(
            generator_api_key, generator_model, temperature=temperature,
            generator=generator)

        # Initialise all specific classifiers
        self.specific_classifiers = {}
        print("--- Initialising Vocabulary-Specific Classifiers ---")
        for vocab_name, paths in self.vocab_paths.items():
            print(f"Loading classifier for '{vocab_name}'...")
            try:
                self.specific_classifiers[vocab_name] = RAGObjectClassifier(
                    index_path=paths["index"],
                    metadata_path=paths["metadata"],
                    embedding_model=embedding_model,
                    generator_model=generator_model,
                    generator_api_key=generator_api_key,
                    temperature=temperature,
                    require_generator=require_generator,
                    generator=generator,
                )
            except Exception as e:
                print(f"ERROR: Could not load classifier for {vocab_name}. " \
                      f"Please check paths: {paths}. Error: {e}")
        print("--- All classifiers loaded. ---\n")

    def classify(self, description: str,
                 top_k: int = 10,
                 prompt_type: str = 'single',
                 object_category: str = None,
                 predicted_type: str = "",
                 retrieval_query: str = None):
        """
        Performs the full classification for an object description.
        If object_category is provided, Step 1 is skipped.
        """
        outcome = self.classify_instrumented(
            description=description,
            top_k=top_k,
            prompt_type=prompt_type,
            object_category=object_category,
            predicted_type=predicted_type,
            retrieval_query=retrieval_query,
        )
        return outcome["selected_indices"], outcome["candidates"]

    def classify_instrumented(self, description: str,
                              top_k: int = 10,
                              prompt_type: str = 'single',
                              object_category: str = None,
                              predicted_type: str = "",
                              retrieval_query: str = None) -> dict:
        """
        `classify` with the routing recorded: which category was used and
        whether it was given or predicted, which vocabulary it routed to,
        whether the AAT fallback fired, and the selector's own instrumentation
        for each attempt.

        The fallback is production behaviour and stays on; recording it is what
        lets the share of type assertions drawn from AAT be reported.
        """
        result = {
            "selected_indices": [],
            "candidates": [],
            "scores": [],
            "object_category": object_category,
            "category_source": "given" if object_category else "predicted",
            "vocab": None,
            "fallback_triggered": False,
            "attempts": [],
            "outcome": None,
            "error": None,
        }

        # Step 1: Broad Categorisation (conditional)
        if object_category:
            assert object_category in [e.value for e in ObjectCategory]
        else:
            object_category = self.categoriser.classify(description)
            result["object_category"] = object_category

        # Step 2: Object classification
        target_vocab = VOCAB_MAP.get(object_category)
        if not target_vocab:
            logger.error(f"[hierarchical] Category '{object_category}' is not mapped.")
            result["outcome"] = "error_routing"
            result["error"] = f"unmapped category: {object_category!r}"
            return result
        result["vocab"] = target_vocab

        vocab_classifier = self.specific_classifiers.get(target_vocab)
        if not vocab_classifier:
            logger.error(f"[hierarchical] No classifier loaded for '{target_vocab}'.")
            result["outcome"] = "error_routing"
            result["error"] = f"no classifier for vocabulary: {target_vocab}"
            return result

        attempt = vocab_classifier.classify_instrumented(
            description=description,
            top_k=top_k,
            classification_prompt=prompt_type,
            object_category=object_category if target_vocab == "AAT" else "",
            predicted_type=predicted_type,
            retrieval_query=retrieval_query,
        )
        result["attempts"].append({"vocab": target_vocab, **attempt})

        # Fallback Logic: try with AAT. Only a genuine abstention — the routed
        # vocabulary holding no suitable term — justifies searching AAT. An API
        # or parse failure is not evidence about the vocabulary, and falling
        # back on it would turn a transient outage into an AAT type assertion
        # and hide the failure from the error attribution.
        if attempt["outcome"] == "abstained" and target_vocab != "AAT":
            logger.info(f"[hierarchical] '{target_vocab}' returned no match; "
                        "falling back to AAT.")
            fallback_classifier = self.specific_classifiers.get("AAT")
            if fallback_classifier is not None:
                result["fallback_triggered"] = True
                result["vocab"] = "AAT"
                attempt = fallback_classifier.classify_instrumented(
                    description=description,
                    top_k=top_k,
                    classification_prompt=prompt_type,
                    object_category=object_category,
                    predicted_type=predicted_type,
                    retrieval_query=retrieval_query,
                )
                result["attempts"].append({"vocab": "AAT", **attempt})

        result["selected_indices"] = attempt["selected_indices"]
        result["candidates"] = attempt["candidates"]
        result["scores"] = attempt["scores"]
        result["outcome"] = attempt["outcome"]
        result["error"] = attempt["error"]
        if not attempt["selected_indices"]:
            logger.info("[hierarchical] No suitable term found.")
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RAG Vector DB Builder & Object Classifier")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Subparser for building the vector database ---
    build_parser = subparsers.add_parser('build', help='Build a vocabulary-specific vector database')
    build_parser.add_argument('--vocabulary', required=True, type=str, help='Path to vocabulary CSV file')
    build_parser.add_argument('--subset', nargs='+', type=str, help='A space-separated list of vocabulary prefixes to include')
    build_parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    build_parser.add_argument('--index_strategy', type=str, choices=list(INDEX_STRATEGIES), default='combine', help='Index construction strategy')
    build_parser.add_argument('--similarity', type=str, choices=list(SIMILARITIES), default=SIMILARITY_L2, help='Similarity metric the index is built for')
    build_parser.add_argument('--output_dir', type=str, default='vocabularies/vectordb', help='Output directory for vector DB')
    build_parser.add_argument('--vocabulary_name', type=str, default='vocabulary', help='Name for saved index file')

    # --- Subparser for low-level, direct classification ---
    classify_parser = subparsers.add_parser('classify', help='Classify an object using a single specified vector DB')
    classify_parser.add_argument('--description', type=str, required=True, help='Object description for classification')
    classify_parser.add_argument('--index_path', type=str, required=True, help='Path to FAISS index file')
    classify_parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata pickle file')
    classify_parser.add_argument('--generator_api_key', type=str, default='', help='API key for Gemini')
    classify_parser.add_argument('--generator_model', type=str, default='gemini-2.5-flash', help='Gemini model name')
    classify_parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    classify_parser.add_argument('--top_k', type=int, default=10, help='Number of candidates to retrieve')
    classify_parser.add_argument('--classification_prompt', type=str, choices=['single', 'multi'], default='single', help='Prompt type')

    # --- Subparser for the full hierarchical classification (catalogue)
    catalogue_parser = subparsers.add_parser('catalogue', help='Perform full two-step hierarchical cataloguing')
    catalogue_parser.add_argument('--description', type=str, required=True, help='Object description for classification')
    catalogue_parser.add_argument('--generator_api_key', type=str, default='', help='API key for Gemini')
    catalogue_parser.add_argument('--generator_model', type=str, default='gemini-2.5-flash', help='Gemini model name')
    catalogue_parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    catalogue_parser.add_argument('--top_k', type=int, default=10, help='Number of candidates to retrieve')
    catalogue_parser.add_argument('--classification_prompt', type=str, choices=['single', 'multi'], default='single', help='Prompt type')
    catalogue_parser.add_argument('--category', type=str, choices=[c.value for c in ObjectCategory],
                                  help='Optionally, provide the broad object category to skip the first classification step.')

    args = parser.parse_args()

    if args.command == 'build':
        full_vocab_df = pd.read_csv(args.vocabulary)
        build_vector_database(
            full_vocab_df,
            model_name=args.model_name,
            index_strategy=args.index_strategy,
            similarity=args.similarity,
            output_dir=args.output_dir,
            vocabulary_name=args.vocabulary_name,
            subset=args.subset
        )
    elif args.command == 'classify':
        classifier = RAGObjectClassifier(
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            embedding_model=args.embedding_model,
            generator_model=args.generator_model,
            generator_api_key=args.generator_api_key
        )
        indices, candidates = classifier.classify(
            description=args.description,
            top_k=args.top_k,
            classification_prompt=args.classification_prompt
        )
        print("\n--- Classification Result ---")
        if not indices:
            print("No suitable term found by the LLM.")
        else:
            for i in indices:
                term = candidates[i]
                print(f"-> Term: {term['term']} | URI: {term['uri']}")
        print("\n--- Top Retrieved Candidates ---")
        for i, candidate in enumerate(candidates):
            print(f"{i+1}. {candidate['term']} ({candidate['source_vocabulary']})")
            
    elif args.command == 'catalogue':
        # Use the hierarchical classifier
        classifier = HierarchicalObjectClassifier(
            generator_api_key=args.generator_api_key,
            embedding_model=args.embedding_model,
            generator_model=args.generator_model
        )
        selected_indices, candidates = classifier.classify(
            description=args.description,
            top_k=args.top_k,
            prompt_type=args.classification_prompt,
            object_category=args.category
        )

        print("\n--- Final Catalogue Result ---")
        if selected_indices:
            for term in [candidates[i] for i in selected_indices]:
                print(f"-> Term: {term['term']}\n   URI: {term['uri']}\n "\
                      f"Source: {term['source_vocabulary']}")
        else:
            print("The cataloguing process did not yield a final result.")
