"""
Provides a full pipeline for ingesting, processing, and triplifying item data
from a CSV file or dictionary data into a knowledge graph in Turtle format. The pipeline includes:
- Reading and validating CSV data, mapping fields to a Pydantic Item model.
- Processing dictionary data with pre-structured fields (lists for multi-value fields).
- Classifying items by category and type using a hierarchical object classifier.
- Linking entities (e.g., genre, artist, work) to external URIs via a WD linker.
- Triplifying processed items into RDF triples and serializing them as Turtle files.
- Aggregating individual Turtle files into a single knowledge graph.

Configuration options are available via command-line arguments for input paths,
API keys, output directories, and debug mode. The script relies on external
modules for classification, linking, triplification, and namespace management.

Main Functions:
---------------
- create_linked_entity_list(value): Splits a string into LinkedEntity objects.
- process_csv_row(row): Converts a CSV row into an Item model instance.
- process_dict_row(data_dict): Converts a dictionary into an Item model instance.
- ingest_data_from_csv(file_path): Reads CSV and returns a list of Item objects.
- ingest_data_from_dicts(data_list): Processes dictionary list and returns Item objects.
- process_single_item(...): From Item to classification, linking, and triplification.
- aggregate_graphs(source_dir, output_file): Merges all Turtle files into one RDF graph.
- main(): Orchestrates the pipeline based on command-line arguments.

Usage:
------
Run the script from the command line. Example:
    python populate.py path/to/data.csv --api-key YOUR_API_KEY 
       --output path/to/output --aggregate-file knowledge_graph.ttl

Or use programmatically with dictionary data:
    items = ingest_data_from_dicts([your_dict_data])
"""
import uuid
import warnings
import argparse
import json

from pathlib import Path
from typing import List, Dict, Any, Optional
from curses.ascii import isdigit

import pandas as pd
from rdflib import Graph
from tqdm import tqdm

from .metamodel import Item, LinkedEntity
from .classification import HierarchicalObjectClassifier
from .linker import RAGEntityLinker
from .scho_triplifier import Triplifier
from .namespaces import bind_namespaces

# --- Configuration ---
# Define the default path to your CSV file for test runs or when no path is provided.
DEFAULT_MAPPINGS_PATH = Path(__file__).parent.parent / "mappings_min.json"
DEFAULT_VOCAB_FILE_PATH = Path(__file__).parent / "vocabularies" / "all_concepts.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "assets" / "triples"
DEFAULT_AGGREGATE_FILE = "knowledge_graph.ttl"

# --- Field Mappings ---
MULTI_VALUE_FIELDS = [
    "genre", "music_work", "music_album", "music_artist", "event",
    "country", "place", "venue", "colour", "material", "exhibition"
]
SINGLE_ENTITY_FIELDS = ["category", "type"]
LITERAL_LIST_FIELDS = ["year"]  # plain string lists, not LinkedEntity

# --- Core Functions ---

def create_linked_entity_list(value: Optional[str]) -> List[LinkedEntity]:
    """
    Takes a string (potentially with '|' separators), splits it, and returns
    a list of LinkedEntity objects with no URI.
    """
    if not value or pd.isna(value):
        return []
    labels = [label.strip() for label in str(value).split('|')]
    return [LinkedEntity(uri=None, label=label) for label in labels if label]

def process_csv_row(row: pd.Series) -> Item:
    """
    Takes a single row from a pandas DataFrame and converts it into an
    Item Pydantic model instance.
    """
    item_data: Dict[str, Any] = {}
    item_data["identifier"] = ""  # tmp
    for model_field, value in row.items():
        if pd.isna(value):
            continue
        if model_field not in Item.model_fields:
            continue
        if model_field in SINGLE_ENTITY_FIELDS:
            item_data[model_field] = LinkedEntity(uri=None, label=str(value))
        elif model_field in MULTI_VALUE_FIELDS:
            item_data[model_field] = create_linked_entity_list(str(value))
        else:
            item_data[model_field] = str(value)
    # Create the Item instance and generate identifier
    item = Item.model_validate(item_data)
    item.identifier = item.generate_identifier()
    return item

def process_dict_row(data_dict: Dict[str, Any]) -> Item:
    """
    Takes a dictionary with item data and converts it into an Item Pydantic model instance.
    Unlike process_csv_row, this function expects list values for multi-value fields
    rather than pipe-separated strings.
    
    Args:
        data_dict: Dictionary containing item data where multi-value fields are already lists
        
    Returns:
        Item: Validated Item instance with generated identifier
    """
    item_data: Dict[str, Any] = {}
    item_data["identifier"] = ""  # tmp
    
    for model_field, value in data_dict.items():
        # Skip None values and fields not in the model
        if value is None:
            continue
        if model_field not in Item.model_fields:
            continue
            
        if model_field in SINGLE_ENTITY_FIELDS:
            # Single entity fields expect a string value
            item_data[model_field] = LinkedEntity(uri=None, label=str(value))
        elif model_field in LITERAL_LIST_FIELDS:
            # Plain string list fields — no LinkedEntity wrapping
            if isinstance(value, list):
                item_data[model_field] = [str(v) for v in value if v]
            elif value:
                item_data[model_field] = [str(value)]
            else:
                item_data[model_field] = []
        elif model_field in MULTI_VALUE_FIELDS:
            # Multi-value fields expect a list of strings
            if isinstance(value, list):
                # Create LinkedEntity objects from list items
                item_data[model_field] = [
                    LinkedEntity(uri=None, label=str(item)) 
                    for item in value if item  # Skip empty/None items
                ]
            elif value:  # Handle case where it's a single string instead of list
                item_data[model_field] = [LinkedEntity(uri=None, label=str(value))]
            else:
                item_data[model_field] = []
        else:
            # Regular string fields
            item_data[model_field] = str(value)
    
    # Create the Item instance and generate identifier
    item = Item.model_validate(item_data)
    item.identifier = item.generate_identifier()
    return item

def ingest_data_from_csv(file_path: Path) -> List[Item]:
    """
    Reads the specified CSV file, performs sanity checks, and returns a list of
    Item objects.
    """
    if not file_path.exists():
        print(f"Error: CSV file not found at {file_path}")
        return []
    print(f"Reading data from {file_path}...")
    df = pd.read_csv(file_path)
    model_fields = set(Item.model_fields.keys())
    for col in df.columns:
        if col not in model_fields:
            warnings.warn(
                f"CSV column '{col}' not in metamodel and will be ignored."
            )
    items: List[Item] = []
    for index, row in df.iterrows():
        try:  # catch any row-specific errors
            items.append(process_csv_row(row))
        except Exception as e:
            print(f"Error processing row {index + 2}: {e}")
    print(f"Successfully ingested and created {len(items)} Item objects.")
    return items

def ingest_data_from_dicts(data_list: List[Dict[str, Any]]) -> List[Item]:
    """
    Takes a list of dictionaries with item data and returns a list of Item objects.
    Each dictionary should contain item fields where multi-value fields are already lists.
    
    Args:
        data_list: List of dictionaries containing item data
        
    Returns:
        List[Item]: List of validated Item objects
    """
    print(f"Processing {len(data_list)} dictionary records...")
    model_fields = set(Item.model_fields.keys())
    
    # Check for unknown fields in the first dictionary as a sample
    if data_list:
        for field in data_list[0].keys():
            if field not in model_fields:
                warnings.warn(
                    f"Field '{field}' not in metamodel and will be ignored."
                )
    
    items: List[Item] = []
    for index, data_dict in enumerate(data_list):
        try:  # catch any row-specific errors
            items.append(process_dict_row(data_dict))
        except Exception as e:
            print(f"Error processing dictionary {index + 1}: {e}")
    
    print(f"Successfully ingested and created {len(items)} Item objects.")
    return items

def find_best_vocab_match(vocab_df, query):
    """
    Finds the best matching term in a DataFrame and returns the URI and term.
    This implements a fallback strategy for the testset in case the linker fails.
    Given a string label as query, the best match is defined as the term that contains the query and has the minimum number of extra characters.
    
    XXX This should be refactored into a separate Linker class to complete RAG. 

    Args:
        df (pd.DataFrame): The DataFrame to search.
        query (str): The search query.

    Returns:
        dict: A dictionary containing the 'uri' and 'term' of the best match, 
              or None if no match is found.
    """
    # Filter for terms containing the query
    containing_query = vocab_df[vocab_df['term'].str.contains(
        query, case=False, na=False)].copy()
    if containing_query.empty:
        return None
    # Calculate the number of extra characters
    containing_query['extra_chars'] = containing_query['term'].str.len() - len(query)
    # Find the minimum number of extra characters
    min_extra_chars = containing_query['extra_chars'].min()
    # Get all records with the minimum number of extra characters
    best_matches = containing_query[containing_query['extra_chars'] == min_extra_chars]
    # Get the first of the best matches
    best_match_record = best_matches.iloc[0]
    # Return only the URI and term (label)
    return {
        'uri': best_match_record['uri'],
        'term': best_match_record['term']
    }

def process_single_item(
    item: Item,
    classifier: HierarchicalObjectClassifier,
    linker: RAGEntityLinker,
    triplifier: Triplifier,
    mappings: Dict,
    output_dir: Path,
    vocab_df: pd.DataFrame,
    classifier_top_types: int = 20,
    linker_top_entities: int = 10,
    linker_validate_entities: bool = False,
    ttl_dir: Path | None = None,
):
    """
    Processes a single item through classification, linking, and triplification.
    """
    if linker_validate_entities:  # TODO Plugging entichecks ASK queries
        raise NotImplementedError("Entity validation not yet integrated.")
    print(f"\n--- Processing Item: {item.name or item.identifier[:8]} ---")

    # 1. Classification (Category and Type)
    print("Classifying category and type...")
    try:
        # Category classification
        cat_label = classifier.categoriser.classify(item.description).strip()
        cat_key = cat_label.replace(" ", "")
        if cat_key in mappings["category"]:
            cat_dict = mappings["category"][cat_key]
            item.category = LinkedEntity(
                uri=cat_dict['_type_uri'][0],
                label=cat_dict['_description']
            )
            print(f"  Category classified as: {item.category.label}")

            # Type classification (depends on category)
            # predicted_type (specific_type from extraction) is passed as a hint:
            # - steers FAISS retrieval via compound query "specific_type: description"
            # - shown in prompt as PREDICTED TYPE (indicative, not definitive)
            predicted_type = item.type.label if item.type and item.type.label else ""
            indices, candidates = classifier.classify(
                item.description,
                object_category=cat_label,
                top_k=classifier_top_types,
                predicted_type=predicted_type)
            if indices:  # at least 1 type found
                obj_type = candidates[indices[0]]
                item.type = LinkedEntity(
                    uri=obj_type['uri'],
                    label=obj_type['term']
                )
                print(f"  Type classified as: {item.type.label}")
            elif item.type and item.type.label:  # try vocab match on extracted label
                print(f"  No type found, trying vocab match on extracted label: {item.type.label}")
                vocab_match = find_best_vocab_match(vocab_df, item.type.label)
                if vocab_match:
                    item.type = LinkedEntity(
                        uri=vocab_match['uri'],
                        label=vocab_match['term']
                    )
                    print(f"  Fallback linked to: {item.type.uri}")
                else:
                    # No match — clear type so triplifier does not mint a new URI
                    print(f"  No vocabulary match for '{item.type.label}' — type will be omitted.")
                    item.type = None
            else:
                print(f"  No type found and no extracted label — type will be omitted.")
    except Exception as e:
        print(f"  Classification failed: {e}")

    # 2. Entity Linking
    fields_to_link = [
        "genre", "music_work", "music_album", "music_artist", "event",
        "country", "place", "venue", "material"
    ]
    for field_name in fields_to_link:
        entities = getattr(item, field_name)
        if entities:  # at least 1 entity to link
            for i, entity in enumerate(entities):
                if entity.uri is None:
                    print(f"Linking '{entity.label}' (field: {field_name})...")
                    try:
                        linked_entity = linker.link(
                            search_term=entity.label,
                            context_query=item.description,
                            entity_type=field_name,
                            validate=linker_validate_entities,
                            top_k=linker_top_entities,
                        )
                        if linked_entity:
                            uri = "http://www.wikidata.org/entity/" + linked_entity["wikidata_id"]
                            entities[i] = LinkedEntity(uri=uri, label=linked_entity["label"])
                            print(f"  -> Linked to: {uri}")
                        else:
                            print(f"  -> No link found.")
                    except Exception as e:
                        print(f"  -> Linking failed: {e}")

    # 3. Final data touches (e.g., exhibition label and URIs)
    if item.exhibition:
        exhibition_mapping = mappings.get("exhibition", {})
        for ex in item.exhibition:
            if ex.label and isdigit(ex.label) and "Zone" not in ex.label:
                ex.label = "Zone" + ex.label
            # An exhibition URI is minted only while `exhibition` still declares a type to
            # mint it under. It no longer does: the exhibition is derived on read from the
            # item's display case (scho:location -> Case -> scho:containedInPlace -> Zone)
            # rather than stored, so mappings_min.json gives the field no relation and no
            # type, and the triplifier emits nothing for it. Guarded rather than deleted so
            # that restoring the mapping restores the behaviour.
            if ex.uri is None and exhibition_mapping.get("_type_uri"):
                entity_class_name = triplifier._get_class_name_from_uri(
                    exhibition_mapping["_type_uri"][0]
                )
                ex.uri = triplifier.base_uri[f"{entity_class_name}/{ex.label}"]

    # 4. Triplification
    dest_dir = ttl_dir if ttl_dir is not None else output_dir
    output_path = dest_dir / f"{item.identifier}.ttl"
    print(f"Triplifying item to {output_path}...")
    triplifier.triplify(item, output_path=str(output_path))


def aggregate_graphs(source_dir: Path, output_file: Path):
    """
    Reads all .ttl files in a directory and merges them into a single graph,
    excluding the output file itself if it already exists.
    """
    print(f"\n--- Aggregating all TTL files from '{source_dir}' ---")
    aggregate_graph = Graph()

    ttl_files = [
        f for f in source_dir.glob("*.ttl")
        if f.resolve() != output_file.resolve()
    ]
    if not ttl_files:
        print("No TTL files found to aggregate.")
        return

    for f in ttl_files:
        aggregate_graph.parse(str(f), format="turtle")
    # Bind namespaces and dump the final graph
    bind_namespaces(aggregate_graph)
    aggregate_graph.serialize(destination=str(output_file), format="turtle")
    print(f"Knowledge graph successfully aggregated and saved to '{output_file}'.")


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline to ingest, process, and triplify museum item data."
    )
    parser.add_argument(
        "csv_path",
        help="Path to the input CSV file (the collection export to ingest)."
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Gemini API Key for classification and linking."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save individual TTL files. Defaults to '{DEFAULT_OUTPUT_DIR}'."
    )
    parser.add_argument(
        "--aggregate-file",
        default=str(DEFAULT_AGGREGATE_FILE),
        help=f"Name of the final aggregated knowledge graph file. Defaults to '{DEFAULT_AGGREGATE_FILE}'."
    )
    parser.add_argument(
        "--vocab-path",
        default=str(DEFAULT_VOCAB_FILE_PATH),
        help=f"Path to the vocabulary CSV. Defaults to '{DEFAULT_VOCAB_FILE_PATH}'."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to process only the first 5 items."
    )
    parser.add_argument(
        "--base-uri",
        default="http://radio.liv.ac.uk/bme/",
        help="Base URI for minting new resource URIs. Defaults to 'http://radio.liv.ac.uk/bme/'."
    )
    parser.add_argument(
        "--uri-strategy",
        choices=['uuid', 'slug', 'hash'],
        default='hash',
        help="Strategy for minting new URIs. Defaults to 'hash'."
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model for similarity tasks. Defaults to 'all-MiniLM-L6-v2'."
    )
    parser.add_argument(
        "--generator-model",
        default="gemini-2.5-flash",
        help="Generator model for text generation tasks. Defaults to 'gemini-2.5-flash'."
    )
    parser.add_argument(
        "--linker_top_entities",
        type=int,
        default=10,
        help="Number of entities to retrieve for linking. Defaults to 10."
    )
    parser.add_argument(
        "--classifier_top_types",
        type=int,
        default=20,
        help="Number of types to retrieve for classification. Defaults to 20."
    )
    args = parser.parse_args()

    # --- Setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with open(DEFAULT_MAPPINGS_PATH, 'r') as f:
        mappings = json.load(f)
    # Also load the vocabularies CSV
    vocab_df = pd.read_csv(DEFAULT_VOCAB_FILE_PATH)

    print("Initialising services (Classifier, Linker, Triplifier)...")
    # Initialising the classifier (with both category and type models)
    classifier = HierarchicalObjectClassifier(
        generator_api_key=args.api_key,
        embedding_model=args.embedding_model,
        generator_model=args.generator_model,
    )
    # Initialising the entity linker for the entities to be linked
    linker = RAGEntityLinker(
        generator_api_key=args.api_key,
        generator_model=args.generator_model,
    )
    # Initialising the triplifier with mappings and URI strategy
    triplifier = Triplifier(
        mappings_path=str(DEFAULT_MAPPINGS_PATH),
        base_uri=args.base_uri,  # from argparse
        uri_strategy=args.uri_strategy,  # from argparse
    )

    # Run Pipeline on the items
    items_to_process = ingest_data_from_csv(Path(args.csv_path))
    print(f"Found {len(items_to_process)} items to process.")
    if args.debug:  # Process only first 5 items in debug mode
        items_to_process = items_to_process[:5]
        print("Debug mode enabled: processing only the first 5 items.")

    for item in tqdm(items_to_process, desc="Processing items"):
        try:
            process_single_item(
                item, classifier, linker, triplifier,
                mappings, output_dir, vocab_df=vocab_df,
                classifier_top_types=args.classifier_top_types,
                linker_top_entities=args.linker_top_entities
            )
        except Exception as e:
            print(f"FATAL: Unexpected error while processing item {item.identifier}. Skipping. Error: {e}")

    # Aggregate all the individual TTL files into one for deployment
    final_output_path = output_dir / args.aggregate_file
    aggregate_graphs(output_dir, final_output_path)


if __name__ == "__main__":
    main()
