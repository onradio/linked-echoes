"""
This module is designed to evaluate the performance of a language model (LLM) in
extracting structured data from unstructured text.

"""
import json
import re
import unicodedata
from collections import defaultdict
from typing import Optional, List, Any, Dict
from typing import Tuple

import pandas as pd
import pydantic
from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from utils.dataset import load_and_preprocess_csv_pandas

class AnnotationItem(BaseModel):
    """
    Represents a structured description expressing the "annotatable" content of
    a music item. This class omits fields that are not relevant for evaluation,
    such as the title, description, caption, or any other metadata-specific
    information. It focuses on the core attributes that can be extracted or 
    annotated from the item itself, via prior knowledge or external sources.
    
    XXX This definition necessitates harmonisation with Item in lechoes!!!
    """
    base_type: str = Field(
        ...,
        title="Base Type",
        description="The broad category or high-level classification of the item (e.g., Instrument, Record, Clothing)."
    )
    specific_type: Optional[str] = Field(
        default=None,
        title="Specific Type",
        description="The most granular type of item (e.g., Acoustic Guitar, LP, Suit, Poster)."
    )
    music_genres: Optional[List[str]] = Field(
        default=None,
        title="Music Genres",
        description="The music genres associated with the item, if any."
    )
    music_works: Optional[List[str]] = Field(
        default=None,
        title="Music Works",
        description="The music works (e.g., song titles) associated with the item, if any."
    )
    music_albums: Optional[List[str]] = Field(
        default=None,
        title="Music Albums",
        description="The music albums titles associated with the item, if any."
    )
    music_artists: Optional[List[str]] = Field(
        default=None,
        title="Music Artists",
        description="The music artists or groups associated with the item, if any."
    )
    events: Optional[List[str]] = Field(
        default=None,
        title="Events",
        description="The events (e.g., concerts, festivals, specific contests) associated with the item, if any."
    )
    countries: Optional[List[str]] = Field(
        default=None,
        title="Countries",
        description="The countries associated with the item (e.g., origin, event location), if any."
    )
    places: Optional[List[str]] = Field(
        default=None,
        title="Places",
        description="The place(s) associated with the item (e.g., origin, event location), if any."
    )
    venues: Optional[List[str]] = Field(
        default=None,
        title="Venues",
        description="The specific venues (e.g., clubs, theatres, halls) associated with the item, if any."
    )
    years: Optional[List[str]] = Field(
        default=None,
        title="Years",
        description="The year or years associated with the item (e.g., creation, release, event), if any."
    )
    colours: Optional[List[str]] = Field(
        default=None,
        title="Colours",
        description="The prominent colours associated with the item, if any."
    )
    cardinality: Optional[int] = Field(
        default=None,
        title="Cardinality",
        description="The quantity of this item in the collection, if specified (often 1)."
    )
    materials: Optional[List[str]] = Field(
        default=None,
        title="Materials",
        description="The materials the item is made from, if known."
    )

    # --- Fields Excluded from Evaluation (but defined for completeness) ---
    item_title: Optional[str] = Field(
        default=None,
        title="Item Title",
        description="The most descriptive, unique, yet concise title for the item."
    )
    item_description: Optional[str] = Field(
        default=None,
        title="Item Description",
        description="A more verbose description of the item."
    )

    @classmethod
    def model_json_schema(cls, **kwargs):
        # Get the original schema
        schema = super().model_json_schema(**kwargs)
        # Post-process: Replace `anyOf` with `type` + `nullable`
        for field, definition in schema.get("properties", {}).items():
            if "anyOf" in definition:
                anyof = definition["anyOf"]
                if len(anyof) == 2 and {"type": "null"} in anyof:
                    # Find the non-null schema type
                    non_null = [t for t in anyof if t.get("type") != "null"][0]

                    # Preserve existing metadata
                    preserved_keys = {k: v for k, v in definition.items() if k != "anyOf"}

                    # Start fresh but preserve metadata
                    definition.clear()
                    definition.update(non_null)
                    definition.update(preserved_keys)  # Add back title, description, etc.
                    definition["nullable"] = True

        return schema

# --- Evaluation Configuration ---
# List of fields to be included in the evaluation metrics
EVALUATED_FIELDS = [
    "base_type", "specific_type", "music_genres", "music_works",
    "music_albums", "music_artists", "events", "countries", "places",
    "venues", "years", "colours", "materials", "item_title", "cardinality"
]

# Set of fields that contain lists of values
LIST_FIELDS = {
    "music_genres", "music_works", "music_albums", "music_artists",
    "events", "countries", "places", "venues", "years", "colours", "materials"
}

# Set of fields that contain single values
SINGLE_VALUE_FIELDS = {"base_type", "specific_type", "item_title", "cardinality"}

# --- Helper Functions ---

def normalise_values(value: Any) -> Any:
    """
    Applies normalization to a value for comparison purposes. For strings:
        - Converts to lowercase.
        - Normalizes Unicode characters (e.g., converts accents like é to e).
        - Removes apostrophes (both ' and ’).
        - Removes common punctuation (keeps letters, numbers, spaces, hyphens).
        - Collapses multiple whitespace characters into single spaces.
        - Strips leading/trailing whitespace.

    Args:
        value: The input value (string, int, list, None, etc.).

    Returns:
        The normalized value, or the original value if not a string or None.
    """
    if value is None or not isinstance(value, str):
        return value  # keep None or non-string values as is
    # 1. Convert to lowercase for case-insensitive comparison
    text = value.lower()
    # 2. Normalize Unicode characters (e.g., decompose accents)
    #    NFKD compatibility decomposition splits chars like 'é' into 'e' and '´'
    #    We then filter out the non-spacing marks (like the accent '´').
    try:
        # Normalize to decomposed form (Compatibility Decomposition)
        text = unicodedata.normalize('NFKD', text)
        # Filter out combining characters (diacritics/accents)
        text = "".join([c for c in text if not unicodedata.combining(c)])
    except Exception:
        # Fallback if normalization fails for any reason
        pass # Keep text as is after lowercasing

    # 3. Remove apostrophes (both straight and curly)
    text = re.sub(r"['’]", "", text)
    # 4. Remove characters that are NOT alphanumeric, whitespace, or hyphens.
    #    Keeps a-z, 0-9, space, tab, newline, etc. (\s), and hyphen (-)
    #    Adjust the regex '[^a-z0-9\s-]' to keep other specific symbols
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    # 5. Collapse multiple whitespace characters (spaces, tabs) into 1 space
    text = re.sub(r'\s+', ' ', text)
    # 6. Strip leading/trailing whitespace
    text = text.strip()

    return text


def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Calculates Precision, Recall, and F1-Score. Handles division by zero,
    returning 0.0 in such cases.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_list_field_set(gt_list: Optional[List[str]],pred_list: Optional[List[str]]) -> Tuple[int, int, int]:
    """
    Calculates TP, FP, FN for list fields using set operations.
    This is a simple set-based evaluation without fuzzy matching.
    """
    gt_set = set(normalise_values(v) for v in gt_list if v is not None) if gt_list else set()
    pred_set = set(normalise_values(v) for v in pred_list if v is not None) if pred_list else set()
    # Calculate TP, FP, FN using set operations
    tp = len(gt_set.intersection(pred_set))
    fp = len(pred_set.difference(gt_set))
    fn = len(gt_set.difference(pred_set))

    return tp, fp, fn


def evaluate_list_field_fuzzy(gt_list: Optional[List[str]],
                              pred_list: Optional[List[str]],
                              threshold=90) -> Tuple[int, int, int]:
    """
    Calculates TP, FP, FN for list fields using fuzzy matching.
    This uses rapidfuzz to find the best match in the ground truth list
    """
    if gt_list is None: gt_list = []
    if pred_list is None: pred_list = []

    # Normalize all items first to ensure consistent comparison
    gt_norm = [normalise_values(v) for v in gt_list
               if v is not None and normalise_values(v)]
    pred_norm = [normalise_values(v) for v in pred_list
                 if v is not None and normalise_values(v)]

    if not gt_norm and not pred_norm:
        return 0, 0, 0
    # Use process.extract to find best match in gt for each pred item
    # This is one way to handle matching; others exist (e.g., bipartite graph matching)
    tp = 0
    matched_gt_indices = set()
    pred_matched_indices = set()

    for i, p_item in enumerate(pred_norm):
        # Find the best match in the ground truth list above the threshold
        # best_match is a tuple (matched_item, match_score, match_index)
        best_match = process.extractOne(
            p_item, gt_norm, scorer=fuzz.WRatio, score_cutoff=threshold)
        if best_match:  # If a match is found in gt_norm
            match_score, match_index = best_match[1], best_match[2]
            # Ensure a GT item is only matched once to avoid inflated TP
            if match_index not in matched_gt_indices:
                tp += 1
                matched_gt_indices.add(match_index)
                pred_matched_indices.add(i) # Mark this prediction as matched

    # FP is predicted items that didn't find a good match in GT
    fp = len(pred_norm) - len(pred_matched_indices)
    # FN is ground truth items that weren't matched by any prediction
    fn = len(gt_norm) - len(matched_gt_indices)

    return tp, fp, fn


def evaluate_items(ground_truth_items: List[AnnotationItem],
                   predicted_items: List[AnnotationItem],
                   eval_fn=evaluate_list_field_fuzzy,
                   **eval_kwargs) -> Dict[str, Any]:
    """
    Evaluates LLM predictions against ground truth using specified fields and
    macro-averaging. Evaluation of list fields is done using the provided
    `eval_fn` (default is fuzzy matching). The function calculates True
    Positives (TP), False Positives (FP), and False Negatives (FN) for each
    field, and computes Precision, Recall, and F1-Score for each field. It also
    computes overall macro-averaged metrics across all evaluated fields.

    Args:
        ground_truth_items: A list of Pydantic Item objects representing the ground truth.
        predicted_items: A list of Pydantic Item objects representing the LLM predictions.
                         Assumes items correspond by index with the ground truth.
        eval_fn: The evaluation function to use for list fields (default is fuzzy matching).
        **eval_kwargs: Additional keyword arguments for the evaluation function.

    Returns:
        A dictionary containing per-field metrics and overall macro-averaged metrics.
        Example:
        {
            "per_field_metrics": {
                "field_name": {"tp": N, "fp": N, "fn": N, "precision": X, "recall": Y, "f1": Z},
                ...
            },
            "overall_macro_metrics": {
                "precision": A, "recall": B, "f1": C
            }
        }
    """        
    if len(ground_truth_items) != len(predicted_items):
        print(f"length of ground_truth_items : {len(ground_truth_items)}")
        print(f"length of predicted_items: {len(predicted_items)}")
        raise ValueError("Ground truth and predicted lists must have the same length.")

    # Dictionary to store TP, FP, FN counts for each evaluated field
    field_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    # Iterate through each item pair (ground truth vs prediction)
    for gt_item, pred_item in zip(ground_truth_items, predicted_items):
        # Iterate through each field designated for evaluation
        for field in EVALUATED_FIELDS:
            # Get the value from ground truth and prediction, None if missing
            gt_val = getattr(gt_item, field, None)
            pred_val = getattr(pred_item, field, None)
            tp, fp, fn = 0, 0, 0 # Initialize counts for this field in this item

            # --- Comparison Logic ---
            if field in LIST_FIELDS:
                tp, fn, fp = eval_fn(gt_val, pred_val, **eval_kwargs)
            elif field in SINGLE_VALUE_FIELDS:
                # Normalize the single value
                gt_norm = normalise_values(gt_val)
                pred_norm = normalise_values(pred_val)
                # Determine if values exist (treat None and empty string ""
                # as non-existent for evaluation);
                gt_exists = gt_norm is not None and gt_norm != ""
                pred_exists = pred_norm is not None and pred_norm != ""
                # Check if the existing values match
                match = gt_exists and pred_exists and (gt_norm == pred_norm)
                # Calculate TP, FP, FN based on existence and match
                tp = 1 if match else 0
                fp = 1 if pred_exists and not match else 0
                fn = 1 if gt_exists and not match else 0

            # Accumulate stats for the field across all items
            field_stats[field]["tp"] += tp
            field_stats[field]["fp"] += fp
            field_stats[field]["fn"] += fn

    # --- Calculate Metrics ---
    per_field_metrics = {}
    for field in EVALUATED_FIELDS:
        stats = field_stats[field]
        metrics = calculate_metrics(stats["tp"], stats["fp"], stats["fn"])
        per_field_metrics[field] = {
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"],
            **metrics
        }

    # --- Calculate Original Macro-Averaged Metrics (All Fields) ---
    overall_macro_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    num_fields_evaluated = len(EVALUATED_FIELDS)
    if num_fields_evaluated > 0:
        total_precision = sum(metrics["precision"] for metrics in per_field_metrics.values())
        total_recall = sum(metrics["recall"] for metrics in per_field_metrics.values())
        total_f1 = sum(metrics["f1"] for metrics in per_field_metrics.values())
        overall_macro_metrics["precision"] = total_precision / num_fields_evaluated
        overall_macro_metrics["recall"] = total_recall / num_fields_evaluated
        overall_macro_metrics["f1"] = total_f1 / num_fields_evaluated

    # --- Calculate Filtered Macro-Averaged Metrics (Populated Fields Only) ---
    # Identify fields with actual positive examples in the ground truth (TP + FN > 0)
    populated_fields = [
        field for field in EVALUATED_FIELDS
        if (field_stats[field]["tp"] + field_stats[field]["fn"]) > 0
    ]
    num_populated_fields = len(populated_fields)

    overall_macro_metrics_populated = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if num_populated_fields > 0:
        total_precision_pop = sum(per_field_metrics[field]["precision"] for field in populated_fields)
        total_recall_pop = sum(per_field_metrics[field]["recall"] for field in populated_fields)
        total_f1_pop = sum(per_field_metrics[field]["f1"] for field in populated_fields)

        overall_macro_metrics_populated["precision"] = total_precision_pop / num_populated_fields
        overall_macro_metrics_populated["recall"] = total_recall_pop / num_populated_fields
        overall_macro_metrics_populated["f1"] = total_f1_pop / num_populated_fields

    # Return structure including both sets of overall metrics
    return {
        "per_field_metrics": per_field_metrics,
        "overall_macro_metrics": overall_macro_metrics, # Original average over ALL fields
        "overall_macro_metrics_populated_fields": overall_macro_metrics_populated # Filtered average
    }
    
def run_evaluation(
    model_class,
    eval_fn=evaluate_list_field_fuzzy,
    eval_kwargs=None,
    gt_path: str = None,
    pred_path: str = None,
) -> Dict[str, Any]:
    if not gt_path or not pred_path:
        raise ValueError("Both ground truth and predicted file paths must be provided.")

    # Load and validate ground truth
    if gt_path.endswith(".csv"):
        gt_raw = load_and_preprocess_csv_pandas(gt_path)
    else:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_raw = json.load(f)

    # --- Load predictions ---
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_raw = json.load(f)

    # --- Validate using model class ---
    ground_truth_items = [model_class.model_validate(item) for item in gt_raw]
    predicted_items = [model_class.model_validate(item) for item in pred_raw]

    # --- Evaluate ---
    evaluation_results = evaluate_items(
        ground_truth_items,
        predicted_items,
        eval_fn=eval_fn,
        **eval_kwargs,
    )

    return evaluation_results



# --- Example Usage ---
if __name__ == '__main__':
    # 1. Define your Ground Truth Data (as list of dicts or Item objects)
    #    Replace this with loading your actual ground truth data
    # ground_truth_raw_data = [
    #     {"base_type": "Instrument", "specific_type": "Acoustic Guitar", "music_artists": ["Lonnie Donegan", "Van Morrison"], "year": ["1958"], "quantity": 1, "materials": ["wood", "metal"]},
    #     {"base_type": "Recording", "specific_type": "LP", "music_artists": ["Ken Colyer's Jazzmen"], "year": ["1954"], "music_works": ["Back To The Delta"], "music_genres": ["Jazz", "Trad Jazz"]},
    #     {"base_type": "Ephemera", "specific_type": "Poster", "music_artists": ["Chas McDevitt Skiffle Group"], "venues": ["Metropolitan Club"], "cities": ["London"], "year": ["1957"], "colours": ["black", "white"]},
    #     {"base_type": "Instrument", "specific_type": "Tea chest bass", "materials": ["wood", "string", "broom handle"]},
    #     {"base_type": "Recording", "specific_type": "EP", "music_artists": ["Lonnie Donegan"], "year": ["1954"], "music_works": ["Rock Island Line", "John Henry"], "quantity": None}, # Explicit None quantity
    #     {"base_type": "Clothing", "specific_type": "Suit", "music_artists": ["Lonnie Donegan"], "colours": ["black"]}
    # ]

    ground_truth_raw_data = load_and_preprocess_csv_pandas("inputs/item_descriptions_with_captions_green_1.csv")

    # 2. Define your LLM Predicted Data (as list of dicts or Item objects)
    #    Replace this with loading your actual LLM output data
    # predicted_raw_data = [
    #     {"base_type": "Instrument", "specific_type": "acoustic guitar", "music_artists": ["Lonnie Donegan"], "year": ["1958"], "quantity": 1, "materials": ["Wood"]}, # Norm diff, missing artist
    #     {"base_type": "Recording", "specific_type": "LP", "music_artists": ["Ken Colyer"], "year": ["1954"], "music_works": ["Back To The Delta"], "music_genres": ["Jazz"], "countries": ["UK"]}, # Artist variation, missing genre, extra country
    #     {"base_type": "Ephemera", "specific_type": "Concert Poster", "music_artists": ["Chas McDevitt"], "venues": ["Metropolitan Club"], "cities": ["London", "Edgware"], "year": ["1957"], "colours": ["black"]}, # Type variation, extra city, missing artist detail, missing colour
    #     {"base_type": "Instrument", "specific_type": "Bass", "materials": ["wood"], "quantity": 1}, # Less specific type, missing materials, extra quantity
    #     {"base_type": "Recording", "specific_type": "EP", "music_artists": ["Lonnie Donegan"], "year": ["1954", "1955"], "music_works": ["Rock Island Line"], "quantity": 0}, # Missing work, extra year, wrong quantity
    #     {"base_type": "Apparel", "specific_type": "Suit", "music_artists": ["Donegan"], "colours": ["Black", "White"]} # Diff base_type, artist var, case diff colour, extra colour
    # ]

    with open("outputs/response_20250708_155953.json", "r") as f:
        predicted_raw_data = json.load(f)

    # 3. Validate data using the Pydantic model (good practice)
    #    Use model_validate for Pydantic v2, or parse_obj for v1
    try:
        ground_truth_items = [AnnotationItem.model_validate(item) for item in ground_truth_raw_data]
        predicted_items = [AnnotationItem.model_validate(item) for item in predicted_raw_data]
        # If using Pydantic v1:
        # ground_truth_items = [Item.parse_obj(item) for item in ground_truth_raw_data]
        # predicted_items = [Item.parse_obj(item) for item in predicted_raw_data]
    except Exception as e:
        print(f"Error validating data with Pydantic model: {e}")
        # Handle error appropriately, e.g., exit or log
        exit()


    # 4. Run the evaluation
    evaluation_results = evaluate_items(ground_truth_items, predicted_items)

    # 5. Print the results
    print("--- Evaluation Results ---")
    # Use json.dumps for pretty printing the nested dictionary
    print(json.dumps(evaluation_results, indent=2))

    print("\n--- Overall Macro Metrics ---")
    for metric, value in evaluation_results["overall_macro_metrics"].items():
        print(f"{metric.capitalize():<10}: {value:.4f}") # Formatted output

    print("\n--- Per-Field Metrics ---")
    for field, metrics in evaluation_results["per_field_metrics"].items():
        print(f"\nField: {field}")
        print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")