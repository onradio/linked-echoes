import argparse
import functools
import json
from datetime import datetime
from pathlib import Path
from typing import Type, Tuple, List, Any
import re
from pydantic import BaseModel, ValidationError

from batch_gemini_handler import BatchGeminiHandler
import pandas as pd
from evaluation import evaluate_items, AnnotationItem
from gemini_handler import GeminiHandler
from generate_prompt import generate_prompt
from utils.config_loader import load_config
from utils.dataset import load_and_preprocess_csv_pandas


def nest_keys(flat_dict):
    nested = {}
    for flat_key, value in flat_dict.items():
        keys = flat_key.split(".")
        d = functools.reduce(lambda d, k: d.setdefault(k, {}), keys[:-1], nested)
        d[keys[-1]] = value
    return nested


def flatten_nested_list(nested_data):
    """
    Flatten a nested list structure to a simple List[Dict]

    Args:
        nested_data: List of lists containing dictionaries

    Returns:
        List[Dict]: Flattened list of all dictionaries
    """
    flattened = []

    for sublist in nested_data:
        if isinstance(sublist, list):
            # Extend the flattened list with all items from the sublist
            flattened.extend(sublist)
        else:
            # If it's not a list, add the item directly
            flattened.append(sublist)

    return flattened

def validate_items_with_logging(raw_data: List[dict], model: Type[BaseModel], label: str = "input", id_field: str = "item_title") -> Tuple[List[Any], List[dict]]:
    """
        Validates a list of dicts against a Pydantic model.
        Logs detailed errors with index, content, and reasons.

        Args:
            raw_data: List of dicts to validate.
            model: The Pydantic model to validate.
            label: Descriptive label (e.g., 'ground_truth').
            id_field: Optional field to help identify rows (e.g., item_title).

        Returns:
            Tuple of (validated_items, failed_items_with_errors)
        """
    validated_items = []
    failed_items = []

    for i, row in enumerate(raw_data):
        try:
            validated_items.append(model.model_validate(row))
        except ValidationError as e:
            row_id = row.get(id_field, f"row_{i}")
            error_details = {
                "index": i,
                "row_id": row_id,
                "raw_data": row,
                "errors": [
                    {"field": err.get("loc", []), "reason": err.get("msg", ""), "type": err.get("type", "")}
                    for err in e.errors()
                ]
            }
            print(f"Validation failed for {label} [{row_id}] at index {i}:")
            for err in error_details["errors"]:
                print(f"  - Field: {err['field']} | Reason: {err['reason']}")
            failed_items.append(error_details)

    if failed_items:
        Path("logs").mkdir(exist_ok=True)
        out_path = Path(f"logs/{label}_validation_errors.json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(failed_items, f, indent=2, ensure_ascii=False)
        print(f"{len(failed_items)} validation errors logged to: {out_path}")

    return validated_items, failed_items


def parse_overrides(override_list):
    override_dict = {}
    if override_list:
        for entry in override_list:
            if "=" in entry:
                key, value = entry.split("=", 1)
                if value.isdigit():
                    value = int(value)
                elif value.lower() in {"true", "false"}:
                    value = value.lower() == "true"
                override_dict[key] = value
    return override_dict



def count_bulk_items(prompt_text: str) -> int:
    """
    Counts the number of bulk entries in a prompt.
    Bulk entries are marked like: **Item 1**, **Item 2**, etc.
    Returns 0 if no bulk markers are found.
    """
    return len(re.findall(r"\*\*Item\s+\d+\*\*", prompt_text))

def run_pipeline(test_config_path: str, overrides: dict = None):
    config = load_config(test_config_path, overrides_dict=overrides)
    evaluation_type = config.get("evaluation", {}).get("type", "row_by_row")
    input_mode = config.get("input_block", {}).get("mode", "sync")

    pipeline_errors = pd.DataFrame(columns=["Error"])

    output_dir = Path(config.get("output_dir", "outputs"))
    trace_dir = Path(config.get("trace_dir", "traces"))
    output_dir.mkdir(exist_ok=True, parents=True)
    trace_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    eval_config = config.get("evaluation", {})
    n_rows = eval_config.get("n_rows")
    gt_path = eval_config.get("ground_truth_path")

    # Step 1: Load and preprocess ground truth data
    ground_truth_raw_data = load_and_preprocess_csv_pandas(
        csv_file_path=gt_path,
        subset=None,
        n_rows=n_rows,
        evaluation_type=evaluation_type
    )

    # Split the ground truth data into selected columns based on evaluation type
    if evaluation_type == "case_by_case":
        # Already grouped in load_and_preprocess_csv_pandas
        selected_columns_data = [
            [
                {'item_description': row.get('item_description'), 'caption': row.get('caption')}
                for row in case_group
            ]
            for case_group in ground_truth_raw_data
        ]
    else:
        selected_columns_data = [
            {
                'item_description': item.get('item_description'),
                'caption': item.get('caption')
            }
            for item in ground_truth_raw_data
        ]


    # Step 2: Generate prompts
    prompt_outputs = generate_prompt(
        config_path=Path("config/default_config.yaml"),
        overrides=config,
        input_items=selected_columns_data
    )

    # Initialize predicted list to store results
    # This will hold the final output from the Gemini model
    # It will be a list of dictionaries, each representing a response for a prompt
    predicted = []
    if input_mode == "sync":
        if evaluation_type == "case_by_case":
            print("Running Gemini in BULK mode...")
            handler = GeminiHandler(overrides=config)
            #print(f"prompt_outputs : {prompt_outputs}")
            for idx, (prompt_text, filename) in enumerate(prompt_outputs):
                max_retries = 3
                retry_delay = 2
                expected_count = count_bulk_items(prompt_text)
                import time
                # Adding fixed and finite retry logic
                for attempt in range(max_retries):
                    #print(f"attempt :{attempt}; prompt_text : {prompt_text}")
                    try:
                        results = handler.extract_bulk(prompt_text)

                        # Custom check: ensure we got the expected number of responses
                        if len(results) != expected_count:
                            print(f"results with wrong count : {results}")
                            raise ValueError(
                                f"Only {len(results)} responses received, expected {expected_count}"
                            )

                        # If everything is okay, process results and break out of retry loop
                        predicted.append([item.model_dump() for item in results])
                        break

                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Attempt {attempt + 1} failed for prompt: {prompt_text}. Retrying...")
                            time.sleep(retry_delay)
                        else:
                            error_msg = f"{prompt_text}: {str(e)}"
                            pipeline_errors.loc[len(pipeline_errors)] = [error_msg]
                            print(f"All retries failed for prompt: {prompt_text}. Logged to pipeline_errors.")
        else:
            print("Running Gemini in SINGLE mode...")
            handler = GeminiHandler(overrides=config)
            import time

            max_retries = 3
            retry_delay = 2

            for idx, (prompt_text, filename) in enumerate(prompt_outputs):
                # Adding fixed and finite retry logic
                for attempt in range(max_retries):
                    try:
                        item = handler.extract(prompt_text)
                        predicted.append(item.model_dump())
                        break

                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Attempt {attempt + 1} failed for prompt {idx}: {prompt_text[:50]}... Retrying...")
                            time.sleep(retry_delay)
                        else:
                            error_msg = f"{prompt_text}: {str(e)}"
                            pipeline_errors.loc[len(pipeline_errors)] = [error_msg]
                            print(f"All retries failed for prompt {idx}. Logged to pipeline_errors.")



    elif input_mode == "batch":
        print("Running Gemini in BATCH mode...")
        batch_handler = BatchGeminiHandler( overrides=config)
        prompts = [p[0] for p in prompt_outputs]
        try:
            results = batch_handler.run_batch(prompts)
            predicted = [[item.model_dump() for item in sublist] for sublist in results]
        except Exception as e:
            pipeline_errors.loc[len(pipeline_errors)] = [
                " | ".join(prompts),  # join list of prompts
                str(e)
            ]
    else:
        raise ValueError(f"Unsupported input_mode: '{input_mode}'")


    output_path = output_dir / f"response_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predicted, f, indent=2)


    # Step 2.5: Flatten nested lists
    ground_truth_raw_data = flatten_nested_list(ground_truth_raw_data)
    predicted = flatten_nested_list(predicted)

    #print(f"ground_truth_raw_data : {ground_truth_raw_data} , predicted : {predicted}")

    # Step 3: Validation + Evaluation
    ground_truth_items, gt_errors = validate_items_with_logging(ground_truth_raw_data, AnnotationItem, label="ground_truth")
    predicted_items, pred_errors = validate_items_with_logging(predicted, AnnotationItem, label="predicted")

    trace = {}

    try:
        eval_results = evaluate_items(ground_truth_items, predicted_items)
        # Step 4: Trace
        trace = {
            "timestamp": timestamp,
            "overrides": overrides,
            "prompt_path": [p[0] for p in prompt_outputs],
            "output_path": str(output_path),
            "config_used": config,
            "input_mode": input_mode,
            "evaluation_type": evaluation_type,
            "prompt_mode": config["prompt_mode"],
            "n_cases_or_rows": len(selected_columns_data),
            "output_variant": config.get("output_block", {}).get("variant"),
            "evaluation_results": eval_results,
            "validation_summary": {
                "ground_truth_count": len(ground_truth_items),
                "predicted_count": len(predicted_items),
                "gt_validation_errors": gt_errors,
                "pred_validation_errors": pred_errors
            }
        }
    except Exception as e:
        pipeline_errors.loc[len(pipeline_errors)] = [str(e)]



    # trace_path = trace_dir / f"trace_{timestamp}.json"
    # with trace_path.open("w", encoding="utf-8") as f:
    #     json.dump(trace, f, indent=2)
    return trace, pipeline_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full LLM prompt → output → eval pipeline")
    parser.add_argument("config_path", type=str, help="Path to test config YAML")
    parser.add_argument("--override", action="append", help="Override config values with --override key=value")
    args = parser.parse_args()

    flat_overrides = parse_overrides(args.override)
    nested_overrides = nest_keys(flat_overrides)

    run_pipeline(args.config_path, overrides=nested_overrides)
