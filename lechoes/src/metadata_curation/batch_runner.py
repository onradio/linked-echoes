"""
batch_runner.py

Runs the metadata extraction pipeline with a fixed configuration.
No evaluation or ground-truth comparison is performed — outputs are
saved directly to the outputs/ directory as JSON.
"""

import csv
import json
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

from evaluation_pipeline import nest_keys, flatten_nested_list
from batch_gemini_handler import BatchGeminiHandler
from gemini_handler import GeminiHandler
from generate_prompt import generate_prompt
from utils.config_loader import load_config
from utils.dataset import load_and_preprocess_csv_pandas

# ---------------------------------------------------------------------------
# Fixed configuration — edit these values directly rather than via overrides
# ---------------------------------------------------------------------------
FIXED_CONFIG_OVERRIDES = {
    "gemini.model_name": "gemini-2.5-flash",
    "gemini.temperature": 0,
    "gemini.top_p": 0.95,
    "gemini.top_k": 80,
    "gemini.max_output_tokens": 8000,
    "prompt_mode": "few-shot",
    "role_block.variant": "strict_archivist",
    "instruction_block.variant": "strict_schema_extraction",
    "input_block.mode": "sync",
    "input_block.independent_mode": True,
    "output_block.variant": "tool_call",
    "evaluation.ground_truth_path": "inputs/item_descriptions_with_captions_all_but_examples.csv",
    "evaluation.type": "row_by_row",
    "evaluation.n_rows": 1,
}

BASE_CONFIG_PATH = "config/default_config.yaml"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def count_bulk_items(prompt_text: str) -> int:
    return len(re.findall(r"\*\*Item\s+\d+\*\*", prompt_text))


def run_batch_inference():
    overrides = nest_keys(FIXED_CONFIG_OVERRIDES)
    config = load_config(BASE_CONFIG_PATH, overrides_dict=overrides)

    evaluation_type = config.get("evaluation", {}).get("type", "row_by_row")
    input_mode = config.get("input_block", {}).get("mode", "sync")
    input_path = config.get("evaluation", {}).get("ground_truth_path")
    n_rows = config.get("evaluation", {}).get("n_rows")

    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Load input data (descriptions + captions only — no GT labels used)
    raw_data = load_and_preprocess_csv_pandas(
        csv_file_path=input_path,
        subset=None,
        n_rows=n_rows,
        evaluation_type=evaluation_type,
    )

    if evaluation_type == "case_by_case":
        selected_columns_data = [
            [
                {"item_description": row.get("item_description"), "caption": row.get("caption")}
                for row in case_group
            ]
            for case_group in raw_data
        ]
    else:
        selected_columns_data = [
            {"item_description": item.get("item_description"), "caption": item.get("caption")}
            for item in raw_data
        ]

    # Step 2: Generate prompts
    prompt_outputs = generate_prompt(
        config_path=Path("config/default_config.yaml"),
        overrides=config,
        input_items=selected_columns_data,
    )

    # Step 3: Run inference
    predicted = []
    errors = []

    if input_mode == "sync":
        handler = GeminiHandler(overrides=config)

        if evaluation_type == "case_by_case":
            print("Running Gemini in BULK mode...")
            for idx, (prompt_text, filename) in enumerate(prompt_outputs):
                expected_count = count_bulk_items(prompt_text)
                success = False
                for attempt in range(MAX_RETRIES):
                    try:
                        results = handler.extract_bulk(prompt_text)
                        if len(results) != expected_count:
                            raise ValueError(
                                f"Expected {expected_count} responses, got {len(results)}"
                            )
                        predicted.append([item.model_dump() for item in results])
                        success = True
                        break
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            print(f"Prompt {idx} attempt {attempt + 1} failed: {e}. Retrying...")
                            time.sleep(RETRY_DELAY)
                        else:
                            msg = f"Prompt {idx}: {e}"
                            print(f"All retries failed — {msg}")
                            errors.append(msg)
                if not success:
                    predicted.append([])  # placeholder to keep index alignment
        else:
            print("Running Gemini in SINGLE mode...")
            for idx, (prompt_text, filename) in enumerate(prompt_outputs):
                for attempt in range(MAX_RETRIES):
                    try:
                        item = handler.extract(prompt_text)
                        predicted.append(item.model_dump())
                        break
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            print(f"Prompt {idx} attempt {attempt + 1} failed: {e}. Retrying...")
                            time.sleep(RETRY_DELAY)
                        else:
                            msg = f"Prompt {idx}: {e}"
                            print(f"All retries failed — {msg}")
                            errors.append(msg)

    elif input_mode == "batch":
        print("Running Gemini in BATCH mode...")
        batch_handler = BatchGeminiHandler(overrides=config)
        prompts = [p[0] for p in prompt_outputs]
        try:
            results = batch_handler.run_batch(prompts)
            predicted = [[item.model_dump() for item in sublist] for sublist in results]
        except Exception as e:
            traceback.print_exc()
            errors.append(str(e))
    else:
        raise ValueError(f"Unsupported input_mode: '{input_mode}'")

    # Step 4: Flatten and save
    predicted = flatten_nested_list(predicted)

    output_path = output_dir / f"response_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predicted, f, indent=2)
    print(f"Saved {len(predicted)} predictions to {output_path}")

    if errors:
        error_path = output_dir / f"errors_{timestamp}.json"
        with error_path.open("w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        print(f"{len(errors)} errors logged to {error_path}")

    # Step 5: Merge all JSON files in outputs/ into a single CSV
    merge_jsons_to_csv(output_dir)

    return predicted, errors


CSV_COLUMNS = [
    "description", "case_number", "exhibition", "category", "type", "name",
    "item_model", "genre", "music_work", "music_album", "music_artist",
    "event", "country", "place", "venue", "year", "colour", "material", "caption",
]

# Maps CSV column → JSON field (or None if not present in JSON)
_JSON_FIELD_MAP = {
    "description": "item_description",
    "category":    "base_type",
    "type":        "specific_type",
    "name":        "item_title",
    "genre":       "music_genres",
    "music_work":  "music_works",
    "music_album": "music_albums",
    "music_artist":"music_artists",
    "event":       "events",
    "country":     "countries",
    "place":       "places",
    "venue":       "venues",
    "year":        "years",
    "colour":      "colours",
    "material":    "materials",
}


def _record_to_row(record: dict) -> dict:
    row = {col: "" for col in CSV_COLUMNS}
    for csv_col, json_field in _JSON_FIELD_MAP.items():
        value = record.get(json_field, "")
        if isinstance(value, list):
            value = " | ".join(str(v) for v in value if v)
        row[csv_col] = value if value is not None else ""
    return row


def merge_jsons_to_csv(output_dir: Path) -> Path:
    json_files = sorted(output_dir.glob("response_*.json"))
    if not json_files:
        print("No response JSON files found to merge.")
        return None

    all_rows = []
    for jf in json_files:
        with jf.open(encoding="utf-8") as f:
            records = json.load(f)
        if isinstance(records, list):
            all_rows.extend(_record_to_row(r) for r in records)

    csv_path = output_dir / "merged_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Merged {len(all_rows)} records from {len(json_files)} file(s) → {csv_path}")
    return csv_path


if __name__ == "__main__":
    predictions, errors = run_batch_inference()
    print(f"\nDone. {len(predictions)} predictions, {len(errors)} errors.")
