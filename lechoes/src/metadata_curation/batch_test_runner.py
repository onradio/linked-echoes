from evaluation_pipeline import run_pipeline, nest_keys  # Your local module
import pandas as pd
from datetime import datetime
import traceback

# Define test cases
test_cases = [
    {
        "overrides": {
            "gemini.temperature": 0,
            "evaluation.type": "case_by_case",
            "prompt_mode": "zero-shot",
            "role_block.variant" : "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 0,
            "evaluation.type": "case_by_case",
            "prompt_mode": "one-shot",
            "role_block.variant" : "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 0,
            "evaluation.type": "case_by_case",
            "prompt_mode": "few-shot",
            "role_block.variant" : "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 0,
            "evaluation.type": "row_by_row",
            "prompt_mode": "zero-shot",
            "role_block.variant": "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 0,
            "evaluation.type": "row_by_row",
            "prompt_mode": "one-shot",
            "role_block.variant": "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 0,
            "evaluation.type": "row_by_row",
            "prompt_mode": "few-shot",
            "role_block.variant": "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 0.5,
            "evaluation.type": "case_by_case",
            "prompt_mode": "zero-shot",
            "role_block.variant" : "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 0.5,
            "evaluation.type": "case_by_case",
            "prompt_mode": "one-shot",
            "role_block.variant" : "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 0.5,
            "evaluation.type": "case_by_case",
            "prompt_mode": "few-shot",
            "role_block.variant" : "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 0.5,
            "evaluation.type": "row_by_row",
            "prompt_mode": "zero-shot",
            "role_block.variant": "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 0.5,
            "evaluation.type": "row_by_row",
            "prompt_mode": "one-shot",
            "role_block.variant": "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 0.5,
            "evaluation.type": "row_by_row",
            "prompt_mode": "few-shot",
            "role_block.variant": "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 1,
            "evaluation.type": "case_by_case",
            "prompt_mode": "zero-shot",
            "role_block.variant" : "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 1,
            "evaluation.type": "case_by_case",
            "prompt_mode": "one-shot",
            "role_block.variant" : "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 1,
            "evaluation.type": "case_by_case",
            "prompt_mode": "few-shot",
            "role_block.variant" : "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 1,
            "evaluation.type": "row_by_row",
            "prompt_mode": "zero-shot",
            "role_block.variant": "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 1,
            "evaluation.type": "row_by_row",
            "prompt_mode": "one-shot",
            "role_block.variant": "strict_archivist"
        }
    },
    {
        "overrides": {
            "gemini.temperature": 1,
            "evaluation.type": "row_by_row",
            "prompt_mode": "few-shot",
            "role_block.variant": "strict_archivist"
        }
    },
    # Add more test cases if needed...
]

# Hardcoded field names
field_names = [
    "base_type", "specific_type", "music_genres", "music_works",
    "music_albums", "music_artists", "events", "countries", "places",
    "venues", "years", "colours", "materials", "item_title", "cardinality"
]
field_metrics_keys = ["precision", "recall", "f1", "tp", "fp", "fn"]

aggregated_metrics_keys = ["precision", "recall", "f1"]

# Column layout
column_tuples = [
    ("meta", "test_name"),
    ("meta", "timestamp"),
    ("meta", "overrides"),
    ("meta", "config_used"),
    ("meta", "gt_count"),
    ("meta", "pred_count"),
    ("meta", "gt_errors"),
    ("meta", "pred_errors"),
]

# Add macro metrics before field-level breakdown
column_tuples += [("overall_macro_metrics", metric) for metric in aggregated_metrics_keys]
column_tuples += [("overall_macro_metrics_populated_fields", metric) for metric in aggregated_metrics_keys]

# Add per-field metrics
for field in field_names:
    for metric in field_metrics_keys:
        column_tuples.append((field, metric))

# Flatten config for readability
def flatten_config(config_dict):
    return "; ".join([f"{k}={v}" for k, v in config_dict.items()])


def build_testcase_name(test_case):
    overrides = test_case.get("overrides", {})
    values = [str(v) for v in overrides.values()]
    return "__".join(values) if values else "default"


summary_rows = []
df_errors = pd.DataFrame(columns=["Test Case", "Error"])

for test_case in test_cases:
    overrides = nest_keys(test_case["overrides"])
    test_case_name = build_testcase_name(test_case)
    test_case["name"] = f"test_case__{test_case_name}"

    print(f"test_case_name: {test_case_name}")
    try:
        # Run the pipeline with the current test case overrides
        trace, pipeline_errors = run_pipeline("config/test_config.yaml", overrides=overrides)
        if not pipeline_errors.empty:
            # Add Test Case column to errors
            pipeline_errors["Test Case"] = test_case_name

            print(f"pipeline_errors : {pipeline_errors}")

            # Append to master df_errors
            df_errors = pd.concat([df_errors, pipeline_errors[["Test Case", "Error"]]], ignore_index=True)
        else:
            eval_results = trace.get("evaluation_results", {})
            print(f"Test case {test_case_name} ran successfully")
            row = {}
            # Meta
            row[("meta", "test_name")] = test_case["name"]
            row[("meta", "timestamp")] = trace.get("timestamp")
            row[("meta", "overrides")] = flatten_config(trace.get("overrides", {}))
            row[("meta", "config_used")] = flatten_config(trace.get("config_used", {}))
            row[("meta", "gt_count")] = trace["validation_summary"]["ground_truth_count"]
            row[("meta", "pred_count")] = trace["validation_summary"]["predicted_count"]
            row[("meta", "gt_errors")] = len(trace["validation_summary"]["gt_validation_errors"])
            row[("meta", "pred_errors")] = len(trace["validation_summary"]["pred_validation_errors"])

            # Overall macro metrics
            macro = eval_results.get("overall_macro_metrics", {})
            for metric in aggregated_metrics_keys:
                row[("overall_macro_metrics", metric)] = round(macro.get(metric, 0.0), 2)

            macro_pop = eval_results.get("overall_macro_metrics_populated_fields", {})
            for metric in aggregated_metrics_keys:
                row[("overall_macro_metrics_populated_fields", metric)] = round(macro_pop.get(metric, 0.0), 2)

            # Per-field metrics
            per_field = eval_results.get("per_field_metrics", {})
            for field in field_names:
                field_metrics = per_field.get(field, {})
                for metric in field_metrics_keys:
                    row[(field, metric)] = round(field_metrics.get(metric, 0.0), 2)
            # Append the row to summary_rows
            summary_rows.append(row)
    except Exception as e:
        print(f"An error occurred while running test case {test_case_name} : {e}")
        traceback.print_exc()
        if test_case in df_errors["Test Case"].values:
            # Append to existing list of errors
            df_errors.loc[df_errors["Test Case"] == test_case, "Errors"].iloc[0].extend(str(e))
        else:
            # Create new row with errors as a list
            df_errors.loc[len(df_errors)] = [test_case, str(e)]



print(f"summary_rows : {len(summary_rows)}")

# Create DataFrame with MultiIndex columns
columns = pd.MultiIndex.from_tuples(column_tuples)
df = pd.DataFrame(summary_rows, columns=columns)
df = df.dropna(how="all")

file_name = f'evaluation/evaluation_summary_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.xlsx'
print(f"file_name : {file_name}")
# Save to Excel — index=True required for MultiIndex columns
df.to_excel(file_name, index=True, engine="openpyxl", sheet_name="Processed_Data")


with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
    df_errors.to_excel(writer, index=False, sheet_name="Error_Log")
