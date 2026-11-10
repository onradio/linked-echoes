"""
The module provides functions to preprocess and clean data from a collection CSV
file, specifically designed for handling infer tags and pipe-separated values.
It includes functions to strip infer tags, split and trim strings, and load
data into a structured format suitable for further processing or validation.
"""
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union


def strip_annotation_tags(text: Any) -> Optional[str]:
    """Removes annotation tags, such as <infer> ... </infer> from text."""
    if pd.isna(text) or not isinstance(text, str):
        return None
    # Simplified regex for <infer> tags only
    cleaned_text = re.sub(r'<[/]?\s*infer\s*>', '', text)
    result = cleaned_text.strip()
    return result if result else None # Return None if empty after stripping


def split_trim_strip_tags(value: Any, delimiter: str = '|') -> Optional[List[str]]:
    """
    Strips infer tags, splits string by delimiter, trims parts, filters empty,
    returning a list or None.
    """
    # 1. Strip the annotation tags first
    value_no_tags = strip_annotation_tags(value)
    # 2. Proceed with splitting and trimming only if the value is valid
    if pd.isna(value_no_tags) or not isinstance(value_no_tags, str) or value_no_tags.strip() == '':
        return None
    # Strip outer whitespace from the potentially tag-stripped value before splitting
    value_stripped = value_no_tags.strip()
    parts = [part.strip() for part in value_stripped.split(delimiter)]
    cleaned_parts = [part for part in parts if part]
    return cleaned_parts if cleaned_parts else None


def get_first_split_item_stripped(value: Any, delimiter: str = '|') -> Optional[str]:
    """Strips infer tags, splits, trims, filters empty, returns first item or None."""
    # Apply the combined list processing function first
    cleaned_list = split_trim_strip_tags(value, delimiter)
    # Then take the first item if the list is not empty
    return cleaned_list[0] if cleaned_list else None





# Assuming these helper functions exist from your original context
def split_trim_strip_tags(cell_value):
    # Placeholder for your actual implementation
    if not isinstance(cell_value, str):
        return []
    items = [item.strip() for item in cell_value.split('|')]
    return [item for item in items if item]


def get_first_split_item_stripped(cell_value):
    # Placeholder for your actual implementation
    if not isinstance(cell_value, str):
        return ''
    return cell_value.split('|')[0].strip()


def load_and_preprocess_csv_pandas(
    csv_file_path: str,
    subset: str = None,
    n_rows: int = None,
    evaluation_type: str = "row_by_row"
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Loads and preprocesses data from a CSV file using Pandas.

    Args:
        csv_file_path: Path to the input CSV file.
        subset: Optional; filters rows where df['set'] == subset.
        n_rows: Optional; return only the first N rows.
        evaluation_type: 'row_by_row' or 'case_by_case'.
                         If 'case_by_case', groups data by 'case' column.

    Returns:
        A list of dictionaries (row-by-row) or a list of lists (grouped by case).
    """

    processed_data = []
    try:
        df = pd.read_csv(csv_file_path, quotechar='"', low_memory=False)

        # --- Filtering Section ---

        # 1. Apply subset filter first (if provided)
        if subset is not None:
            print(f"Filtering for subset: '{subset}'")
            df = df[df['set'] == subset].copy()  # Use .copy() to avoid SettingWithCopyWarning

        # 2. Apply row-level filtering on the potentially subsetted data
        #    This is more efficient as it happens before preprocessing.
        if n_rows is not None and n_rows > 0:
            # Otherwise, use n_rows to get the first N rows.
            print(f"Applying row limit: getting first {n_rows} rows.")
            df = df.iloc[:n_rows]

        # --- Preprocessing Steps (on the final filtered DataFrame) ---

        list_columns = [
            'music_genres', 'music_works', 'music_albums', 'music_artists',
            'events', 'countries', 'places', 'venues', 'years', 'colours', 'materials'
        ]
        string_columns = [
            'item_description', 'caption', 'item_title', 'base_type', 'specific_type'
        ]
        numeric_columns = ['case', 'cardinality']

        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(split_trim_strip_tags)

        if "specific_type" in df.columns:
            df["specific_type"] = df["specific_type"].apply(get_first_split_item_stripped)
            df['specific_type'] = df['specific_type'].replace('', np.nan)

        for col in string_columns[:-1]:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).str.strip()
                df[col] = df[col].replace('', np.nan)

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        # --- Finalise and export ---
        columns = [c for c in df.columns if c in string_columns + list_columns + numeric_columns]
        df_final = df[columns].copy().fillna(np.nan).replace([np.nan], [None])
        processed_data = df_final.to_dict(orient='records')

        # --- Output Formatting ---
        if evaluation_type == "case_by_case":
            if 'case' not in df_final.columns:
                raise ValueError("evaluation_type is 'case_by_case' but 'case' column not found in data")

            grouped = df_final.groupby('case')
            return [group.to_dict(orient='records') for _, group in grouped]

        return df_final.to_dict(orient='records')

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return []
    except Exception as e:
        print(f"An error occurred during CSV processing: {e}")
        import traceback
        traceback.print_exc()
        return []



def _format_raw_row_to_item_dict(row: Dict[str, Any]) -> Dict[str, str]:
    """
    Internal helper to extract and format 'item_description' and 'caption'
    from a single raw row dictionary.

    This function maps raw CSV row data to the standardized 'item' dictionary format.

    Args:
        row (dict): A dictionary representing a single row of data, typically
                    from csv.DictReader.

    Returns:
        dict: A new dictionary containing only 'item_description' and 'caption',
              with leading/trailing whitespace stripped.
    """
    return {
        'item_description': row.get('item_description', '').strip(),
        'caption': row.get('caption', '').strip()
    }

def get_formatted_item_by_index(raw_data_rows: List[Dict[str, Any]], index: int = 0) -> Dict[str, str]:
    """
    Retrieves and formats a single item's data from a list of raw data rows
    based on its 0-based index.

    This method acts as the 'single item' mode for the library.

    Args:
        raw_data_rows (list[dict]): A list of dictionaries, where each dictionary
                                     represents a raw row of data (e.g., from csv.DictReader).
        index (int): The 0-based index of the row to retrieve and format.
                     Defaults to 0 (the first row).

    Returns:
        dict: A dictionary representing the single formatted item.

    Raises:
        ValueError: If raw_data_rows is empty or index is out of bounds.
    """
    if not raw_data_rows:
        raise ValueError("Input 'raw_data_rows' list cannot be empty for single item extraction.")
    if not (0 <= index < len(raw_data_rows)):
        raise ValueError(f"Index {index} is out of bounds for data with {len(raw_data_rows)} rows.")

    return _format_raw_row_to_item_dict(raw_data_rows[index])


def get_formatted_items_in_range(raw_data_rows: List[Dict[str, Any]], start_index: int = 0, end_index: Union[int, None] = None) -> List[Dict[str, str]]:
    """
    Retrieves and formats a range of items for bulk processing from a list of raw data rows.

    This method acts as the 'bulk items' mode for the library.

    Args:
        raw_data_rows (list[dict]): A list of dictionaries, where each dictionary
                                     represents a raw row of data (e.g., from csv.DictReader).
        start_index (int): The 0-based starting index (inclusive) for the slice.
                           Defaults to 0 (the first row).
        end_index (int | None): The 0-based exclusive ending index for the slice.
                                If None, processes to the end of the raw_data_rows list.

    Returns:
        list[dict]: A list of formatted item dictionaries. Returns an empty list
                    if the input list is empty or the range is invalid.

    Raises:
        ValueError: If start_index is invalid.
    """
    if not raw_data_rows:
        # For bulk, returning an empty list for empty input is often acceptable
        # rather than raising an error, but depends on desired behavior.
        return []

    # Ensure start_index is valid. The slice `raw_data_rows[start:end]` handles
    # end out of bounds gracefully, so only check start.
    if not (0 <= start_index < len(raw_data_rows)):
        raise ValueError(f"Start index {start_index} is out of bounds for data with {len(raw_data_rows)} rows.")

    # Slice the raw data and format each row
    # Python slicing handles `end_index` being None or out of bounds correctly
    return [_format_raw_row_to_item_dict(row) for row in raw_data_rows[start_index:end_index]]