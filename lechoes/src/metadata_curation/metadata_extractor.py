"""
metadata_extractor.py

Single-entry metadata extraction class for embedding in a pipeline.
Wraps prompt generation + Gemini inference for one item at a time.
"""

import os
import time
from pathlib import Path
from typing import Optional

from evaluation_pipeline import nest_keys
from gemini_handler import GeminiHandler
from generate_prompt import generate_prompt
from utils.config_loader import load_config

DEFAULT_CONFIG_OVERRIDES = {
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
    "evaluation.type": "row_by_row",
}

BASE_CONFIG_PATH = str(Path(__file__).parent / "config" / "default_config.yaml")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class MetadataExtractor:
    """
    Extracts structured metadata from a single museum item description + caption.

    Usage:
        extractor = MetadataExtractor()
        result = extractor.extract({"item_description": "...", "caption": "..."})
    """

    def __init__(self, config_overrides: Optional[dict] = None, api_key: Optional[str] = None):
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        overrides = nest_keys({**DEFAULT_CONFIG_OVERRIDES, **(config_overrides or {})})
        self.config = load_config(BASE_CONFIG_PATH, overrides_dict=overrides)
        self.handler = GeminiHandler(overrides=self.config)

    def extract(self, item: dict) -> dict:
        """
        Run the full extraction pipeline for a single item.

        Args:
            item: dict with keys 'item_description' and 'caption'.

        Returns:
            Extracted metadata as a dict (AnnotationItem fields).

        Raises:
            RuntimeError: if all retries are exhausted.
        """
        _mc_dir = Path(__file__).parent
        prompt_text, _ = generate_prompt(
            config_path=Path(BASE_CONFIG_PATH),
            overrides=self.config,
            template_dir=_mc_dir / "templates",
            prompt_output_dir=_mc_dir / "prompts",
            input_items=[item],
        )[0]

        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                result = self.handler.extract(prompt_text)
                return result.model_dump()
            except Exception as e:
                last_exc = e
                if attempt < MAX_RETRIES - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(RETRY_DELAY)

        raise RuntimeError(f"Extraction failed after {MAX_RETRIES} attempts: {last_exc}")
