import json
import logging
import os
import time
from datetime import datetime

import google.generativeai as genai

logger = logging.getLogger(__name__)

from evaluation import AnnotationItem
from utils.config_loader import load_config
from utils.main_utils import clean_schema_for_gemini

PRESETS = {
    "strict": {
        "temperature": 0.1,
        "top_p": 0.85,
        "top_k": 20,
        "max_output_tokens": 512
    },
    "balanced": {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 1024
    },
    "creative": {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 100,
        "max_output_tokens": 1024
    },
    "bulk": {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 100,
        "max_output_tokens": 8192
    }
}


class GeminiHandler:
    def __init__(self, mode=None, **overrides):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        full_config = load_config(default_path=os.path.join(os.path.dirname(__file__), "config", "default_config.yaml"))

        gemini_config = full_config.get("gemini", {})

        # Mode resolution: CLI > YAML > default
        mode = mode or gemini_config.get("mode", "balanced")

        config = PRESETS.get(mode, PRESETS["balanced"]).copy()

        # YAML-level overrides
        for key in ["temperature", "top_p", "top_k", "max_output_tokens"]:
            if gemini_config.get(key) is not None:
                config[key] = gemini_config[key]

        if overrides.get('overrides', {}).get('gemini'):
            # The inner dictionary you actually want to use for the update
            inner_overrides = overrides['overrides']['gemini']

            # CLI-level overrides
            config.update(inner_overrides)

        self.model_name = config.get("model_name",
                                        gemini_config.get("model_name", os.getenv("GEMINI_MODEL", "gemini-2.0-flash")))

        self.temperature = config["temperature"]
        self.top_p = config["top_p"]
        self.top_k = config["top_k"]
        self.max_output_tokens = config["max_output_tokens"]

        print(f"Using Gemini config: {mode} | "
              f"temperature={self.temperature}, top_p={self.top_p}, "
              f"max_tokens={self.max_output_tokens}")

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_output_tokens,
            }
        )

        # Prepare function tool schema
        schema = AnnotationItem.model_json_schema()

        clean_single_schema = clean_schema_for_gemini(schema)

        wrapper_bulk_schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": clean_single_schema["properties"],
                        "required": clean_single_schema.get("required", [])
                    }
                }
            },
            "required": ["items"]
        }

        # Tool for single item
        self.tool_single = {
            "function_declarations": [{
                "name": "record_item_data",
                "description": "Record a single item of music memorabilia",
                "parameters": clean_single_schema
            }]
        }

        # Tool for bulk items
        self.tool_bulk = {
            "function_declarations": [{
                "name": "record_item_data",
                "description": "Record a list of music memorabilia items",
                "parameters": wrapper_bulk_schema
            }]
        }

    @staticmethod
    def save_to_file(data, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"response_{timestamp}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in data], f, indent=2, ensure_ascii=False)

    def extract(self, prompt: str) -> AnnotationItem:
        tool_config = {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["record_item_data"]
            }
        }

        logger.info("Sending prompt to Gemini:\n%s", prompt)

        response = self.model.generate_content(
            prompt,
            tools=[self.tool_single],
            tool_config=tool_config
        )

        logger.info("Raw Gemini response:\n%s", response)

        time.sleep(3)

        return self.parse_single_response(response)

    def extract_bulk(self, prompt: str) -> list[AnnotationItem]:
        """
        Processes a bulk prompt using Gemini function-calling API and extracts structured AnnotationItem records.
        """
        tool_config = {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["record_item_data"]
            }
        }

        response = self.model.generate_content(
            prompt,
            tools=[self.tool_bulk],
            tool_config=tool_config
        )

        return self.parse_bulk_response(response)

    def get_model_info(self):
        return self.model_name

    @staticmethod
    def parse_single_response(response) -> AnnotationItem:
        """Parse a single Gemini function-call response into an AnnotationItem."""
        try:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)

            if not hasattr(candidate, "content") or not candidate.content.parts:
                logger.warning(
                    "Gemini returned no content parts. finish_reason=%s\nFull response:\n%s",
                    finish_reason, response
                )
                raise ValueError(
                    f"Gemini returned no content (finish_reason={finish_reason}). "
                    "This may indicate a malformed function call or safety block."
                )

            part = candidate.content.parts[0]
            if part.function_call and part.function_call.name == "record_item_data":
                args = dict(part.function_call.args)
                if 'cardinality' in args and isinstance(args['cardinality'], str):
                    args['cardinality'] = int(args['cardinality'])
                return AnnotationItem(**args)
            else:
                logger.warning("Gemini returned plain text instead of function call:\n%s", response.text)
                raise ValueError("No valid function call in single response.")
        except (AttributeError, IndexError) as e:
            logger.error("Error parsing single response: %s\nFull response:\n%s", e, response)
            raise ValueError(f"Error parsing single response: {e}")

    @staticmethod
    def parse_bulk_response(response) -> list[AnnotationItem]:
        """Parse a bulk Gemini function-call response into a list of AnnotationItems."""
        try:
            #print(f"response : {response}")
            parts = response.candidates[0].content.parts
            items = []

            #print(f"parts : {parts}")
            for idx, part in enumerate(parts):
                if not hasattr(part, "function_call") or part.function_call.name != "record_item_data":
                    continue

                args = dict(part.function_call.args)

                # Normalize numeric fields
                for i, item_dict in enumerate(args.get("items", [])):
                    if isinstance(item_dict.get("cardinality"), str) and item_dict["cardinality"].isdigit():
                        item_dict["cardinality"] = int(item_dict["cardinality"])
                    try:
                        items.append(AnnotationItem(**item_dict))
                    except Exception as e:
                        print(f"Validation failed for item {i + 1}: {e}")
                        continue

            if not items:
                raise ValueError("No valid structured outputs found in bulk response.")
            return items
        except Exception as e:
            raise ValueError(f"Error parsing bulk response: {e}")
