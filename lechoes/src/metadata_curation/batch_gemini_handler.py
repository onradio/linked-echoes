import ast
import json
import os
import re
import time
from typing import List

from google import genai

from evaluation import AnnotationItem
from utils.main_utils import clean_schema_for_gemini


class BatchGeminiHandler:
    """
    Handles Gemini batch jobs with inline responses.
    Submits a batch of prompts, waits for completion, fetches inline responses,
    and parses them into structured AnnotationItem objects.
    """

    def __init__(self, mode: str = "batch", overrides: dict = None):
        self.mode = mode
        self.overrides = overrides or {}
        self.model_name = self.overrides.get("gemini", {}).get("model_name", "models/gemini-2.5-flash")
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={'api_version': 'v1alpha'})
        print(f"[BatchGeminiHandler] Initialized with model: {self.model_name}")

        # Prepare function tool schema
        schema = AnnotationItem.model_json_schema()
        self.clean_single_schema = clean_schema_for_gemini(schema)

    # ---------------------------
    # 1. Submit batch job
    # ---------------------------
    def submit_batch_job(self, prompts: List[str]) -> str:
        """Submit a batch job using inline requests."""
        inline_requests = [
            {
                "contents": [
                    {"role": "user", "parts": [{"text": prompt}]}
                ]
            }
            for prompt in prompts
        ]

        batch_job = self.client.batches.create(
            model=self.model_name,
            src=inline_requests,
            config={"display_name": "metadata_curation_batch"}
        )

        print(f"[BatchGeminiHandler] Batch job submitted: {batch_job.name}")
        return batch_job.name

    # ---------------------------
    # 2. Wait for job completion
    # ---------------------------
    def wait_for_completion(self, job_name: str):
        """Polls until the batch job reaches a terminal state."""
        print(f"⏳ Waiting for batch job to complete: {job_name}")
        while True:
            job = self.client.batches.get(name=job_name)
            state = job.state.name
            print(f"  Current state: {state}")
            if state in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}:
                break
            time.sleep(30)

        print(f"🏁 Job completed with state: {state}")
        if state == "JOB_STATE_FAILED":
            print(f"Error: {job.error}")
            raise RuntimeError(f"Batch job failed: {job.error}")

        return job

    # ---------------------------
    # 3. Fetch inline responses
    # ---------------------------
    def fetch_responses(self, job) -> List[dict]:
        """
        Fetch inline responses and return them as a list of dicts:
        [{index: i, response: "..."}] or [{index: i, error: "..."}]
        """
        if job.state.name != 'JOB_STATE_SUCCEEDED':
            raise RuntimeError(f"Batch job incomplete. State: {job.state.name}, Error: {job.error}")

        inline_responses = getattr(job.dest, "inlined_responses", None)
        if not inline_responses:
            raise ValueError("No inline responses found in batch job.")

        print(f"inline_responses : {inline_responses}")

        responses = []
        for i, inline_response in enumerate(inline_responses):
            if inline_response.response:
                text_output = getattr(inline_response.response, "text", str(inline_response.response))
                responses.append({"index": i, "response": text_output})
            elif inline_response.error:
                responses.append({"index": i, "error": str(inline_response.error)})

        print(f"Fetched {len(responses)} inline responses.")
        return responses

    # ---------------------------
    # 4. Parse Gemini tool output
    # ---------------------------
    def sanitize_gemini_output(self, text: str) -> str:
        """Clean Gemini output to be safely parsed as JSON-like."""
        # Strip Markdown code fences
        cleaned = re.sub(r"^```[a-z_]*|```$", "", text.strip(),
                         flags=re.IGNORECASE | re.MULTILINE).strip()

        # Normalize smart quotes
        cleaned = cleaned.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')

        # Escape apostrophes between letters (e.g., Suzi Quatro's -> Suzi Quatro\'s)
        cleaned = re.sub(r"(?<=\w)'(?=\w)", "\\'", cleaned)

        # Collapse multiline strings to a single line to avoid ast.literal_eval errors
        cleaned = re.sub(r"\s*\n\s*", " ", cleaned)

        # Remove literal ellipsis or truncated tail
        cleaned = cleaned.replace("...", "")

        # Drop any trailing incomplete entries like 'Sir ...
        cleaned = re.sub(r",\s*'[^']*$", "", cleaned)
        cleaned = re.sub(r',\s*"[^"]*$', "", cleaned)

        # Remove trailing commas before } or ]
        cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

        # Drop stray unmatched closing parenthesis
        if cleaned.endswith(")") and cleaned.count("(") == 0:
            cleaned = cleaned[:-1]

        # Balance brackets
        def balance(s, open_char, close_char):
            diff = s.count(open_char) - s.count(close_char)
            return s + (close_char * diff if diff > 0 else "")
        cleaned = balance(cleaned, "{", "}")
        cleaned = balance(cleaned, "[", "]")

        return cleaned

    def parse_tool_code_response(self, text: str) -> list[AnnotationItem]:
        """
        Parse Gemini's tool_code output with multiple record_item_data() calls,
        safely handling the record_items_data(...) wrapper and Python-style dicts.
        """
        raw = self.sanitize_gemini_output(text)
        print(f"Raw sanitized:\n{raw[:300]}...\n")

        # --- Extract all record_item_data(...) blocks, ignoring outer wrapper ---
        pattern = r"record_item_data\s*\((.*?)\)"
        matches = re.findall(pattern, raw, flags=re.DOTALL)

        if not matches:
            print(f"[WARN] No record_item_data blocks found in response:\n{raw[:200]}...")
            return []

        items = []
        for idx, block in enumerate(matches, 1):
            try:
                block = block.strip()

                # Drop stray closing parenthesis if present
                if block.endswith(")"):
                    block = block[:-1].strip()

                # Normalize quotes and whitespace
                block = block.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
                block = re.sub(r"(?<=\w)'(?=\w)", "\\'", block)
                block = re.sub(r"\s*\n\s*", " ", block)

                # Convert Python-style kwargs to dict literal
                block_dict_literal = "{" + block + "}"
                block_dict_literal = re.sub(r"(\w+)\s*=", r'"\1":', block_dict_literal)

                # Parse safely
                parsed = ast.literal_eval(block_dict_literal)

                # Normalize cardinality
                if isinstance(parsed.get("cardinality"), (str, float)):
                    try:
                        parsed["cardinality"] = int(parsed["cardinality"])
                    except ValueError:
                        parsed["cardinality"] = 1

                items.append(AnnotationItem(**parsed))

            except Exception as e:
                print(f"[WARN] Failed to parse record_item_data #{idx}: {e}\nBlock:\n{block}\n")

        return items

    # ---------------------------
    # 5. Run full batch pipeline
    # ---------------------------
    def run_batch(self, prompts: List[str]) -> List[List[AnnotationItem]]:
        """
        Submit a batch job, wait for completion, fetch responses,
        and parse them into structured AnnotationItem lists.
        Always returns a list of lists.
        """
        print(f"[BatchGeminiHandler] Running batch with {len(prompts)} prompts...")
        job_name = self.submit_batch_job(prompts)
        job = self.wait_for_completion(job_name)
        raw_responses = self.fetch_responses(job)

        print(f"[BatchGeminiHandler] Raw responses fetched: {len(raw_responses)}")

        all_items = []
        for resp in raw_responses:
            # Case: Success with text response
            if "response" in resp and resp["response"]:
                try:
                    items = self.parse_tool_code_response(resp["response"]) or []
                    all_items.append(items)
                except Exception as e:
                    print(f"[ERROR] Failed to parse response {resp['index']}: {e}")
                    all_items.append([])
            else:
                # Case: Error or missing response
                print(f"[WARN] Response {resp['index']} returned error: {resp.get('error')}")
                all_items.append([])

        print(f"[BatchGeminiHandler] Parsed {sum(len(lst) for lst in all_items)} total items.")
        return all_items
