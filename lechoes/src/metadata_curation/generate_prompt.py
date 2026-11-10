from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

from jinja2 import Environment, FileSystemLoader
from utils.config_loader import load_config


def generate_prompt(
    config_path: Path = Path("config/default_config.yaml"),
    overrides: dict = None,
    template_dir: Path = Path("templates"),
    prompt_output_dir: Path = Path("prompts"),
    input_items: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]] = None,
) -> List[Tuple[str, str]]:
    """
    Generate prompts for either sync (per row) or batch (case-by-case or full).

    Args:
        config_path: Path to YAML configuration file.
        overrides: Optional dictionary to override config values.
        template_dir: Directory containing Jinja2 prompt templates.
        prompt_output_dir: Directory to save generated prompts.
        input_items: Flat list (sync) or list of lists (batch/cases).

    Returns:
        A list of (prompt_text, filename) tuples.
    """
    config = load_config(config_path, overrides_dict=overrides)

    env = Environment(loader=FileSystemLoader(template_dir))
    template_name = config.get("prompt_variant", "base_prompt") + ".jinja2"
    template = env.get_template(template_name)

    prompt_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prompt_outputs = []

    # New naming convention
    prompt_mode = config.get("input_block", {}).get("mode", "sync").lower()

    evaluation_type = config.get("evaluation", {}).get("type", "row_by_row")

    print(f"[Prompt Generation] Mode: {prompt_mode}")

    if prompt_mode == "sync":

        if evaluation_type == "row_by_row":
            for idx, item in enumerate(input_items):
                config["input_data"] = item
                prompt = template.render(**config)
                prompt = prompt.replace("[START_SCHEMA]", "```python").replace("[END_SCHEMA]", "```")

                filename = prompt_output_dir / f"prompt_single_{timestamp}_{idx:03}.txt"

                prompt_outputs.append((prompt, str(filename)))

        elif evaluation_type == "case_by_case":
            for case_idx, case_items in enumerate(input_items):
                case_config = config.copy()
                case_config["input_data"] = case_items
                case_config["input_block"]["case_info"] = case_config["input_block"].get("case_info",
                                                                                         f"case_{case_idx + 1}")

                prompt = template.render(**case_config)
                prompt = prompt.replace("[START_SCHEMA]", "```python").replace("[END_SCHEMA]", "```")

                filename = prompt_output_dir / f"prompt_case_{case_idx + 1}_{timestamp}.txt"

                prompt_outputs.append((prompt, str(filename)))


    elif prompt_mode == "batch":
        # Case-by-case or full-bulk
        if isinstance(input_items[0], list):  # list of cases
            for case_idx, case_items in enumerate(input_items):
                config["input_data"] = case_items
                rendered = template.render(**config)
                prompt = rendered.replace("[START_SCHEMA]", "```python").replace("[END_SCHEMA]", "```")

                filename = prompt_output_dir / f"prompt_case_{timestamp}_{case_idx:03}.txt"

                prompt_outputs.append((prompt, str(filename)))
        else:
            # Flat full-bulk
            config["input_data"] = input_items
            rendered = template.render(**config)
            prompt = rendered.replace("[START_SCHEMA]", "```python").replace("[END_SCHEMA]", "```")

            filename = prompt_output_dir / f"prompt_batch_{timestamp}.txt"

            prompt_outputs.append((prompt, str(filename)))

        # with open(filename, "w", encoding="utf-8") as f:
        #     f.write(prompt)

    else:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}. Use 'sync' or 'batch'.")

    return prompt_outputs
