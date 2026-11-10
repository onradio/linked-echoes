from gemini_handler import GeminiHandler
from evaluation import AnnotationItem


from typing import get_origin, get_args, List, Union
from pydantic import BaseModel

def run_prompt(prompt_path: str, overrides: dict = None):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    overrides = overrides or {}
    gemini_cfg = overrides.get("gemini", {})
    mode = gemini_cfg.pop("mode", "bulk")

    handler = GeminiHandler(mode=mode, **gemini_cfg)

    if "Item 2" in prompt:
        print("Detected bulk prompt. Extracting multiple records...")
        results = handler.extract_bulk(prompt)
        output_data = [item.model_dump() for item in results]
    else:
        print("Detected single prompt. Extracting one record...")
        item = handler.extract(prompt)
        output_data = item.model_dump()

    return output_data



def get_clean_type(type_):
    origin = get_origin(type_)
    args = get_args(type_)

    if origin == list or origin == List:
        inner = get_clean_type(args[0]) if args else "Any"
        return f"List[{inner}]"
    elif type_ == str:
        return "str"
    elif type_ == int:
        return "int"
    elif type_ == float:
        return "float"
    elif origin == Union:
        # Strip NoneType from Optional[...]
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return get_clean_type(non_none_args[0])
        else:
            return "Union[" + ", ".join(get_clean_type(arg) for arg in non_none_args) + "]"
    else:
        return str(type_.__name__)

def generate_python_style_schema(model: BaseModel, function_name: str = "record_item_data"):
    lines = [f"[START_SCHEMA]", f"{function_name}("]
    for field_name, field in model.model_fields.items():
        field_type = get_clean_type(field.annotation)
        lines.append(f"    {field_name}: {field_type},")
    lines[-1] = lines[-1].rstrip(",")  # Remove trailing comma
    lines.append(")")
    lines.append("[END_SCHEMA]")
    return "\n".join(lines)



if __name__ == "__main__":
    print("PYTHON-STYLE:")
    print(generate_python_style_schema(AnnotationItem))
    run_prompt("prompts/prompt_bulk_20250715_225846.txt", {})
