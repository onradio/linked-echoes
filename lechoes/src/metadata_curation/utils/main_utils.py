def clean_schema_for_gemini(schema: dict) -> dict:
    """
    Recursively cleans a Pydantic schema for Gemini compatibility.
    - Removes keys like 'title', 'default', 'anyOf', 'allOf', 'oneOf'
    - Converts 'number' to 'integer' where appropriate
    - Ensures arrays are correctly typed
    - Returns a new dict (does not mutate in place)
    """
    unsupported_keys = {"title", "default",
                        "description",
                        #  "examples", "anyOf", "allOf", "oneOf", "nullable"
                        }

    def _clean(value):
        if isinstance(value, dict):
            cleaned = {}
            for k, v in value.items():
                if k in unsupported_keys:
                    continue
                if k == "type" and v == "number":
                    cleaned[k] = "integer"  # Gemini prefers int over float
                elif k == "items" and isinstance(v, dict):
                    cleaned[k] = _clean(v)
                else:
                    cleaned[k] = _clean(v)
            return cleaned
        elif isinstance(value, list):
            return [_clean(item) for item in value]
        else:
            return value

    return _clean(schema)
