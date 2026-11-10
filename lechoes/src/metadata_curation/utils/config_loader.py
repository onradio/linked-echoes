import yaml

from deepmerge import always_merger  # Assuming you use deepmerge

def load_config(default_path="configs/default_config.yaml", override_path=None, overrides_dict=None):
    # Load default config
    with open(default_path, "r") as f:
        base_config = yaml.safe_load(f)

    # If override_path is a dict, treat it as override config directly
    if isinstance(override_path, dict):
        return always_merger.merge(base_config, override_path)

    # If override_path is a path or string, treat it as a YAML file
    elif override_path:
        try:
            with open(override_path, "r") as f:
                override_config = yaml.safe_load(f)
            return always_merger.merge(base_config, override_config)
        except FileNotFoundError:
            print(f"Override file '{override_path}' not found. Using default config.")

        # Merge CLI key=value overrides
    if overrides_dict:
        base_config = always_merger.merge(base_config, overrides_dict)

    return base_config
