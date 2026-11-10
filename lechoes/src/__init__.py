import dataclasses
import inspect

from lechoes.src import descriptors


def init_descriptors(config_file):
    """
    Initializes all the type classes with mappings from the config file.
    """

    # List of type class names
    type_classes = [
        cls[0]
        for cls in inspect.getmembers(
            descriptors,
            lambda cls: inspect.isclass(cls) and dataclasses.is_dataclass(cls),
        )
    ]

    for cls_name in type_classes:
        try:
            cls = getattr(descriptors, cls_name)
            cls.load_mappings(config_file)
        except KeyError:
            print(f"Warning: Class '{cls_name}' not found.")
