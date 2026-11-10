import json
import re
from datetime import datetime
from typing import List, Literal, Optional

from fastapi import HTTPException
from pydantic import BaseModel, field_validator

from lechoes.config import settings
from lechoes.src import descriptors
from lechoes.src.utils import to_camel_case, to_snake_case


mappings_file = "./mappings.json"
descriptors_file = "./src/descriptors.py"


with open(mappings_file) as file:
    mappings = json.loads(file.read())


def name_not_in_mappings(cls, name: str) -> str:
    if name not in mappings:
        raise HTTPException(
            status_code=400,
            detail=f"Field <{name}> does not exist in the mappings!",
        )
    return name


def name_in_mappings(cls, name: str) -> str:
    if name in mappings:
        raise HTTPException(
            status_code=400,
            detail=f"Field <{name}> already exists in the mappings!",
        )
    return name

class NewItem(BaseModel):
    name: str
    type_uris: List[str]
    relation_uris: Optional[List[str]]
    lookups: Optional[List[dict[str, List[str]]]]
    base_class: Literal["BaseType", "LiteralType", "LookupableType"]

    @field_validator("name")
    def check_name(cls, name: str) -> str:
        return name_in_mappings(cls, to_snake_case(name))

    def write_descriptors(self) -> None:
        template = f"""

@dataclass
class {to_camel_case(self.name)}({self.base_class}):
    _mappings = {{}}
"""

        with open(descriptors_file, "a") as file:
            file.write(template)

    def write_mappings(self, fields: dict[str, str | List[str]]) -> None:
        mappings[self.name] = fields

        with open(mappings_file, "w") as file:
            file.write(json.dumps(mappings))

    def insert(self):
        # NOTE: This needs to be strict as it writes to file.
        fields = {"_type_uri": self.type_uris}

        for lookup in self.lookups:
            fields.update(lookup)

        if self.relation_uris:
            fields.update({"_relation_uri": self.relation_uris})

        self.write_mappings(fields)
        self.write_descriptors()


class UpdateItem(BaseModel):
    name: str
    type_uris: List[str]
    relation_uris: Optional[List[str]]
    lookups: Optional[List[dict[str, List[str]]]]
    base_class: Literal["BaseType", "LiteralType", "LookupableType"]

    @field_validator("name")
    def check_name(cls, name: str):
        return name_not_in_mappings(cls, name)

    def update(self) -> None:
        pass


class DeleteItem(BaseModel):
    name: str

    @field_validator("name")
    def check_name(cls, name: str) -> str:
        return name_not_in_mappings(cls, name)

    def delete(self) -> None:
        mappings.pop(self.name)

        with open(mappings_file, "+w") as file:
            file.write(json.dumps(mappings, indent=2))

        name = to_camel_case(self.name)
        desc_class = getattr(descriptors, name)
        base_class = desc_class.__bases__[0].__name__

        with open(descriptors_file, "r") as file:
            desc = file.read()

        pattern = rf"@dataclass\s+class\s+{name}\s*\(\s*{base_class}\s*\):\s*_mappings\s*=\s*\{{\s*\}}"

        updated_desc = re.sub(pattern, "", desc, flags=re.MULTILINE)
        with open(descriptors_file, "w") as file:
            file.write(updated_desc)


class Response(BaseModel):
    message: str | None = None