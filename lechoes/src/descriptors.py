"""
Types are used as a bridge between the API and the KG schema, to encapsulate
the key concepts and their mappings. Each type class is a dataclass that
contains a mapping of keys to URIs. The mappings are loaded from a JSON config
file, which is passed to the `init_types` function to initialize all the types.
This is done to keep the types and their mappings separate and to allow for easy
extension and modification of the mappings while keeping the logic sound.

Notes
-----
(*) The load_mappings function could already URIRef the URIs rather than doing
    this every time in the uri method.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

from rdflib import Literal, URIRef

from lechoes.src.utils import to_snake_case

logger = logging.getLogger(__name__)


class BaseType(ABC):
    """
    Base class for type handling with basic functionality.
    """

    _mappings = {}  # the allowed base keys and their corresponding URI(s)
    _type_uri = None  # the basic type URI of this concept/descriptor
    _relation_uri = None  # the relation URI between the Item and this decriptor

    @classmethod
    def _handle_n_relations(cls, key: list):
        nodes = []
        for k in key:
            if k in cls._mappings:
                current_node = cls._mappings[k]
                if isinstance(current_node, dict):
                    current_node = current_node.get("_type_uri")

                if isinstance(current_node, str):
                    nodes.append(URIRef(current_node))

                else:
                    for node in current_node:
                        nodes.append(URIRef(node))
        return nodes

    @classmethod
    def load_mappings(cls, config_file):
        with open(config_file, "r") as f:
            data = json.load(f)
            mappings = data.get(to_snake_case(cls.__name__), {})
            cls._type_uri = mappings.pop("_type_uri", None)
            cls._relation_uri = mappings.pop("_relation_uri", None)
            cls._mappings.update(mappings)

    @classmethod
    def all(cls) -> list[str]:
        """List all defined keys."""
        return list(cls._mappings.keys())

    @classmethod
    def uri(cls, key: str | int | list[str] = None) -> Union[str, List[str], None]:
        if not key:  # concept URI requested
            if isinstance(cls._type_uri, list) and len(cls._type_uri) == 1:
                # Single type uri but in list
                return URIRef(cls._type_uri[0])
            elif isinstance(cls._type_uri, list):
                # multiple type uris'
                return [URIRef(uri) for uri in cls._type_uri]
            else:
                # string type uri.
                return URIRef(cls._type_uri)

        if isinstance(key, list):
            return cls._handle_n_relations(key)

        key = str(key)

        if key not in cls._mappings:
            raise ValueError(f"Invalid or unsupported {cls.__name__}: {key}")

        node = cls._mappings[key]
        if isinstance(node, dict):
            node = node.get("_type_uri")

        return URIRef(node) if isinstance(node, str) else [URIRef(n) for n in node]

    @classmethod
    def relation_uri(cls) -> str:
        """Returns the relation URI for this type."""
        if cls._relation_uri is None:
            raise ValueError(f"No relation URI defined for {cls.__name__}")
        return URIRef(cls._relation_uri[0])  # XXX assume one relation for now!


class LiteralType(BaseType):
    """
    Redefines BaseType for types that are represented as literals in the KG.
    """

    @classmethod
    def uri(cls, key: str | int = None) -> Union[str, List[str], None]:
        # logger.warning(f"{cls.__name__} is a literal type with no URIs.")
        # In theory, literals don't have URIs, but we can still use them for
        # mapping purposes. This method is overridden to provide a warning.
        # return key as a string for now with @en language tag
        return Literal(key) if key else None

    # @classmethod
    # def relation_uri(cls) -> str:
    #     return cls._relation_uri[0]  # XXX assume one relation for now!


class LookupableType(BaseType):
    """
    Extends BaseType for types that require entity lookup functionality. This
    allows to extend the available mappings that are registered in the type
    with additional entities that are looked up from a SPARQL endpoint (e.g.
    DBpedia, Wikidata, etc.).
    """

    @classmethod
    @abstractmethod
    def lookup_entity(cls, key: str) -> str:
        """Looks up the entity URI. Must be implemented by subclasses."""
        pass


@dataclass
class Type(BaseType):
    _mappings = {
        "instrument": {},
        "costume": {},
        "record": {},
        "notation": {},
        "graphic_art": {},
    }


@dataclass
class Name(LiteralType):
    _mappings = {}


@dataclass
class Description(LiteralType):
    _mappings = {}


@dataclass
class Caption(LiteralType):
    _mappings = {}


@dataclass
class Image(LiteralType):
    _mappings = {}


@dataclass
class Audio(LiteralType):
    _mappings = {}


@dataclass
class Video(LiteralType):
    _mappings = {}


@dataclass
class Genre(LookupableType):
    _mappings = {}


@dataclass
class MusicWork(LookupableType):
    _mappings = {}


@dataclass
class MusicArtist(LookupableType):
    _mappings = {}


@dataclass
class Publisher(LookupableType):
    _mappings = {}


@dataclass
class Event(LookupableType):
    _mappings = {}


@dataclass
class Country(LookupableType):
    _mappings = {}


@dataclass
class City(LookupableType):
    _mappings = {}


@dataclass
class Venue(LookupableType):
    _mappings = {}


@dataclass
class Year(BaseType):
    _mappings = {}


@dataclass
class Colour(LookupableType):
    _mappings = {}


@dataclass
class Material(LookupableType):
    _mappings = {}


@dataclass
class Condition(BaseType):
    _mappings = {}


@dataclass
class Lender(LiteralType):
    _mappings = {}


@dataclass
class LendStart(LiteralType):
    _mappings = {}


@dataclass
class LendEnd(LiteralType):
    _mappings = {}
