"""
URI Factory Module

This module provides a factory for generating URIs for items based on
different strategies. It supports creating URIs using short hashes,
title slugs with short hashes, UUIDs, and sequential IDs.

The module is designed to be used with the BME (British Music Experience)
API and integrates with the `Item` data model.

Key Functions:

- `generate_short_hash(item_data, hash_length=8)`: Generates a short
  hash from item data using SHA-256.
- `create_uri(base_uri, item, strategy)`: Creates a URI for an item
  based on the specified strategy.
- `_get_next_sequential_id(item_type)`: Simulates fetching and
  incrementing the next sequential ID for a given item type (for
  demonstration purposes).

Strategies:

- `short_hash`: Creates a URI with the item type and a short hash
  generated from the item's data.
- `title_slug_hash`: Creates a URI with the item type, a URL-safe
  slug of the item's title, and a short hash.
- `uuid`: Creates a URI with a universally unique identifier (UUID).
- `sequential_id`: Creates a URI with the item type and a sequential
  ID (requires a mechanism for tracking and incrementing IDs in a
  real application).

Example Usage:

```python
from lechoes.src.schema import Item
from uri_factory import create_uri

base_uri = "[http://radio.liv.ac.uk/bme/](http://radio.liv.ac.uk/bme/)"
item = Item(**item_data)  # Assuming you have item_data defined

# Create a URI using the short hash strategy
uri = create_uri(base_uri, item, "short_hash")

# Create a URI using the title slug + hash strategy
uri = create_uri(base_uri, item, "title_slug_hash")

# Create a URI using the UUID strategy
uri = create_uri(base_uri, item, "uuid")

# Create a URI using the sequential ID strategy
uri = create_uri(base_uri, item, "sequential_id")
"""

import hashlib
import logging
from urllib.parse import quote

from rdflib import URIRef

from lechoes.src.metamodel import Item
from lechoes.src.utils import to_camel_case

logger = logging.getLogger(__name__)


def generate_short_hash(item_data, hash_length=8):
    """
    Generates a short hash from item data.

    Parameters
    ----------
    item_data : dict
        A dictionary containing the item's data.
    hash_length : int, optional
        The desired length of the short hash, by default 8.

    Returns
    -------
    str
        The short hash as a string.
    """
    field_order = [
        "type",
        "name",
        "description",
        "caption",
        "case_number",
        "image",
        "audio",
        "video",
        "genre",
        "music_artist",
        "colour",
        "material",
        "condition",
        "year",
        "lender",
        "lend_start",
        "lend_end",
    ]

    concatenated_string = ""
    for field in field_order:
        value = item_data.get(field)
        if value:
            concatenated_string += str(value) + "|"

    concatenated_string = concatenated_string.rstrip("|")
    sha256_hash = hashlib.sha256(concatenated_string.encode()).hexdigest()
    return sha256_hash[:hash_length]


def create_uri(base_uri: str, item: Item, strategy: str) -> URIRef:
    """
    URI factory that creates a URI for an item based on a given strategy.

    Parameters
    ----------
    base_uri : str
        The base URI to use for minting the new URI.
    item : Item
        The item for which to create a URI.
    strategy : str
        The URI creation strategy to use. Must be one of:
        'short_hash', 'title_slug_hash', 'uuid', 'sequential_id'.

    Returns
    -------
    URIRef
        The generated URI.

    Raises
    ------
    ValueError
        If an unsupported strategy is specified.
    """

    item_data = item.model_dump()

    if strategy == "short_hash":
        short_hash = generate_short_hash(item_data)
        uri_string = f"{base_uri}{to_camel_case(item.type)}/{short_hash}"

    elif strategy == "title_slug_hash":
        short_hash = generate_short_hash(item_data, hash_length=4)
        title_slug = quote(item.item_title.lower().replace(" ", "-"))
        uri_string = f"{base_uri}{to_camel_case(item.type)}/{title_slug}-{short_hash}"

    elif strategy == "uuid":
        import uuid

        uri_string = f"{base_uri}{uuid.uuid4()}"

    elif strategy == "sequential_id":
        # TODO we still need a mechanism to track and increment the IDs
        logger.warning("Sequential ID strategy is not implemented, but simulated.")

        sequential_id = _get_next_sequential_id(item.item_type)
        uri_string = f"{base_uri}{to_camel_case(item.item_type)}/{sequential_id:04}"  # Pad with zeros to maintain a consistent format
    else:
        raise ValueError(f"Unsupported URI creation strategy: {strategy}")

    return URIRef(uri_string)


# In-memory storage for simulating sequential IDs
_sequential_id_store = {}


def _get_next_sequential_id(item_type: str) -> int:
    """
    Simulates fetching and incrementing the next sequential ID for a given item type.

    Notes:
    (*) XXX Currently uses the in-memory `_sequential_id_store`
    (*) The sequential id strategy should increment the last used ID for each
        item type. The idea is to fire a SPAQRL query to get the last ID from
        the last URI and increment it by 1. For simplicity, this can also be
        done as a queue with some semantic sugar.

    Parameters
    ----------
    item_type : str
        The type of the item for which to get the next sequential ID.

    Returns
    -------
    current_id : int
        The next sequential ID.
    """
    current_id = _sequential_id_store.get(item_type, 0) + 1
    _sequential_id_store[item_type] = current_id
    return current_id
