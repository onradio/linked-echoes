"""
Wraps the type and endpoint specific SPAQRL queries for the entity lookup.
"""

import re

from fastapi import HTTPException
from sqlmodel import delete as sql_delete
from pathlib import Path
from collections import defaultdict
from typing import Optional, Any, Dict, List, Tuple

from lechoes import SessionDep, sparql_query, sparql_update, logger
from lechoes.src.models import ItemIdentifiers
from lechoes.src.metamodel import Item, LinkedEntity
from lechoes.src.scho_triplifier import Triplifier
from lechoes.src.uri_factory import create_uri
from lechoes.src.utils import hash_exists
from rdflib import Literal, URIRef
from lechoes.src.namespaces import BME
from lechoes.src.queries.sparql_queries import (
    ENTITY_ITEM_COUNTS_QUERY,
    RESOLVE_CATEGORY_URI_QUERY,
    RETRIEVE_ITEMS_BY_CATEGORY_QUERY,
    RETRIEVE_OBJECT_QUERY,
    SEARCH_OBJECTS_QUERY,
    ZONE_ITEMS_QUERY,
)


# Path to the mappings configuration file
MAPPINGS_PATH = str(Path(__file__).parent.parent.parent / "mappings_min.json")

# --- Field maps and location helpers ---

# SPARQL variable pairs backing each LinkedEntity field of `metamodel.Item`.
# `location` is the zone the item's display case sits in; `exhibition` is the real
# ExhibitionEvent (core.ttl) hosted in that zone, reached by reversing its
# scho:location link (see sparql_queries.py's module docstring) — no longer a stand-in
# for the zone itself.
# `colour` reads `?colourValue`, which is a plain literal in the current KG.
LINKED_ENTITY_FIELDS = {
    'category': ('categoryUri', 'categoryLabel'),
    'type': ('typeUri', 'typeLabel'),
    'location': ('zoneUri', 'zoneLabel'),
    'exhibition': ('exhibitionUri', 'exhibitionLabel'),
    'genre': ('genreUri', 'genreLabel'),
    'music_work': ('musicWorkUri', 'musicWorkLabel'),
    'music_album': ('musicAlbumUri', 'musicAlbumLabel'),
    'music_artist': ('musicArtistUri', 'musicArtistLabel'),
    'event': ('eventUri', 'eventLabel'),
    'country': ('countryUri', 'countryLabel'),
    'place': ('placeUri', 'placeLabel'),
    'venue': ('venueUri', 'venueLabel'),
    'colour': ('colourValue', 'colourLabel'),
    'material': ('materialUri', 'materialLabel'),
}

# Fields the API contract types as a single LinkedEntity rather than a list.
SINGLE_VALUE_LINKED_FIELDS = {'category', 'type'}

SIMPLE_FIELDS = ["identifier", "name", "description", "caption", "image", "audio", "video", "url"]

BME_BASE = "http://radio.liv.ac.uk/bme"
ZONE_URI_TEMPLATE = BME_BASE + "/Zone/%s"


def zone_uri_for(zone: str) -> str:
    """
    Normalise whatever a caller passes as a zone into the KG's zone URI.

    The dashboard and CMS have historically used several spellings of the same zone —
    ``5``, ``Zone5``, ``Zone 5`` — and the superseded KG's own node was
    ``bme/ExhibitionEvent/Zone5`` while the code that queried it built
    ``bme/Exhibition/{zone}``, so nothing matched. All of those spellings are accepted
    here and resolve to ``bme/Zone/5``. A full URI is passed through untouched.
    """
    token = str(zone).strip()
    if token.startswith("http://") or token.startswith("https://"):
        return token
    # Strip a leading "zone" in any casing, plus any separator that follows it.
    token = re.sub(r"^zone[\s_-]*", "", token, flags=re.IGNORECASE).strip()
    return ZONE_URI_TEMPLATE % token


BME_NAMESPACE = BME_BASE + "/"


# Populated on first use by `entity_item_counts`, cleared by `invalidate_entity_item_counts`.
_ENTITY_ITEM_COUNTS: Optional[Dict[str, int]] = None


def entity_item_counts() -> Dict[str, int]:
    """
    How many items link to each entity, keyed by the entity's raw URI.

    Cached because it is read once per linked entity on every payload; CMS writes call
    `invalidate_entity_item_counts`. A failure degrades to no counts rather than failing
    the lookup.
    """
    global _ENTITY_ITEM_COUNTS
    if _ENTITY_ITEM_COUNTS is not None:
        return _ENTITY_ITEM_COUNTS

    counts: Dict[str, int] = {}
    try:
        sparql_query.setQuery(ENTITY_ITEM_COUNTS_QUERY)
        results = sparql_query.queryAndConvert()
        for row in results.get("results", {}).get("bindings", []):
            uri = row.get("entity", {}).get("value")
            raw_count = row.get("count", {}).get("value")
            if uri and raw_count is not None:
                counts[uri] = int(raw_count)
    except Exception as e:  # noqa: BLE001 - see docstring: counts must never break a lookup
        logger.error(f"Failed to compute entity item counts: {e}")
        return {}

    _ENTITY_ITEM_COUNTS = counts
    return counts


def invalidate_entity_item_counts() -> None:
    """Drop the cached counts so the next read recomputes them after a CMS write."""
    global _ENTITY_ITEM_COUNTS
    _ENTITY_ITEM_COUNTS = None


def case_number_from_uri(case_uri: Optional[str]) -> Optional[str]:
    """
    Recover the display-case number from a ``scho:location`` URI.

    The case number used to be a literal (``scho:itemLocation "15"``) and is now the last
    segment of ``bme/Case/15``. `Item.case_number` is still a string, so the API contract
    is unchanged for consumers.
    """
    if not case_uri:
        return None
    return case_uri.rstrip("/").rsplit("/", 1)[-1] or None


def _binding_pair(result: Dict[str, Dict[str, str]], uri_key: str, label_key: str) -> Optional[Tuple[Optional[str], str]]:
    """
    Turn one row's (uri, label) binding pair into a (uri, label) tuple, or None.

    A binding that is a *literal* rather than a URI is its own label — that is how
    `scho:color "blue"` arrives now that colours are no longer typed wd: entities. Reading
    the binding's ``type`` rather than assuming a URI is what keeps this working for both
    the current KG and the superseded one, where colour was `wd:Q1088` with a label.
    """
    uri_binding = result.get(uri_key)
    label_binding = result.get(label_key)

    if uri_binding is not None and uri_binding.get("type") == "uri":
        # A URI with no label carries nothing displayable, so it is dropped rather than
        # surfaced as a bare IRI in a field the front-ends render as text.
        if label_binding and label_binding.get("value"):
            return (uri_binding["value"], label_binding["value"])
        return None

    # Literal-valued binding (colour), or a label with no URI at all.
    for binding in (label_binding, uri_binding):
        if binding and binding.get("value"):
            return (None, binding["value"])
    return None


# --- Helper Function for Item Processing ---

def _process_sparql_bindings_to_item(bindings: List[Dict[str, Dict[str, str]]], item_uri: Optional[str] = None) -> Item:
    """
    Processes a list of SPARQL result bindings (rows) for a single item
    into a structured Item object.

    Args:
        bindings: The list of SPARQL result bindings for one item.
        item_uri: The URI of the item if known (used for grouping in list queries).
                  If None, it assumes the bindings are for a single item only.

    Returns:
        The fully populated Item object.

    Raises:
        HTTPException: If no bindings are provided.
    """
    if not bindings:
        raise HTTPException(status_code=404, detail=f"No data found for item{' with URI: ' + item_uri if item_uri else ''}.")

    # Use item_uri as a key if provided, otherwise use a placeholder
    key = item_uri if item_uri else 'single_item'

    # Entities are collected per (field) as an ordered set of (uri, label) pairs. Pairing
    # happens per row: the previous implementation appended URIs and labels to two
    # parallel lists and zipped them by index afterwards, which silently mismatched a
    # label to the wrong URI as soon as one row bound a URI without a label.
    items_data = defaultdict(lambda: defaultdict(dict))
    single_value_data = defaultdict(dict)
    years_data = defaultdict(dict)
    lending_data = defaultdict(dict)

    # --- Grouping/Extraction Logic ---
    for result in bindings:
        current_key = result.get("itemUri", {}).get("value") if "itemUri" in result else key

        # Store single-value fields
        for field in SIMPLE_FIELDS:
            if field in result:
                single_value_data[current_key][field] = result[field]["value"]

        # Case number: the URI suffix of scho:location. `caseNumber` is still honoured so
        # the API keeps working against a store that has not been reloaded yet.
        case_uri = result.get("caseUri", {}).get("value")
        if case_uri:
            single_value_data[current_key]['case_number'] = case_number_from_uri(case_uri)
        elif "caseNumber" in result:
            single_value_data[current_key]['case_number'] = result["caseNumber"]["value"]

        # Case name: the case's own rdfs:label (e.g. "New Romantics"), for grouping items
        # by their physical display case rather than just its number.
        if result.get("caseLabel", {}).get("value"):
            single_value_data[current_key]['case_name'] = result["caseLabel"]["value"]

        # Year: plain literals off scho:temporalCoverage, de-duplicated but order-preserved
        if "year" in result and result["year"].get("value"):
            years_data[current_key][result["year"]["value"]] = None

        # Store linked entity fields
        for field, (uri_key, label_key) in LINKED_ENTITY_FIELDS.items():
            pair = _binding_pair(result, uri_key, label_key)
            if pair is not None:
                items_data[current_key][field][pair] = None

        # Store lending information
        if "lender" in result:
            lending_data[current_key]['lender'] = result["lender"]["value"]
        if "lendStart" in result:
            lending_data[current_key]['lend_start'] = result["lendStart"]["value"]
        if "lendEnd" in result:
            lending_data[current_key]['lend_end'] = result["lendEnd"]["value"]

    # --- Transformation Logic ---

    # We expect only one item if item_uri was not provided
    if not item_uri:
        if len(single_value_data) != 1:
             raise HTTPException(status_code=500, detail="Internal error: Multiple items returned for single identifier query.")
        key = next(iter(single_value_data.keys()))

    pydantic_fields = {**single_value_data[key], **lending_data[key]}

    if years_data[key]:
        pydantic_fields['year'] = list(years_data[key])

    counts = entity_item_counts()
    for field, entities in items_data[key].items():
        linked_entities_list = [
            LinkedEntity(uri=uri, label=label, count=counts.get(uri))
            for uri, label in entities
        ]

        if field in SINGLE_VALUE_LINKED_FIELDS:
            pydantic_fields[field] = linked_entities_list[0] if linked_entities_list else None
        else:
            pydantic_fields[field] = linked_entities_list if linked_entities_list else None

    # Handle mandatory fields
    if 'category' not in pydantic_fields or pydantic_fields['category'] is None:
        pydantic_fields['category'] = LinkedEntity(label="Unknown")

    if 'name' not in pydantic_fields or pydantic_fields['name'] is None:
        pydantic_fields['name'] = pydantic_fields.get('identifier', 'Unknown')

    return Item(**pydantic_fields)


# --- Core Functions ---

def insert_hash(case_number: int, item_uri: str, session: SessionDep) -> None:
    if hash_exists(item_uri, session):
        raise HTTPException(status_code=400, detail="Item already exists!")

    session.add(
        ItemIdentifiers(case_number=case_number, hash=item_uri),
    )
    session.commit()


CASE_ZONE_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX scho: <http://schema.org/>

SELECT ?caseUri ?caseLabel ?zoneUri WHERE {
    ?caseUri scho:containedInPlace ?zoneUri .
    OPTIONAL { ?caseUri rdfs:label ?caseLabel . }
}
"""


def get_location_lookup() -> Dict[str, Dict[str, Any]]:
    """
    Read the case -> zone hierarchy out of the KG, in the shape `Triplifier` expects.

    `Triplifier._add_location_info` needs this to place a new item in its zone. Without
    it every inserted item is written under `bme/Zone/unknown` and is then invisible to
    `get_items_by_zone`, because the zone is a *property of the case*, not of the item —
    the API cannot infer it from the payload alone.

    Read from the store rather than hard-coded: cases are museum furniture and get
    renamed and re-zoned, and a stale table here would silently misplace new items.
    """
    sparql_query.setQuery(CASE_ZONE_QUERY)
    results = sparql_query.queryAndConvert()

    lookup: Dict[str, Dict[str, Any]] = {}
    for row in results.get("results", {}).get("bindings", []):
        case_uri = row.get("caseUri", {}).get("value")
        zone_uri = row.get("zoneUri", {}).get("value")
        case_number = case_number_from_uri(case_uri)
        if not case_number or not zone_uri:
            continue
        lookup[case_number] = {
            "zone": zone_uri.rstrip("/").rsplit("/", 1)[-1],
            "case_name": row.get("caseLabel", {}).get("value") or f"Case {case_number}",
        }
    return lookup


def insert(item: Item, session: SessionDep):
    triplifier = Triplifier(
        mappings_path=MAPPINGS_PATH,
        base_uri=str(BME),
        uri_strategy='slug',
        location_lookup=get_location_lookup(),
    )
    graph = triplifier.triplify(item)
    # N-Triples, not Turtle: a Turtle serialisation opens with `@prefix` directives, which
    # are a syntax error inside `INSERT DATA { ... }`. N-Triples writes absolute IRIs and
    # needs no prologue, so the payload is valid wherever it is sent.
    generated = graph.serialize(format="nt")

    sparql_update.setQuery(
        f"""
        INSERT DATA {{{generated}}}
        """
    )
    sparql_update.setRequestMethod("POST")
    sparql_update.query().convert()
    # Invalidated here rather than in the route so direct query-layer callers are covered.
    invalidate_entity_item_counts()
    insert_hash(item.case_number, triplifier.uri, session)

    return {"detail": "Added item", "item": item}


def get_item_by_identifier(identifier: str) -> Item:
    """
    Fetch and wrap a single item using the comprehensive RETRIEVE_OBJECT_QUERY.
    Returns the matching Item object.
    """
    # Use RDFLib's Literal to properly escape the identifier for SPARQL
    escaped_identifier = Literal(identifier).n3()
    sparql_query.setQuery(RETRIEVE_OBJECT_QUERY % escaped_identifier)
    results = sparql_query.queryAndConvert()
    
    bindings = results.get("results", {}).get("bindings")
    
    if not bindings:
        raise HTTPException(status_code=404, detail=f"No triples found for item with identifier: {identifier}")
    
    return _process_sparql_bindings_to_item(bindings)


def resolve_category_uri(category_label: str) -> Optional[str]:
    """
    Resolve a category label to the URI the KG classifies items under.

    Matching is case-insensitive and ignores the language tag: labels are stored as
    `"Clothing"@en`, and a caller that sends `clothing` — as the CMS's own filter chips do
    — should not silently get an empty list.

    Returns None when no category carries that label.
    """
    # A plain literal: the query lowercases both sides and compares them as strings, so a
    # language tag here would make the comparison fail against "Clothing"@en.
    escaped_category_label = Literal(category_label).n3()
    sparql_query.setQuery(RESOLVE_CATEGORY_URI_QUERY % escaped_category_label)
    results = sparql_query.queryAndConvert()

    for row in results.get("results", {}).get("bindings", []):
        uri = row.get("categoryUri", {}).get("value")
        if uri:
            return uri
    return None


def get_items_by_category(category_label: str) -> list[Item]:
    """
    Fetch all items belonging to a specific category using the comprehensive 
    RETRIEVE_ITEMS_BY_CATEGORY_QUERY. Returns a list of Item objects.
    
    Args:
        category_label: The human-readable label of the category (e.g., "Musical Instrument")
    
    Returns:
        List of Item objects belonging to the specified category.
    """
    category_uri = resolve_category_uri(category_label)
    if category_uri is None:
        # An unknown category is an empty collection, not an error: the CMS asks for
        # categories that may legitimately hold nothing.
        return []

    escaped_category_uri = URIRef(category_uri).n3()
    sparql_query.setQuery(RETRIEVE_ITEMS_BY_CATEGORY_QUERY % (escaped_category_uri, escaped_category_uri))
    results = sparql_query.queryAndConvert()
    
    bindings = results.get("results", {}).get("bindings", [])
    if not bindings:
        return []
    
    # 1. Group all bindings by item_uri
    items_bindings = defaultdict(list)
    for result in bindings:
        # 'itemUri' is expected to be present in RETRIEVE_ITEMS_BY_CATEGORY_QUERY results
        item_uri = result.get("itemUri", {}).get("value")
        if item_uri:
            items_bindings[item_uri].append(result)
        # else: ignore rows without an item URI

    # 2. Process bindings for each unique item into an Item object
    items = []
    for item_uri, item_bindings in items_bindings.items():
        try:
            item = _process_sparql_bindings_to_item(item_bindings, item_uri=item_uri)
            items.append(item)
        except HTTPException as e:
            logger.error(f"Failed to process item {item_uri}: {e}")
            continue
    return items


def search_items(search_term: str) -> list[Item]:
    """
    Search for items across multiple fields using the comprehensive SEARCH_OBJECTS_QUERY.
    Returns a list of Item objects matching the search term.

    Args:
        search_term: The search term to look for across multiple fields.

    Returns:
        List of Item objects matching the search criteria.
    """
    # Use RDFLib's Literal to properly escape the search term for SPARQL
    escaped_search_term = Literal(search_term).n3()
    sparql_query.setQuery(
        SEARCH_OBJECTS_QUERY % (escaped_search_term, escaped_search_term, escaped_search_term, escaped_search_term)
    )

    results = sparql_query.queryAndConvert()
    
    bindings = results.get("results", {}).get("bindings", [])
    if not bindings:
        return []
    
    items_bindings = defaultdict(list)
    for result in bindings:
        identifier = result.get("identifier", {}).get("value")
        if identifier:
            items_bindings[identifier].append(result)

    items = []
    for identifier, item_bindings in items_bindings.items():
        try:
            item = _process_sparql_bindings_to_item(item_bindings, item_uri=identifier)
            items.append(item)
        except HTTPException as e:
            logger.error(f"Failed to process item {identifier}: {e}")
            continue
    return items


def get_items_by_zone(zone: str) -> list[Item]:
    """
    Fetch all items belonging to a specific exhibition zone.

    Zones are reached through the item's display case
    (``item -> scho:location -> Case -> scho:containedInPlace -> Zone``) rather than the
    superseded ``scho:isPartOf`` link, which no longer exists in the KG.

    Args:
        zone: The zone identifier. ``5``, ``Zone5``, ``Zone 5`` and a full zone URI are
              all accepted (see `zone_uri_for`).

    Returns:
        List of Item objects belonging to the specified zone. Empty if the zone is
        unknown — an absent zone is not an error, it simply holds no items.
    """
    zone_uri = URIRef(zone_uri_for(zone)).n3()
    sparql_query.setQuery(ZONE_ITEMS_QUERY % zone_uri)
    results = sparql_query.queryAndConvert()

    bindings = results.get("results", {}).get("bindings", [])
    if not bindings:
        return []

    items_bindings = defaultdict(list)
    for result in bindings:
        item_uri = result.get("itemUri", {}).get("value")
        if item_uri:
            items_bindings[item_uri].append(result)

    items = []
    for item_uri, item_bindings in items_bindings.items():
        try:
            items.append(_process_sparql_bindings_to_item(item_bindings, item_uri=item_uri))
        except HTTPException as e:
            logger.error(f"Failed to process item {item_uri}: {e}")
            continue
    return items


def get_category_triples(category_label: str, limit: int = 100) -> dict[str, str]:
    """
    Fetch raw triples for items by high-level object category using schema:category.

    The category label is resolved against the KG rather than a table of Q numbers held
    in this module. The seven categories are `ObjectCategory` values and the pipeline may
    mint or relabel them, so a hard-coded map here goes stale silently — it returns an
    empty result set, which is indistinguishable from a category that genuinely holds no
    items.

    NOTE: Renamed from get_category to clarify it returns raw triples, not Item objects.
    """
    escaped_label = Literal(category_label).n3()
    sparql_query.setQuery(
        f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX scho: <http://schema.org/>
        PREFIX wd: <http://www.wikidata.org/entity/>

        SELECT ?sub ?pred ?obj WHERE {{
            ?categoryUri rdfs:label ?categoryLabel .
            FILTER(LCASE(STR(?categoryLabel)) = LCASE({escaped_label}))
            ?sub scho:category ?categoryUri .
            ?sub ?pred ?obj .
        }} LIMIT {limit}
        """
    )

    return sparql_query.queryAndConvert()


def get_search_triples(category: str, name: str, genre: str) -> dict[str, str]:
    search_criterion = (
        f" ?sub scho:category ?category . \n" if category else ""
    )
    # Use RDFLib's Literal to properly escape the name parameter for SPARQL REGEX
    escaped_name = Literal(name).n3() if name else ""
    search_criterion += f" FILTER(REGEX(STR(?obj), {escaped_name}, 'i')) ." if name else ""

    sparql_query.setQuery(
        f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX scho: <http://schema.org/>

        SELECT ?sub ?pred ?obj WHERE {{
            ?sub ?pred ?obj .
            {search_criterion}
        }} 

    """
    )

    return sparql_query.queryAndConvert()


def delete(hash: str, session: SessionDep) -> None:
    # Use RDFLib's URIRef to properly construct and serialize the hash URI for SPARQL
    hash_uri = URIRef(f"http://radio.liv.ac.uk/bme/HumanMadeObject/{hash}").n3()
    sparql_update.setQuery(
        f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX scho: <http://schema.org/>

    DELETE {{
        {hash_uri} ?pred ?obj .
    }}
    WHERE {{
        {hash_uri} ?pred ?obj .
    }}
    """
    )
    sparql_update.queryAndConvert()
    invalidate_entity_item_counts()
    # For the SQL query, we still use the original hash with the full URI
    session.exec(
        sql_delete(ItemIdentifiers).where(
            ItemIdentifiers.hash == f"http://radio.liv.ac.uk/bme/HumanMadeObject/{hash}"
        )
    )
    session.commit()


def update(identifier: str, updating_triples: dict[str:str], session) -> None:
    """
    It is better to delete the old version of the item and add
    a newer version.
    """
    item = Item(**updating_triples)
    delete(identifier, session)
    insert(item, session)


def get_category_triples_by_schema_type(category: str, limit: int = 100) -> dict[str, str]:
    """
    Fetch raw triples for items of a specific schema.org type.
    """
    # Use RDFLib's URIRef to properly construct the schema.org type URI
    category_uri = URIRef(f"http://schema.org/{category}").n3()
    sparql_query.setQuery(
        f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX scho: <http://schema.org/>

        SELECT ?sub ?category ?pred ?obj WHERE {{
            ?sub a {category_uri} .
            BIND({category_uri} AS ?type)
            ?sub ?pred ?obj .
        }} LIMIT {limit}
        """
    )
    return sparql_query.queryAndConvert()