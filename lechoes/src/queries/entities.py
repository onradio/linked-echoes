"""
Lookups for minted `bme:` entity nodes (Zone, Exhibition, Person, MusicAlbum,
MusicComposition, Event, EventVenue, Place, DefinedTerm, ...) — the "unlinkable"
entities the entity-linking pipeline could not match to Wikidata (see the main README's
*Neuro-symbolic entity linking* step) and so minted a local URI for instead, plus the
curated `core.ttl` Exhibition/Era nodes that were authored directly rather than mined.

Most of these nodes carry nothing but an `rdfs:label`, so a "detail" for one is just its
label plus the items that reference it. Zone and Exhibition are the two exceptions:
- Zone has a real hierarchy (`Case -> Zone`) reached via `get_items_by_zone`, and is
  cross-linked to whichever Exhibition(s) list it as their `scho:location`.
- Exhibition (`core.ttl`) carries real editorial content — name, description, image — and
  is cross-linked to its Zone(s) the same way, in the other direction.
"""

from collections import defaultdict

from fastapi import HTTPException
from rdflib import URIRef

from lechoes import sparql_query, logger
from lechoes.src.queries.items import (
    BME_NAMESPACE,
    entity_item_counts,
    _process_sparql_bindings_to_item,
    get_items_by_zone,
)
from lechoes.src.queries.sparql_queries import (
    ENTITY_ITEMS_QUERY,
    ENTITY_LABEL_QUERY,
    EXHIBITION_DETAIL_QUERY,
    EXHIBITIONS_FOR_ZONE_QUERY,
)


def _get_items_referencing(entity_uri_n3: str) -> list:
    sparql_query.setQuery(ENTITY_ITEMS_QUERY % entity_uri_n3)
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


def _get_exhibitions_for_zone(zone_uri_n3: str) -> list:
    """
    Every ExhibitionEvent whose `scho:location` is this zone, as plain dicts (not
    LinkedEntity — these back a Zone's own detail payload, not an Item field).
    """
    sparql_query.setQuery(EXHIBITIONS_FOR_ZONE_QUERY % zone_uri_n3)
    results = sparql_query.queryAndConvert()
    bindings = results.get("results", {}).get("bindings", [])

    exhibitions = {}
    for row in bindings:
        uri = row.get("exhibitionUri", {}).get("value")
        if not uri or uri in exhibitions:
            continue
        exhibitions[uri] = {
            "uri": uri,
            "count": entity_item_counts().get(uri),
            "label": row.get("name", {}).get("value") or row.get("exhibitionLabel", {}).get("value") or uri,
            "description": row.get("description", {}).get("value"),
            "image": row.get("image", {}).get("value"),
        }
    return list(exhibitions.values())


def _get_exhibition_detail(entity_uri: str, entity_uri_n3: str, fallback_label: str) -> dict:
    """
    An Exhibition's own name/description/image plus the Zone(s) it is hosted in
    (`scho:location`), matching `_get_exhibitions_for_zone`'s reverse direction.
    """
    sparql_query.setQuery(EXHIBITION_DETAIL_QUERY % ((entity_uri_n3,) * 4))
    results = sparql_query.queryAndConvert()
    bindings = results.get("results", {}).get("bindings", [])

    name = None
    description = None
    image = None
    zones = {}
    for row in bindings:
        name = name or row.get("name", {}).get("value")
        description = description or row.get("description", {}).get("value")
        image = image or row.get("image", {}).get("value")
        zone_uri = row.get("zoneUri", {}).get("value")
        if zone_uri and zone_uri not in zones:
            zones[zone_uri] = {
                "uri": zone_uri,
                "count": entity_item_counts().get(zone_uri),
                "label": row.get("zoneLabel", {}).get("value") or zone_uri,
            }

    return {
        "label": name or fallback_label,
        "description": description,
        "image": image,
        "zones": list(zones.values()),
    }


def get_entity_detail(entity_type: str, entity_id: str) -> dict:
    """
    Fetch a minted (or curated) entity's detail and the items that reference it.

    Args:
        entity_type: The node's class as it appears in its URI, e.g. ``Person``,
                     ``MusicAlbum``, ``Zone``, ``Exhibition``.
        entity_id: The URI's trailing segment, e.g. ``072582f241`` or ``5``.

    Returns:
        ``{"type", "label", "items", ...}``, with ``image``/``description``/``exhibitions``/
        ``zones`` present only when the entity actually carries them. An unknown entity
        returns an empty item list and its id as a fallback label, rather than a 404 — the
        same "absent is not an error" stance `get_items_by_zone` already takes.
    """
    entity_uri = f"{BME_NAMESPACE}{entity_type}/{entity_id}"
    entity_uri_n3 = URIRef(entity_uri).n3()

    sparql_query.setQuery(ENTITY_LABEL_QUERY % entity_uri_n3)
    label_bindings = sparql_query.queryAndConvert()["results"]["bindings"]
    label = label_bindings[0]["label"]["value"] if label_bindings else entity_id

    result = {"type": entity_type, "label": label}

    if entity_type == "Zone":
        result["items"] = get_items_by_zone(entity_id)
        exhibitions = _get_exhibitions_for_zone(entity_uri_n3)
        if exhibitions:
            result["exhibitions"] = exhibitions
            # The zone itself carries no image/description in the graph; borrow the
            # first hosted exhibition's for the page banner.
            result["image"] = exhibitions[0].get("image")
            result["description"] = exhibitions[0].get("description")
    elif entity_type == "Exhibition":
        detail = _get_exhibition_detail(entity_uri, entity_uri_n3, label)
        result["label"] = detail["label"]
        result["description"] = detail["description"]
        result["image"] = detail["image"]
        result["zones"] = detail["zones"]
        items = []
        seen = set()
        for zone in detail["zones"]:
            zone_id = zone["uri"].rstrip("/").rsplit("/", 1)[-1]
            for item in get_items_by_zone(zone_id):
                if item.identifier not in seen:
                    seen.add(item.identifier)
                    items.append(item)
        result["items"] = items
    else:
        result["items"] = _get_items_referencing(entity_uri_n3)

    return result
