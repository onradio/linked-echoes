"""Exhibition lookups.

`core.ttl` carries real `scho:ExhibitionEvent` nodes — name, dates, description, image —
each linked to the zone it is hosted in via `scho:location`. This used to be modelled by
querying the **zone** instead (the superseded KG's `bme/ExhibitionEvent/Zone5` node had a
label and nothing else, and items reached it via `scho:isPartOf`), which is why the
predicate/object shape below still fans out generically rather than selecting named
fields: the dashboard/CMS-facing shape it feeds (`routes/exhibitions.py`) hasn't changed,
only which node the walk starts from.
"""

from rdflib import URIRef

from lechoes import sparql_query
from lechoes.src.queries.items import zone_uri_for


EXHIBITIONS_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX scho: <http://schema.org/>

SELECT ?exhibition ?pred ?obj WHERE {
    ?exhibition a scho:ExhibitionEvent .
    ?exhibition ?pred ?obj .
} LIMIT %s
"""

EXHIBITION_BY_ZONE_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX scho: <http://schema.org/>

SELECT ?exhibition ?pred ?obj WHERE {
    ?exhibition a scho:ExhibitionEvent ;
                scho:location %s .
    ?exhibition ?pred ?obj .
}
"""


def get_exhibitions(limit: int = 100) -> dict[str, str]:
    """Fetch all real exhibitions (`core.ttl` ExhibitionEvent nodes) from the KG."""
    sparql_query.setQuery(EXHIBITIONS_QUERY % int(limit))
    return sparql_query.queryAndConvert()


def get_exhibition_by_zone(zone: str) -> dict[str, str]:
    """
    Fetch the exhibition(s) hosted in a zone, by zone identifier.

    Accepts the same spellings as `get_items_by_zone` — ``5``, ``Zone5``, ``Zone 5`` or a
    full URI.
    """
    zone_uri = URIRef(zone_uri_for(zone)).n3()
    sparql_query.setQuery(EXHIBITION_BY_ZONE_QUERY % zone_uri)
    return sparql_query.queryAndConvert()
