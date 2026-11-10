from lechoes import sparql_query


# Categories are identified by *being used* as `scho:category`, not by their rdf:type.
#
# They were once typed `scho:DefinedTerm, skos:Concept` and the query asked for both.
# `skos:Concept` is gone from the current KG, but simply dropping it is not enough:
# `scho:DefinedTerm` types 213 nodes, of which only 7 are categories — the other 206 are
# the object types behind `scho:additionalType` (135), the genres (50) and the materials
# (21). Selecting on the type alone therefore swings from returning nothing to returning
# thirty times too much, and both failures look plausible from the endpoint.
#
# The `scho:category` join is what actually distinguishes a category, and it degrades
# well: a category nobody has classified anything into is not in the list, which is the
# behaviour the CMS's category picker wants anyway.
CATEGORIES_QUERY = """
PREFIX scho: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?category ?categoryLabel WHERE {
    ?item scho:category ?category .
    ?category rdfs:label ?categoryLabel .
}
ORDER BY ?categoryLabel
"""


def get_categories():
    """
    Fetch all categories in use in the knowledge graph, as (uri, label) pairs.

    The language filter the previous version applied (`lang(?categoryLabel) = "en"`) is
    left off: every label in the KG is tagged @en, so the filter only ever had the power
    to drop a category whose label had lost its tag.
    """
    sparql_query.setQuery(CATEGORIES_QUERY)
    return sparql_query.queryAndConvert()
