"""Raw SPARQL templates powering the API's read paths.

These target the **schema.org re-use as it stands in `assets/bme_kg_bundle/`**, which
differs from the shape the first version of the pipeline emitted. The differences are not
cosmetic — each one silently returned an empty binding rather than an error, so they are
recorded here with the replacement:

===========================  ==========================================================
Superseded                   Current
===========================  ==========================================================
``scho:itemLocation "15"``   ``scho:location <bme/Case/15>`` — case number is the URI
                             suffix, and the Case node carries the display-case name and
                             a ``scho:containedInPlace`` link to its Zone.
``scho:isPartOf``            *dropped.* An item has no direct exhibition link; its zone is
                             reached by ``scho:location/scho:containedInPlace``, and the
                             real ``scho:ExhibitionEvent`` (in ``core.ttl``) is then found
                             by reversing that zone's ``scho:location`` link. The
                             superseded KG's ``bme/ExhibitionEvent/Zone5`` node (label
                             only, no description/image) is gone.
``a scho:MusicArtist``       ``a scho:Person`` (284) or ``a scho:MusicGroup`` (2). Both
                             are matched, or every band drops out of ``music_artist``.
``a skos:Concept``           *dropped* from category nodes; they are ``scho:DefinedTerm``
                             only. Matching on the type alone is in any case wrong —
                             ``DefinedTerm`` also types the 135 ``additionalType`` nodes,
                             the 50 genres and the 21 materials — so categories are
                             identified by **being used as ``scho:category``**.
``scho:color wd:Q1088``      ``scho:color "blue"`` — a plain literal. The colour binding
                             is therefore not necessarily a URI; ``items.py`` reads the
                             binding's ``type`` rather than assuming one.
===========================  ==========================================================

``scho:temporalCoverage`` was never selected here, so ``Item.year`` was always ``null``
even in the superseded KG. It is selected now.

The per-item field block is shared by the single-item and by-category queries rather than
duplicated. The duplication is what let the ``skos:Concept`` fix land in one query and not
the other, and it is why the by-category endpoint stayed broken after it was declared
fixed. Edit ``_ITEM_FIELDS_BLOCK`` and both queries move together.
"""

_PREFIXES = """
PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX scho: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX mimo: <http://www.mimo-db.eu/InstrumentsKeywords/>
PREFIX euf: <http://thesaurus.europeanafashion.eu/thesaurus/>
PREFIX pbcore: <http://pbcore.org/pbcore-controlled-vocabularies/titletype-vocabulary#>
PREFIX aat: <http://vocab.getty.edu/aat/>
"""

# Kept in the order the API's Item model declares them, so a missing variable is easy to
# spot against `metamodel.Item`. `?caseUri` / `?zoneUri` replace the old `?caseNumber` /
# `?exhibitionUri`; `?year` and `?colourValue` are new.
_ITEM_SELECT_VARS = """
    ?itemUri
    ?identifier
    ?name
    ?description
    ?caption
    ?categoryUri ?categoryLabel
    ?typeUri ?typeLabel
    ?caseUri ?caseLabel
    ?zoneUri ?zoneLabel
    ?exhibitionUri ?exhibitionLabel
    ?image
    ?audio
    ?video
    ?url
    ?year
    ?genreUri ?genreLabel
    ?musicWorkUri ?musicWorkLabel
    ?musicAlbumUri ?musicAlbumLabel
    ?musicArtistUri ?musicArtistLabel
    ?eventUri ?eventLabel
    ?countryUri ?countryLabel
    ?placeUri ?placeLabel
    ?venueUri ?venueLabel
    ?colourValue ?colourLabel
    ?materialUri ?materialLabel
    ?lender
    ?lendStart
    ?lendEnd
"""

# Everything below assumes ?itemUri is already bound and typed by the caller.
_ITEM_FIELDS_BLOCK = """
    OPTIONAL { ?itemUri scho:identifier ?identifier . }
    OPTIONAL { ?itemUri scho:name ?name . }
    OPTIONAL { ?itemUri scho:description ?description . }
    OPTIONAL { ?itemUri scho:caption ?caption . }
    OPTIONAL { ?itemUri scho:image ?image . }
    OPTIONAL { ?itemUri scho:audio ?audio . }
    OPTIONAL { ?itemUri scho:video ?video . }
    OPTIONAL { ?itemUri scho:url ?url . }

    # Year (plain literals, not linked entities)
    OPTIONAL { ?itemUri scho:temporalCoverage ?year . }

    # Location: case number is the URI suffix, and the zone is the case's container.
    # Exhibition (list of LinkedEntity) is reached by reversing the real
    # ExhibitionEvent's scho:location -> Zone link (see module docstring) — a zone can in
    # principle host more than one exhibition, so this is not folded into ?zoneUri.
    OPTIONAL {
        ?itemUri scho:location ?caseUri .
        OPTIONAL { ?caseUri rdfs:label ?caseLabel . }
        OPTIONAL {
            ?caseUri scho:containedInPlace ?zoneUri .
            OPTIONAL { ?zoneUri rdfs:label ?zoneLabel . }
            OPTIONAL {
                ?exhibitionUri a scho:ExhibitionEvent ;
                               scho:location ?zoneUri .
                # scho:name is the display name ("Punk & New Romantics"); rdfs:label is
                # prefixed for disambiguation elsewhere in the graph ("Exhibition: Punk &
                # New Romantics") and would read redundantly wherever the UI already
                # labels the field "Exhibition" itself.
                OPTIONAL { ?exhibitionUri scho:name ?exhibitionName . }
                OPTIONAL { ?exhibitionUri rdfs:label ?exhibitionRdfsLabel . }
                BIND(COALESCE(?exhibitionName, ?exhibitionRdfsLabel) AS ?exhibitionLabel)
            }
        }
    }

    # Type (single LinkedEntity)
    OPTIONAL {
        ?itemUri scho:additionalType ?typeUri .
        OPTIONAL { ?typeUri rdfs:label ?typeLabel . }
    }

    # Genre (list of LinkedEntity)
    OPTIONAL {
        ?itemUri scho:genre ?genreUri .
        OPTIONAL { ?genreUri rdfs:label ?genreLabel . }
    }

    # Music Work (list of LinkedEntity via scho:about)
    OPTIONAL {
        ?itemUri scho:about ?musicWorkUri .
        ?musicWorkUri a scho:MusicComposition .
        OPTIONAL { ?musicWorkUri rdfs:label ?musicWorkLabel . }
    }

    # Music Album (list of LinkedEntity via scho:about)
    OPTIONAL {
        ?itemUri scho:about ?musicAlbumUri .
        ?musicAlbumUri a scho:MusicAlbum .
        OPTIONAL { ?musicAlbumUri rdfs:label ?musicAlbumLabel . }
    }

    # Music Artist: solo artists are scho:Person, bands are scho:MusicGroup.
    # Matching only one of the two loses the other entirely.
    OPTIONAL {
        ?itemUri scho:about ?musicArtistUri .
        ?musicArtistUri a ?musicArtistType .
        VALUES ?musicArtistType { scho:Person scho:MusicGroup }
        OPTIONAL { ?musicArtistUri rdfs:label ?musicArtistLabel . }
    }

    # Event (list of LinkedEntity via scho:about)
    OPTIONAL {
        ?itemUri scho:about ?eventUri .
        ?eventUri a scho:Event .
        OPTIONAL { ?eventUri rdfs:label ?eventLabel . }
    }

    # Country (list of LinkedEntity via scho:about)
    OPTIONAL {
        ?itemUri scho:about ?countryUri .
        ?countryUri a scho:Country .
        OPTIONAL { ?countryUri rdfs:label ?countryLabel . }
    }

    # Place (list of LinkedEntity via scho:about). Case and Zone nodes are also
    # scho:Place but hang off scho:location, so they cannot leak in here.
    OPTIONAL {
        ?itemUri scho:about ?placeUri .
        ?placeUri a scho:Place .
        OPTIONAL { ?placeUri rdfs:label ?placeLabel . }
    }

    # Venue (list of LinkedEntity via scho:about)
    OPTIONAL {
        ?itemUri scho:about ?venueUri .
        ?venueUri a scho:EventVenue .
        OPTIONAL { ?venueUri rdfs:label ?venueLabel . }
    }

    # Colour: a plain literal in the current KG, a typed URI in the superseded one.
    # Both are selected; items.py decides from the binding type which it got.
    OPTIONAL {
        ?itemUri scho:color ?colourValue .
        OPTIONAL { ?colourValue rdfs:label ?colourLabel . }
    }

    # Material (list of LinkedEntity)
    OPTIONAL {
        ?itemUri scho:material ?materialUri .
        OPTIONAL { ?materialUri rdfs:label ?materialLabel . }
    }

    # Lending Information (nested structure)
    OPTIONAL {
        ?itemUri scho:hasPart ?borrowAction .
        ?borrowAction a scho:BorrowAction .
        OPTIONAL { ?borrowAction scho:lender ?lender . }
        OPTIONAL { ?borrowAction scho:startTime ?lendStart . }
        OPTIONAL { ?borrowAction scho:endTime ?lendEnd . }
    }
"""

# %s -> an escaped identifier literal, e.g. "0261fe56ed"
RETRIEVE_OBJECT_QUERY = (
    _PREFIXES
    + "\nSELECT DISTINCT"
    + _ITEM_SELECT_VARS
    + """
WHERE {
    ?itemUri a crm:E22_Human-Made_Object ;
             scho:identifier %s . # Filter by a specific identifier

    OPTIONAL {
        ?itemUri scho:category ?categoryUri .
        OPTIONAL { ?categoryUri rdfs:label ?categoryLabel . }
    }
"""
    + _ITEM_FIELDS_BLOCK
    + "}\n"
)

# %s -> an escaped plain-string category label, matched case-insensitively.
#
# Resolving the label to a URI is a separate, cheap query, and the item query below then
# has a *bound* category to start from. Doing it in one query instead — joining the label
# match against the item pattern — makes the engine evaluate the whole per-item OPTIONAL
# block before the filter prunes anything: measured against the released KG through
# rdflib, one lookup went from 20s to under 3s by splitting it, and a label matching
# nothing cost the full 20s rather than returning immediately.
#
# The node's rdf:type is deliberately not constrained. `scho:DefinedTerm` does not
# distinguish a category from an object type, a genre or a material — it types all four —
# so what identifies a category is being used as the object of `scho:category`.
RESOLVE_CATEGORY_URI_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX scho: <http://schema.org/>

SELECT DISTINCT ?categoryUri WHERE {
    ?item scho:category ?categoryUri .
    ?categoryUri rdfs:label ?categoryLabel .
    FILTER(LCASE(STR(?categoryLabel)) = LCASE(%s))
}
"""

# %s -> an escaped category URI, e.g. <http://www.wikidata.org/entity/Q11460>
RETRIEVE_ITEMS_BY_CATEGORY_QUERY = (
    _PREFIXES
    + "\nSELECT DISTINCT"
    + _ITEM_SELECT_VARS
    + """
WHERE {
    ?itemUri a crm:E22_Human-Made_Object ;
             scho:category %s .
    BIND(%s AS ?categoryUri)
    OPTIONAL { ?categoryUri rdfs:label ?categoryLabel . }
"""
    + _ITEM_FIELDS_BLOCK
    + "}\n"
)

# %s (x4) -> the same escaped search term, once per branch of the UNION.
SEARCH_OBJECTS_QUERY = """
PREFIX scho: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>

SELECT DISTINCT ?object ?name ?description ?image ?identifier ?categoryUri ?categoryLabel
WHERE {
  # Select the core information for each object
  ?object a crm:E22_Human-Made_Object ;
          scho:name ?name ;
          scho:identifier ?identifier .
  OPTIONAL { ?object scho:description ?description . }
  OPTIONAL { ?object scho:image ?image . }
  OPTIONAL {
    ?object scho:category ?categoryUri .
    OPTIONAL { ?categoryUri rdfs:label ?categoryLabel . }
  }

  # UNION block to search across multiple fields
  {
    ?object scho:name ?text .
    FILTER(REGEX(STR(?text), %s, "i"))
  } UNION {
    ?object scho:description ?text .
    FILTER(REGEX(STR(?text), %s, "i"))
  } UNION {
    ?object scho:caption ?text .
    FILTER(REGEX(STR(?text), %s, "i"))
  } UNION {
    # This part searches the labels of linked entities (artists, bands, etc.)
    ?object scho:about/rdfs:label ?text .
    FILTER(REGEX(STR(?text), %s, "i"))
  }
}
"""

# %s -> an escaped zone URI, e.g. <http://radio.liv.ac.uk/bme/Zone/5>.
# Reached through the case rather than the superseded scho:isPartOf.
ZONE_ITEMS_QUERY = (
    _PREFIXES
    + "\nSELECT DISTINCT"
    + _ITEM_SELECT_VARS
    + """
WHERE {
    ?itemUri a crm:E22_Human-Made_Object ;
             scho:location ?caseUri .
    ?caseUri scho:containedInPlace %s .

    OPTIONAL {
        ?itemUri scho:category ?categoryUri .
        OPTIONAL { ?categoryUri rdfs:label ?categoryLabel . }
    }
"""
    + _ITEM_FIELDS_BLOCK
    + "}\n"
)

# Retained under its historical name so existing imports keep working.
EXHIBITION_ITEMS_QUERY = ZONE_ITEMS_QUERY

# %s -> an escaped entity URI, e.g. <http://radio.liv.ac.uk/bme/Person/072582f241>.
# Generic reverse lookup for every minted entity type *except* Zone, which is reached
# indirectly through the item's display case (see ZONE_ITEMS_QUERY) rather than by a
# direct predicate — Person/MusicAlbum/MusicComposition/Event/EventVenue/Place all hang
# off the item via a direct predicate (scho:about, scho:material, scho:genre, ...), so one
# unbound-predicate pattern covers all of them without one query per type.
ENTITY_ITEMS_QUERY = (
    _PREFIXES
    + "\nSELECT DISTINCT"
    + _ITEM_SELECT_VARS
    + """
WHERE {
    ?itemUri a crm:E22_Human-Made_Object ;
             ?linkPredicate %s .

    OPTIONAL {
        ?itemUri scho:category ?categoryUri .
        OPTIONAL { ?categoryUri rdfs:label ?categoryLabel . }
    }
"""
    + _ITEM_FIELDS_BLOCK
    + "}\n"
)

# %s -> an escaped entity URI. Looks up the one thing every minted node is guaranteed to
# carry: its rdfs:label (see queries/entities.py's module docstring for why that's all
# there is).
ENTITY_LABEL_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?label WHERE { %s rdfs:label ?label } LIMIT 1
"""

# %s -> an escaped Exhibition URI, e.g. <http://radio.liv.ac.uk/bme/Exhibition/PunkNewRomantics>.
# Exhibition nodes (core.ttl) are the one minted type with real editorial content, so their
# detail is read directly rather than falling back to a bare label.
EXHIBITION_DETAIL_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX scho: <http://schema.org/>

SELECT ?name ?description ?image ?zoneUri ?zoneLabel WHERE {
    OPTIONAL { %s scho:name ?name . }
    OPTIONAL { %s scho:description ?description . }
    OPTIONAL { %s scho:image ?image . }
    OPTIONAL {
        %s scho:location ?zoneUri .
        OPTIONAL { ?zoneUri rdfs:label ?zoneLabel . }
    }
}
"""

# %s -> an escaped Zone URI. Reverse of EXHIBITION_DETAIL_QUERY's scho:location: finds
# every exhibition hosted in this zone, so a Zone's detail page can list them.
EXHIBITIONS_FOR_ZONE_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX scho: <http://schema.org/>

SELECT ?exhibitionUri ?exhibitionLabel ?name ?description ?image WHERE {
    ?exhibitionUri a scho:ExhibitionEvent ;
                   scho:location %s .
    OPTIONAL { ?exhibitionUri rdfs:label ?exhibitionLabel . }
    OPTIONAL { ?exhibitionUri scho:name ?name . }
    OPTIONAL { ?exhibitionUri scho:description ?description . }
    OPTIONAL { ?exhibitionUri scho:image ?image . }
}
"""


# One scan for the whole graph, backing the `count` on each `LinkedEntity`; `items.py`
# caches the result rather than counting per chip.
#
# `scho:color` is excluded: it is a plain literal, so there is no entity to count. The two
# UNION branches re-walk the location path `_ITEM_FIELDS_BLOCK` uses for zone and
# exhibition, which an item links to indirectly.
ENTITY_ITEM_COUNTS_QUERY = (
    _PREFIXES
    + """
SELECT ?entity (COUNT(DISTINCT ?itemUri) AS ?count) WHERE {
    ?itemUri a crm:E22_Human-Made_Object .
    {
        ?itemUri ?linkPredicate ?entity .
        VALUES ?linkPredicate {
            scho:category
            scho:additionalType
            scho:genre
            scho:about
            scho:material
        }
    }
    UNION
    {
        ?itemUri scho:location/scho:containedInPlace ?entity .
    }
    UNION
    {
        ?itemUri scho:location/scho:containedInPlace ?zone .
        ?entity a scho:ExhibitionEvent ;
                scho:location ?zone .
    }
    FILTER(isIRI(?entity))
}
GROUP BY ?entity
"""
)
