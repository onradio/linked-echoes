"""API tests: every query function, against the released knowledge graph.

    pytest lechoes/test_api.py            # from the repository root
    pytest lechoes/test_api.py -k zone    # one group

Offline and no API key: `conftest.py` loads the BME KG's `knowledge_graph.ttl` (from a
`bme-kg` checkout — `BME_KG_DIR`, default `../bme-kg`) into an in-memory rdflib graph and binds the API's SPARQL endpoints to it, so the
templates in `sparql_queries.py` are executed verbatim against real data. A query that is
malformed, or that asks for a predicate the KG no longer carries, fails here.

**Why these particular assertions.** The KG's schema.org re-use changed after the API was
written, and every one of the changes failed *silently* — an unmatched triple pattern
binds nothing, so the endpoint returned a well-formed response with a null field or an
empty list rather than an error. Each group below therefore pins the affected field to a
value counted from the KG, and the counts are spelled out rather than derived from the
same query under test: a test that recomputes its expectation from the graph would agree
with any regression that changed both sides.

The superseded shapes are also asserted *absent* from the query templates
(`TestNoSupersededPredicates`), because the failure mode being guarded against is a
predicate quietly coming back, not a query that stops parsing.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from rdflib import RDF, RDFS, Graph, Literal, URIRef

from lechoes.src.metamodel import Item, LinkedEntity
from lechoes.src.queries import categories as categories_queries
from lechoes.src.queries import exhibitions as exhibitions_queries
from lechoes.src.queries import items as items_queries
from lechoes.src.queries import sparql_queries

SCHO = "http://schema.org/"
BME = "http://radio.liv.ac.uk/bme"

# An item chosen because it exercises every field that the KG change broke at once:
# a case (15) reached through scho:location, a zone (5) reached through the case, a
# literal colour, a temporalCoverage year, and a linked artist.
REFERENCE_ID = "0261fe56ed"

# Linked to The Beatles, which is typed scho:MusicGroup rather than scho:Person.
BAND_ITEM_ID = "03e1051440"

# Counted from bme-kg's knowledge_graph.ttl, kg-v1.1.0 (502 objects in total).
ITEMS_PER_CATEGORY = {
    "Textual and Graphic Works": 237,
    "Audiovisual Media": 118,
    "Clothing": 67,
    "Musical Instrument": 54,
    "Merchandise": 17,
    "Personal Effects": 7,
    "Award and Recognition": 2,
}

# Zone -> item count, via item -> scho:location -> Case -> scho:containedInPlace -> Zone.
ITEMS_PER_ZONE = {"1": 94, "2": 87, "3": 51, "4": 39, "5": 75, "6": 69, "7": 72, "8": 15}

TOTAL_ITEMS = 502


def links(entities):
    """
    Reduce LinkedEntity values to the (uri, label) pairs these tests are about.

    `LinkedEntity` also carries a `count`, a property of the whole fixture rather than of
    the item under test; `TestEntityLinkCounts` covers those.
    """
    if entities is None:
        return None
    if isinstance(entities, LinkedEntity):
        return (entities.uri, entities.label)
    return [(entity.uri, entity.label) for entity in entities]


# ======================================================================================
# Guards against the superseded shapes coming back
# ======================================================================================

class TestNoSupersededPredicates:
    """The templates must not reference anything the KG dropped.

    Each of these once cost a field or a whole endpoint, and none of them raised: the
    triple pattern simply matched nothing.
    """

    ITEM_QUERIES = (
        "RETRIEVE_OBJECT_QUERY",
        "RETRIEVE_ITEMS_BY_CATEGORY_QUERY",
        "ZONE_ITEMS_QUERY",
    )

    @pytest.mark.parametrize("name", ITEM_QUERIES)
    @pytest.mark.parametrize(
        "superseded",
        ["scho:itemLocation", "scho:isPartOf", "scho:MusicArtist", "skos:Concept"],
    )
    def test_item_queries_do_not_use_superseded_terms(self, name, superseded):
        assert superseded not in getattr(sparql_queries, name)

    def test_category_listing_does_not_filter_on_skos_concept(self):
        assert "skos:Concept" not in categories_queries.CATEGORIES_QUERY

    def test_exhibitions_query_real_exhibition_event_nodes(self):
        """core.ttl's real ExhibitionEvent nodes replace the zone-as-exhibition
        stand-in — see queries/exhibitions.py's module docstring."""
        assert "ExhibitionEvent" in exhibitions_queries.EXHIBITIONS_QUERY
        assert "ExhibitionEvent" in exhibitions_queries.EXHIBITION_BY_ZONE_QUERY

    @pytest.mark.parametrize("name", ITEM_QUERIES)
    def test_item_queries_select_the_recovered_fields(self, name):
        query = getattr(sparql_queries, name)
        for variable in ("?caseUri", "?zoneUri", "?year", "?colourValue", "?musicArtistUri"):
            assert variable in query

    def test_both_item_queries_share_one_field_block(self):
        """The single-item and by-category queries must not drift apart again.

        They were near-duplicates, which is how the `skos:Concept` fix landed in one and
        not the other and left the by-category endpoint broken after it was reported
        fixed. They are now composed from the same constant.
        """
        block = sparql_queries._ITEM_FIELDS_BLOCK
        for name in self.ITEM_QUERIES:
            assert block in getattr(sparql_queries, name)


# ======================================================================================
# Helpers
# ======================================================================================

class TestZoneUriNormalisation:
    @pytest.mark.parametrize(
        "given",
        ["5", " 5 ", "Zone5", "Zone 5", "zone5", "zone-5", "ZONE_5"],
    )
    def test_accepted_spellings_resolve_to_one_uri(self, given):
        assert items_queries.zone_uri_for(given) == f"{BME}/Zone/5"

    def test_a_full_uri_passes_through(self):
        assert items_queries.zone_uri_for(f"{BME}/Zone/7") == f"{BME}/Zone/7"

    def test_an_unknown_zone_still_produces_a_uri(self):
        """Normalisation does not validate; the query returns nothing for a bad zone."""
        assert items_queries.zone_uri_for("99") == f"{BME}/Zone/99"


class TestCaseNumberFromUri:
    def test_extracts_the_suffix(self):
        assert items_queries.case_number_from_uri(f"{BME}/Case/15") == "15"

    def test_tolerates_a_trailing_slash(self):
        assert items_queries.case_number_from_uri(f"{BME}/Case/15/") == "15"

    @pytest.mark.parametrize("empty", [None, ""])
    def test_absent_location_is_none(self, empty):
        assert items_queries.case_number_from_uri(empty) is None


class TestBindingPair:
    """`_binding_pair` is what makes a literal-valued colour survive."""

    def test_uri_with_label(self):
        row = {"u": {"type": "uri", "value": "http://x/1"}, "l": {"type": "literal", "value": "blue"}}
        assert items_queries._binding_pair(row, "u", "l") == ("http://x/1", "blue")

    def test_literal_binding_is_its_own_label(self):
        row = {"u": {"type": "literal", "value": "blue"}}
        assert items_queries._binding_pair(row, "u", "l") == (None, "blue")

    def test_uri_without_a_label_is_dropped(self):
        row = {"u": {"type": "uri", "value": "http://x/1"}}
        assert items_queries._binding_pair(row, "u", "l") is None

    def test_absent_binding_is_none(self):
        assert items_queries._binding_pair({}, "u", "l") is None


# ======================================================================================
# get_item_by_identifier
# ======================================================================================

class TestGetItemByIdentifier:
    def test_returns_the_item(self, kg):
        item = items_queries.get_item_by_identifier(REFERENCE_ID)
        assert isinstance(item, Item)
        assert item.identifier == REFERENCE_ID
        assert item.name == "Blue silk suit designed by Anthony Price"

    def test_case_number_comes_from_the_location_uri(self, kg):
        """Was always None: the query asked for `scho:itemLocation`, a literal that the
        KG replaced with `scho:location <bme/Case/15>`."""
        item = items_queries.get_item_by_identifier(REFERENCE_ID)
        assert item.case_number == "15"

    def test_case_name_comes_from_the_case_label(self, kg):
        """The case's own rdfs:label (`core.ttl`), for grouping items by their physical
        display case rather than just its number."""
        item = items_queries.get_item_by_identifier(REFERENCE_ID)
        assert item.case_name == "New Romantics"

    def test_location_is_the_zone_of_the_case(self, kg):
        """Was always None: `scho:isPartOf` no longer exists, and the zone reached
        through the display case is what replaces it."""
        item = items_queries.get_item_by_identifier(REFERENCE_ID)
        assert links(item.location) == [(f"{BME}/Zone/5", "Zone 5")]

    def test_exhibition_is_the_real_exhibition_hosted_in_that_zone(self, kg):
        """Not the zone itself: the real `core.ttl` ExhibitionEvent found by reversing
        its `scho:location` link against the item's zone."""
        item = items_queries.get_item_by_identifier(REFERENCE_ID)
        assert links(item.exhibition) == [
            (f"{BME}/Exhibition/PunkNewRomantics", "Punk & New Romantics")
        ]

    def test_music_artist_matches_people(self, kg):
        """Was always None: artists are `scho:Person` now, not `scho:MusicArtist`."""
        item = items_queries.get_item_by_identifier(REFERENCE_ID)
        labels = sorted(entity.label for entity in item.music_artist)
        assert labels == ["Duran Duran", "Roger Taylor"]

    def test_music_artist_also_matches_bands(self, kg):
        """A band is `scho:MusicGroup`; matching only `scho:Person` loses it silently."""
        item = items_queries.get_item_by_identifier(BAND_ITEM_ID)
        assert "The Beatles" in [entity.label for entity in item.music_artist]

    def test_colour_survives_as_a_literal(self, kg):
        """Was always None: colour is a plain literal now, and the previous code
        required both a URI *and* a label before it would keep a value."""
        item = items_queries.get_item_by_identifier(REFERENCE_ID)
        assert item.colour == [LinkedEntity(uri=None, label="blue")]

    def test_year_comes_from_temporal_coverage(self, kg):
        """Was always None: `scho:temporalCoverage` was never selected at all."""
        item = items_queries.get_item_by_identifier(REFERENCE_ID)
        assert item.year == ["1982"]

    def test_the_fields_that_already_worked_still_do(self, kg):
        item = items_queries.get_item_by_identifier(REFERENCE_ID)
        assert links(item.category) == ("http://www.wikidata.org/entity/Q11460", "Clothing")
        assert links(item.type) == ("http://thesaurus.europeanafashion.eu/thesaurus/10015", "suit")
        assert [entity.label for entity in item.genre] == ["New Romantic"]
        assert [entity.label for entity in item.material] == ["silk fabric"]
        assert item.description and item.caption

    def test_media_and_lending_are_absent_from_this_kg(self, kg):
        """Not a defect in the queries: the base layer carries no image, audio, video,
        url or BorrowAction triples at all. Pinned so that a future layer that adds them
        shows up as a failure here rather than going unnoticed."""
        item = items_queries.get_item_by_identifier(REFERENCE_ID)
        assert item.image is None and item.audio is None and item.video is None
        assert item.lender is None and item.lend_start is None and item.lend_end is None

    def test_unknown_identifier_raises_404(self, kg):
        with pytest.raises(HTTPException) as excinfo:
            items_queries.get_item_by_identifier("does-not-exist")
        assert excinfo.value.status_code == 404

    def test_empty_identifier_raises_404(self, kg):
        with pytest.raises(HTTPException) as excinfo:
            items_queries.get_item_by_identifier("")
        assert excinfo.value.status_code == 404

    def test_a_quote_in_the_identifier_does_not_break_the_query(self, kg):
        """The identifier is escaped through rdflib rather than interpolated raw."""
        with pytest.raises(HTTPException) as excinfo:
            items_queries.get_item_by_identifier('" } UNION { ?s ?p ?o . #')
        assert excinfo.value.status_code == 404


# ======================================================================================
# Categories
# ======================================================================================

class TestGetCategories:
    def test_lists_every_category_in_use(self, kg):
        bindings = categories_queries.get_categories()["results"]["bindings"]
        labels = sorted(row["categoryLabel"]["value"] for row in bindings)
        assert labels == sorted(ITEMS_PER_CATEGORY)

    def test_does_not_leak_other_defined_terms(self, kg):
        """`scho:DefinedTerm` types 213 nodes but only 7 are categories — the rest are
        object types, genres and materials. Selecting on the type alone returns all of
        them, which is as wrong as returning none."""
        bindings = categories_queries.get_categories()["results"]["bindings"]
        labels = {row["categoryLabel"]["value"] for row in bindings}
        assert len(labels) == 7
        for intruder in ("suit", "ska", "silk fabric", "New Romantic"):
            assert intruder not in labels

    def test_each_category_carries_a_wikidata_uri(self, kg):
        bindings = categories_queries.get_categories()["results"]["bindings"]
        assert all(row["category"]["value"].startswith("http://www.wikidata.org/entity/Q") for row in bindings)


class TestResolveCategoryUri:
    def test_resolves_a_known_label(self, kg):
        assert items_queries.resolve_category_uri("Clothing") == "http://www.wikidata.org/entity/Q11460"

    def test_is_case_insensitive(self, kg):
        assert items_queries.resolve_category_uri("clothing") == items_queries.resolve_category_uri("Clothing")

    def test_unknown_label_is_none(self, kg):
        assert items_queries.resolve_category_uri("Not A Category") is None


class TestGetItemsByCategory:
    def test_every_category_returns_its_items(self, kg):
        for label, expected in ITEMS_PER_CATEGORY.items():
            assert len(items_queries.get_items_by_category(label)) == expected, label

    def test_the_categories_partition_the_collection(self, kg):
        assert sum(ITEMS_PER_CATEGORY.values()) == TOTAL_ITEMS

    def test_items_carry_the_category_asked_for(self, kg):
        for item in items_queries.get_items_by_category("Musical Instrument"):
            assert item.category.label == "Musical Instrument"

    def test_items_are_fully_populated(self, kg):
        """The by-category query must return the same fields as the single-item one —
        they are built from the same block precisely so this holds."""
        items = items_queries.get_items_by_category("Clothing")
        subject = next(item for item in items if item.identifier == REFERENCE_ID)
        assert subject.case_number == "15"
        assert links(subject.location) == [(f"{BME}/Zone/5", "Zone 5")]
        assert links(subject.exhibition) == [
            (f"{BME}/Exhibition/PunkNewRomantics", "Punk & New Romantics")
        ]
        assert subject.colour == [LinkedEntity(uri=None, label="blue")]
        assert subject.year == ["1982"]

    def test_is_case_insensitive(self, kg):
        assert len(items_queries.get_items_by_category("clothing")) == ITEMS_PER_CATEGORY["Clothing"]

    def test_unknown_category_returns_empty_list(self, kg):
        assert items_queries.get_items_by_category("Not A Category") == []

    def test_empty_category_returns_empty_list(self, kg):
        assert items_queries.get_items_by_category("") == []


class TestGetCategoryTriples:
    def test_returns_raw_triples_for_a_known_category(self, kg):
        bindings = items_queries.get_category_triples("Clothing", limit=50)["results"]["bindings"]
        assert len(bindings) == 50
        assert {"sub", "pred", "obj"} <= set(bindings[0])

    def test_respects_the_limit(self, kg):
        assert len(items_queries.get_category_triples("Clothing", limit=5)["results"]["bindings"]) == 5

    def test_unknown_category_returns_no_bindings(self, kg):
        assert items_queries.get_category_triples("Not A Category")["results"]["bindings"] == []


# ======================================================================================
# Zones / exhibitions
# ======================================================================================

class TestGetItemsByZone:
    def test_every_zone_returns_its_items(self, kg):
        for zone, expected in ITEMS_PER_ZONE.items():
            assert len(items_queries.get_items_by_zone(zone)) == expected, zone

    def test_the_zones_partition_the_collection(self, kg):
        assert sum(ITEMS_PER_ZONE.values()) == TOTAL_ITEMS

    def test_spellings_agree(self, kg):
        by_number = {item.identifier for item in items_queries.get_items_by_zone("5")}
        by_name = {item.identifier for item in items_queries.get_items_by_zone("Zone 5")}
        assert by_number == by_name and len(by_number) == ITEMS_PER_ZONE["5"]

    def test_items_report_the_zone_they_were_asked_for(self, kg):
        for item in items_queries.get_items_by_zone("5"):
            assert links(item.location) == [(f"{BME}/Zone/5", "Zone 5")]
            assert links(item.exhibition) == [
                (f"{BME}/Exhibition/PunkNewRomantics", "Punk & New Romantics")
            ]
            assert item.case_number in {"13", "14", "15"}

    def test_items_are_fully_populated(self, kg):
        items = items_queries.get_items_by_zone("5")
        subject = next(item for item in items if item.identifier == REFERENCE_ID)
        assert subject.category.label == "Clothing"
        assert subject.colour == [LinkedEntity(uri=None, label="blue")]

    def test_unknown_zone_returns_empty_list(self, kg):
        assert items_queries.get_items_by_zone("99") == []

    def test_nonsense_zone_returns_empty_list(self, kg):
        assert items_queries.get_items_by_zone("not-a-zone") == []


class TestGetExhibitions:
    def test_lists_every_exhibition(self, kg):
        bindings = exhibitions_queries.get_exhibitions()["results"]["bindings"]
        uris = {row["exhibition"]["value"] for row in bindings}
        assert uris == {
            f"{BME}/Exhibition/{name}"
            for name in (
                "PostWarBoom", "SwingingLondon", "CulturalRevolution", "GlamRockRockSpectacle",
                "PunkNewRomantics", "MadchesterLiveAid", "BritpopGirlPower", "TheNewMillennium",
            )
        }

    def test_respects_the_limit(self, kg):
        assert len(exhibitions_queries.get_exhibitions(limit=3)["results"]["bindings"]) == 3

    def test_by_zone_returns_the_exhibition_hosted_there(self, kg):
        bindings = exhibitions_queries.get_exhibition_by_zone("5")["results"]["bindings"]
        assert {row["exhibition"]["value"] for row in bindings} == {f"{BME}/Exhibition/PunkNewRomantics"}
        assert "Punk & New Romantics" in {row["obj"]["value"] for row in bindings}

    @pytest.mark.parametrize("spelling", ["5", "Zone5", "Zone 5"])
    def test_by_zone_accepts_the_usual_spellings(self, kg, spelling):
        bindings = exhibitions_queries.get_exhibition_by_zone(spelling)["results"]["bindings"]
        assert {row["exhibition"]["value"] for row in bindings} == {f"{BME}/Exhibition/PunkNewRomantics"}

    def test_unknown_zone_returns_no_bindings(self, kg):
        assert exhibitions_queries.get_exhibition_by_zone("99")["results"]["bindings"] == []


# ======================================================================================
# Search
# ======================================================================================

class TestSearchItems:
    def test_matches_a_name(self, kg):
        identifiers = {item.identifier for item in items_queries.search_items("Blue silk suit")}
        assert REFERENCE_ID in identifiers

    def test_matches_a_linked_entity_label(self, kg):
        """The `scho:about/rdfs:label` branch — an item whose *artist* matches."""
        identifiers = {item.identifier for item in items_queries.search_items("Duran Duran")}
        assert REFERENCE_ID in identifiers

    def test_is_case_insensitive(self, kg):
        assert {item.identifier for item in items_queries.search_items("beatles")} == {
            item.identifier for item in items_queries.search_items("Beatles")
        }

    def test_results_are_items(self, kg):
        for item in items_queries.search_items("Duran Duran"):
            assert isinstance(item, Item) and item.identifier and item.name

    def test_no_match_returns_empty_list(self, kg):
        assert items_queries.search_items("zzzz-no-such-thing") == []

    def test_a_regex_metacharacter_does_not_raise(self, kg):
        """The term reaches a REGEX; a lone bracket must not take the endpoint down."""
        assert isinstance(items_queries.search_items("Help!"), list)


# ======================================================================================
# Writes
# ======================================================================================

def _new_item(identifier="test-insert-01", case_number="15"):
    return Item(
        identifier=identifier,
        name="Test guitar",
        description="An item inserted by the test suite",
        category=LinkedEntity(uri="http://www.wikidata.org/entity/Q34379", label="Musical Instrument"),
        case_number=case_number,
        colour=[LinkedEntity(label="blue")],
        year=["1984"],
        music_artist=[LinkedEntity(uri="http://www.wikidata.org/entity/Q58381", label="Duran Duran")],
        material=[LinkedEntity(uri="http://www.wikidata.org/entity/Q12019351", label="silk fabric")],
        # Supplied to prove neither is *written*: both are derived from the case on read.
        location=[LinkedEntity(uri=f"{BME}/Zone/5", label="Zone 5")],
        exhibition=[LinkedEntity(uri=f"{BME}/Exhibition/PunkNewRomantics", label="Exhibition: Punk & New Romantics")],
    )


class TestInsert:
    def test_writes_the_current_location_shape(self, kg, session):
        items_queries.insert(_new_item(), session)
        subject = URIRef(f"{BME}/HumanMadeObject/test-insert-01")
        assert (subject, URIRef(SCHO + "location"), URIRef(f"{BME}/Case/15")) in kg.graph
        assert not list(kg.graph.objects(subject, URIRef(SCHO + "itemLocation")))
        assert not list(kg.graph.objects(subject, URIRef(SCHO + "isPartOf")))

    def test_places_the_item_in_the_zone_of_its_case(self, kg, session):
        """`Triplifier` cannot infer the zone from the payload — the zone is a property
        of the case — so `insert` reads the case/zone hierarchy out of the KG. Without
        that the item lands under `bme/Zone/unknown` and no zone query ever finds it."""
        items_queries.insert(_new_item(), session)
        case = URIRef(f"{BME}/Case/15")
        zones = set(kg.graph.objects(case, URIRef(SCHO + "containedInPlace")))
        assert zones == {URIRef(f"{BME}/Zone/5")}
        # The tell-tale of a missing lookup: Triplifier's own fallback zone.
        assert not list(kg.graph.subjects(RDFS.label, Literal("Zone unknown", lang="en")))

    def test_the_inserted_item_reads_back(self, kg, session):
        items_queries.insert(_new_item(), session)
        item = items_queries.get_item_by_identifier("test-insert-01")
        assert item.name == "Test guitar"
        assert item.case_number == "15"
        assert links(item.location) == [(f"{BME}/Zone/5", "Zone 5")]
        assert links(item.exhibition) == [
            (f"{BME}/Exhibition/PunkNewRomantics", "Punk & New Romantics")
        ]
        assert item.colour == [LinkedEntity(uri=None, label="blue")]
        assert item.year == ["1984"]
        assert [entity.label for entity in item.material] == ["silk fabric"]

    def test_the_inserted_item_appears_in_its_zone(self, kg, session):
        items_queries.insert(_new_item(), session)
        identifiers = {item.identifier for item in items_queries.get_items_by_zone("5")}
        assert "test-insert-01" in identifiers

    def test_the_inserted_item_appears_in_its_category(self, kg, session):
        items_queries.insert(_new_item(), session)
        identifiers = {item.identifier for item in items_queries.get_items_by_category("Musical Instrument")}
        assert "test-insert-01" in identifiers

    def test_records_the_hash_in_the_relational_side(self, kg, session):
        items_queries.insert(_new_item(), session)
        assert len(session.added) == 1
        # `Triplifier.uri` is an rdflib URIRef; `delete` looks the row up by the same
        # full URI, so the two sides agree regardless of the type.
        assert str(session.added[0].hash) == f"{BME}/HumanMadeObject/test-insert-01"
        assert session.commits == 1

    def test_a_duplicate_is_rejected(self, kg):
        from conftest import FakeSession

        with pytest.raises(HTTPException) as excinfo:
            items_queries.insert(_new_item(), FakeSession(existing_hashes={"already-there"}))
        assert excinfo.value.status_code == 400

    def test_an_unknown_case_still_inserts(self, kg, session):
        """A case the KG has never seen has no zone. The item must still be written —
        it simply is not reachable by zone until the case is described."""
        items_queries.insert(_new_item(identifier="test-insert-02", case_number="404"), session)
        item = items_queries.get_item_by_identifier("test-insert-02")
        assert item.case_number == "404"

    def test_insert_does_not_reintroduce_skos_concept(self, kg, session):
        """`mappings_min.json` used to list skos:Concept under `category._type_uri`, so
        every insert re-typed the category with a class the pipeline carries nowhere.
        `core.ttl` itself types the 7 top-level categories as `skos:Concept` too (a real,
        curated fact, not a pipeline artefact) — so what this guards against is `insert`
        adding the triple, not the triple ever being present."""
        category = URIRef("http://www.wikidata.org/entity/Q34379")
        concept_triple = (category, RDF.type, URIRef("http://www.w3.org/2004/02/skos/core#Concept"))
        was_present_before = concept_triple in kg.graph

        items_queries.insert(_new_item(), session)

        assert (concept_triple in kg.graph) == was_present_before
        assert (category, RDF.type, URIRef(SCHO + "DefinedTerm")) in kg.graph

    def test_insert_types_material_as_a_defined_term(self, kg, session):
        """The superseded type, wd:Q214609, types nothing in the released KG."""
        items_queries.insert(_new_item(), session)
        material = URIRef("http://www.wikidata.org/entity/Q12019351")
        assert (material, RDF.type, URIRef(SCHO + "DefinedTerm")) in kg.graph
        assert (material, RDF.type, URIRef("http://www.wikidata.org/entity/Q214609")) not in kg.graph

    def test_insert_writes_nothing_for_the_exhibition_field(self, kg, session):
        """The exhibition is derived from the case's zone, so it is reconstructed on read
        and never stored. An `exhibition` in the payload must not resurrect
        `scho:isPartOf`, nor add a new `scho:ExhibitionEvent` node beyond the 8 real ones
        `core.ttl` already carries."""
        exhibitions_before = set(kg.graph.subjects(RDF.type, URIRef(SCHO + "ExhibitionEvent")))
        items_queries.insert(_new_item(), session)
        subject = URIRef(f"{BME}/HumanMadeObject/test-insert-01")
        assert not list(kg.graph.objects(subject, URIRef(SCHO + "isPartOf")))
        assert set(kg.graph.subjects(RDF.type, URIRef(SCHO + "ExhibitionEvent"))) == exhibitions_before

    def test_the_exhibition_still_reads_back_from_the_case(self, kg, session):
        """Not writing it costs nothing: both are derived from the case on read."""
        items_queries.insert(_new_item(), session)
        item = items_queries.get_item_by_identifier("test-insert-01")
        assert links(item.location) == [(f"{BME}/Zone/5", "Zone 5")]
        assert links(item.exhibition) == [
            (f"{BME}/Exhibition/PunkNewRomantics", "Punk & New Romantics")
        ]


class TestDelete:
    def test_removes_the_items_triples(self, kg, session):
        items_queries.insert(_new_item(), session)
        subject = URIRef(f"{BME}/HumanMadeObject/test-insert-01")
        assert list(kg.graph.predicate_objects(subject))

        items_queries.delete("test-insert-01", session)
        assert not list(kg.graph.predicate_objects(subject))

    def test_the_deleted_item_is_no_longer_retrievable(self, kg, session):
        items_queries.insert(_new_item(), session)
        items_queries.delete("test-insert-01", session)
        with pytest.raises(HTTPException) as excinfo:
            items_queries.get_item_by_identifier("test-insert-01")
        assert excinfo.value.status_code == 404

    def test_deleting_an_absent_item_is_a_no_op(self, kg, session):
        before = len(kg.graph)
        items_queries.delete("never-existed", session)
        assert len(kg.graph) == before

    def test_deleting_leaves_the_rest_of_the_collection_alone(self, kg, session):
        items_queries.insert(_new_item(), session)
        items_queries.delete("test-insert-01", session)
        assert items_queries.get_item_by_identifier(REFERENCE_ID).identifier == REFERENCE_ID


# ======================================================================================
# Route handlers — the response shape the dashboard and CMS consume
# ======================================================================================

class TestRouteHandlers:
    """The handlers are called directly rather than through `lechoes.main`.

    `main.py` registers its startup hooks with `app.add_event_handler`, which the FastAPI
    version in this environment no longer provides (the deployment pins fastapi 0.115.6 /
    starlette 0.41.3, where it exists). That is unrelated to the KG change, and going
    through the routers keeps these tests testing the API's own serialisation.
    """

    def test_get_item_returns_an_item(self, kg):
        from lechoes.src.routes.items import get_item_triple_route

        item = asyncio.run(get_item_triple_route(REFERENCE_ID))
        assert item.identifier == REFERENCE_ID
        assert item.case_number == "15"

    def test_get_item_propagates_404(self, kg):
        from lechoes.src.routes.items import get_item_triple_route

        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(get_item_triple_route("does-not-exist"))
        assert excinfo.value.status_code == 404

    def test_category_route_returns_items(self, kg):
        from lechoes.src.routes.items import get_category_items_route

        items = asyncio.run(get_category_items_route("Clothing"))
        assert len(items) == ITEMS_PER_CATEGORY["Clothing"]

    def test_zone_route_returns_items(self, kg):
        from lechoes.src.routes.items import get_zone_items_route

        items = asyncio.run(get_zone_items_route("5"))
        assert len(items) == ITEMS_PER_ZONE["5"]

    def test_search_route_rejects_an_empty_term(self, kg):
        from lechoes.src.routes.items import search_items_route

        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(search_items_route("   "))
        assert excinfo.value.status_code == 400

    def test_categories_route_returns_uri_label_pairs(self, kg):
        from lechoes.src.routes.categories import get_categories_list

        payload = asyncio.run(get_categories_list())
        assert len(payload) == 7
        assert all(set(entry) == {"uri", "label"} for entry in payload)
        assert {entry["label"] for entry in payload} == set(ITEMS_PER_CATEGORY)

    def test_exhibitions_route_keeps_its_response_shape(self, kg):
        """The dashboard reads `name`, `description` and `image` — all real content on
        `core.ttl`'s ExhibitionEvent nodes, unlike the zone stand-in this replaced."""
        from lechoes.src.routes.exhibitions import _get_exhibitions_data

        payload = asyncio.run(_get_exhibitions_data())
        assert len(payload) == 8
        assert all(
            set(entry) == {"uri", "name", "start_date", "end_date", "description", "image"}
            for entry in payload
        )
        assert "Punk & New Romantics" in {entry["name"] for entry in payload}
        assert all(entry["description"] for entry in payload)


# ======================================================================================
# The KG itself — the assumptions every query above rests on
# ======================================================================================

class TestKnowledgeGraphAssumptions:
    """If these change, the queries need revisiting rather than the tests relaxing."""

    def test_the_collection_size(self, kg):
        objects = set(kg.graph.subjects(RDF.type, URIRef("http://www.cidoc-crm.org/cidoc-crm/E22_Human-Made_Object")))
        assert len(objects) == TOTAL_ITEMS

    def test_every_object_has_a_location(self, kg):
        objects = set(kg.graph.subjects(RDF.type, URIRef("http://www.cidoc-crm.org/cidoc-crm/E22_Human-Made_Object")))
        located = {s for s in objects if list(kg.graph.objects(s, URIRef(SCHO + "location")))}
        assert located == objects

    def test_every_case_sits_in_a_zone(self, kg):
        # `scho:location` is reused for a second relation now core.ttl is loaded
        # (ExhibitionEvent -> Zone), so cases are the objects of *items'* scho:location
        # specifically, not every scho:location object in the graph.
        items = set(kg.graph.subjects(RDF.type, URIRef("http://www.cidoc-crm.org/cidoc-crm/E22_Human-Made_Object")))
        cases = {c for item in items for c in kg.graph.objects(item, URIRef(SCHO + "location"))}
        for case in cases:
            assert list(kg.graph.objects(case, URIRef(SCHO + "containedInPlace"))), case

    def test_the_superseded_predicates_are_really_gone(self, kg):
        for predicate in ("itemLocation", "isPartOf"):
            assert not list(kg.graph.subjects(URIRef(SCHO + predicate), None)), predicate
        for rdf_class in ("MusicArtist", "MusicGenre"):
            assert not list(kg.graph.subjects(RDF.type, URIRef(SCHO + rdf_class))), rdf_class

    def test_core_ttl_contributes_eight_real_exhibitions(self, kg):
        """Not superseded — `core.ttl` is the curated layer these ExhibitionEvent nodes
        are meant to come from, replacing the old zone-as-exhibition stand-in."""
        exhibitions = set(kg.graph.subjects(RDF.type, URIRef(SCHO + "ExhibitionEvent")))
        assert len(exhibitions) == 8

    def test_labels_are_english(self, kg):
        languages = {label.language for label in kg.graph.objects(None, RDFS.label)}
        assert languages == {"en"}


# ======================================================================================
# Entity detail (Zone <-> Exhibition cross-linking)
# ======================================================================================

class TestGetEntityDetail:
    def test_zone_detail_lists_its_items_and_hosted_exhibition(self, kg):
        from lechoes.src.queries.entities import get_entity_detail

        detail = get_entity_detail("Zone", "5")
        assert detail["type"] == "Zone"
        assert len(detail["items"]) == ITEMS_PER_ZONE["5"]
        assert [e["label"] for e in detail["exhibitions"]] == ["Punk & New Romantics"]
        assert detail["image"]
        assert detail["description"]

    def test_exhibition_detail_carries_real_content_and_its_zone(self, kg):
        from lechoes.src.queries.entities import get_entity_detail

        detail = get_entity_detail("Exhibition", "PunkNewRomantics")
        assert detail["type"] == "Exhibition"
        assert detail["label"] == "Punk & New Romantics"
        assert detail["image"]
        assert detail["description"]
        assert [z["label"] for z in detail["zones"]] == ["Zone 5"]
        assert len(detail["items"]) == ITEMS_PER_ZONE["5"]

    def test_unknown_zone_has_no_exhibitions(self, kg):
        from lechoes.src.queries.entities import get_entity_detail

        detail = get_entity_detail("Zone", "99")
        assert detail["items"] == []
        assert "exhibitions" not in detail

    def test_unknown_entity_falls_back_to_its_id_as_label(self, kg):
        from lechoes.src.queries.entities import get_entity_detail

        detail = get_entity_detail("Person", "does-not-exist")
        assert detail["label"] == "does-not-exist"
        assert detail["items"] == []


# ======================================================================================
# Per-entity link counts
# ======================================================================================

class TestEntityLinkCounts:
    """`LinkedEntity.count` — how many items link to each entity.

    Asserted against the totals the zone and category tests already pin, counted
    independently of the aggregate query under test.
    """

    def test_counts_cover_the_whole_graph_in_one_pass(self, kg):
        counts = items_queries.entity_item_counts()
        assert counts, "the aggregate returned nothing at all"
        assert counts[f"{BME}/Zone/5"] == ITEMS_PER_ZONE["5"]
        assert counts[f"{BME}/Exhibition/PunkNewRomantics"] == ITEMS_PER_ZONE["5"]

    def test_a_category_uri_counts_its_category_listing(self, kg):
        counts = items_queries.entity_item_counts()
        # Q34379 is the Musical Instrument category node.
        assert counts["http://www.wikidata.org/entity/Q34379"] == ITEMS_PER_CATEGORY["Musical Instrument"]

    def test_an_items_linked_entities_carry_their_counts(self, kg):
        item = items_queries.get_item_by_identifier(REFERENCE_ID)
        assert item.location[0].count == ITEMS_PER_ZONE["5"]
        assert item.exhibition[0].count == ITEMS_PER_ZONE["5"]
        assert item.category.count == ITEMS_PER_CATEGORY["Clothing"]

    def test_a_literal_valued_link_has_no_count(self, kg):
        """Colour is a plain literal with no entity behind it, so there is nothing to
        count — and a zero would read as "nothing else is this colour"."""
        item = items_queries.get_item_by_identifier(REFERENCE_ID)
        assert item.colour[0].count is None

    def test_an_entity_detail_counts_its_own_cross_links(self, kg):
        from lechoes.src.queries.entities import get_entity_detail

        detail = get_entity_detail("Zone", "5")
        assert detail["exhibitions"][0]["count"] == ITEMS_PER_ZONE["5"]

    def test_a_write_invalidates_the_cache(self, kg, session):
        """Invalidation lives in the query layer, so this direct call is covered too."""
        before = items_queries.entity_item_counts()[f"{BME}/Zone/5"]
        items_queries.insert(_new_item(), session)
        assert items_queries.entity_item_counts()[f"{BME}/Zone/5"] == before + 1
