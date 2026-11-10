"""
Entity-type check queries ("ECKs") used to validate candidate entities returned
by the entity-lookup service before an LLM is asked to disambiguate them.

Each ECK is a ``{"wd": <ask query>, "dbp": <ask query>}`` dict whose queries
carry a Jinja ``{{entity_id}}`` placeholder, rendered by
``linker.validate_entity``.

Why the Wikidata queries are shaped the way they are
----------------------------------------------------
The queries are generated from the declarative ``WD_TYPE_SPECS`` below rather
than hand-written, so the accepted class sets cannot drift out of sync with the
SPARQL that tests them. Four empirical constraints drove the shape:

1. **The entity IRI is inlined, never bound with ``BIND``.** WDQS (Blazegraph)
   evaluates a transitive path fastest when *both* endpoints are constants. A
   ``BIND(wd:Q123 AS ?entity)`` in front of ``?entity wdt:P31/wdt:P279* ?x``
   loses that plan.

2. **No unbound intermediate node inside a transitive path.** Splitting
   ``wdt:P31/wdt:P279*`` into ``wdt:P31 ?c . ?c wdt:P279* wd:ROOT`` does *not*
   help and is often far worse (measured: 35-38s vs 0.4s).

3. **Only ``wdt:`` (truthy) statements are visible.** A class that shows up in
   an entity's claims via the MediaWiki API may be invisible to ``wdt:`` if the
   statement is deprecated or not best-rank. Woodstock lists "music festival"
   among its P31 claims but its only *truthy* P31 is "festival occurrence"; the
   colour green likewise has a P279 to "color" that ``wdt:P279`` does not see.
   Roots must therefore be chosen from the truthy graph, not from an API dump.

4. **``wdt:P31/wdt:P279*`` is only affordable when the root's subclass tree is
   small.** It is fine for venue/colour/album/settlement roots, but times out
   against ``wd:Q214609`` (material), whose descendant tree is enormous. Types
   whose members are *classes* rather than instances (material, colour) are
   therefore matched with a plain ``wdt:P279*`` plus a cheap direct ``wdt:P31``
   branch.

Class IDs were verified against live Wikidata; the previous revision of this
file referenced several wrong QIDs (e.g. Q133998 "single" is actually a Belgian
village, Q730221 "EP" is Bourges Airport, Q2002764 does not exist).

The DBpedia queries are kept for API compatibility and have had their invalid
SPARQL corrected (the previous versions used bare ``UNION`` between triple
patterns, which does not parse), but they are **not** covered by the test suite
in ``test_linker.py`` and should be treated as unvalidated.
"""

from __future__ import annotations

# ##############################################################################
# Declarative type specifications
# ##############################################################################

# Each spec accepts an entity when ANY branch matches:
#   instance_of : entity is an instance of the root or of any subclass of it
#                 (wdt:P31/wdt:P279*)  -- for types whose members are instances
#   subclass_of : entity IS the root or any subclass of it
#                 (wdt:P279*)          -- for types whose members are classes
#   instance_of_exact : entity is a direct instance of the root
#                 (wdt:P31)            -- cheap catch for class-like types whose
#                                         members are occasionally modelled as
#                                         instances (e.g. paper -> material)
#   occupation  : entity has an occupation that is the root or a subclass of it
#                 (wdt:P106/wdt:P279*) -- humans only
#
# Comments record the human-readable label of every QID so wrong IDs are visible
# on review rather than only at query time.

WD_TYPE_SPECS = {
    "music_artist": {
        # Q2088357 (musical ensemble) is the correct group root. Q215380
        # ("musical group") is the tempting choice but neither Q42998
        # (orchestra) nor Q239582 (symphony orchestra) is a subclass of it,
        # so the London Symphony Orchestra would not validate; musical group,
        # musical duo and symphony orchestra all sit under Q2088357.
        "instance_of": [("Q2088357", "musical ensemble")],
        # Every music occupation checked resolves under Q639669: singer,
        # composer, conductor, guitarist, drummer, pianist, rapper, disc jockey,
        # record producer, songwriter, singer-songwriter, music arranger.
        "occupation": [("Q639669", "musician")],
    },
    "place": {
        # Settlements alone are not enough: Italian comuni, English metropolitan
        # counties and German federated states are administrative entities and
        # are NOT subclasses of human settlement.
        # Countries are reachable here too (Q6256 is under Q56061); `place` is
        # the generic geographic slot, `country` is the strict one.
        "instance_of": [
            ("Q486972", "human settlement"),
            ("Q56061", "administrative territorial entity"),
        ],
    },
    "country": {
        "instance_of": [
            ("Q6256", "country"),
            ("Q3624078", "sovereign state"),
            ("Q3024240", "historical country"),
            ("Q10711424", "state with limited recognition"),
        ],
    },
    "genre": {
        "instance_of": [("Q188451", "music genre")],
    },
    "music_work": {
        # The abstract work, as opposed to a release (album/single/EP) or a
        # recording (Q55850593 "music track with vocals").
        #
        # Exactly one root, deliberately. Q2188189 ("musical work") looks like
        # the obvious choice but is defined as a *generic* term covering songs
        # AND albums AND recordings: with it in the set, every release and
        # recording validated as a work. Q7366 ("song") and Q207628 ("composed
        # musical work") were also dropped -- they accept none of the works in
        # the fixture, all of which are P31 Q105543609, so they add no recall.
        "instance_of": [("Q105543609", "musical work/composition")],
    },
    "music_album": {
        # Releases. Album/single/EP/LP are independent roots -- notably single,
        # EP and album are NOT subclasses of Q2031291 (musical release).
        "instance_of": [
            ("Q482994", "album"),
            ("Q134556", "single"),
            ("Q169930", "extended play"),
            ("Q841983", "LP record"),
            ("Q2031291", "musical release"),
        ],
    },
    "event": {
        # Q1656682 (planned event) subsumes every root below and is the
        # semantically obvious choice, but its descendant tree covers every
        # sports fixture and election in Wikidata and it is 10-20x slower on the
        # common case -- a candidate that must be *rejected* has to walk the
        # whole tree (measured 4.8-28s per negative, vs 0.4-1.4s for the set
        # below). The domain roots are used instead, at the cost of generality:
        # an event that is neither a concert, festival nor tour (a recording
        # session, a TV appearance) will not validate.
        # Q868557 and Q41582469 are listed separately because, despite the
        # names, neither is a subclass of Q132241 (festival).
        "instance_of": [
            ("Q182832", "concert"),
            ("Q868557", "music festival"),
            ("Q41582469", "music festival edition"),
            ("Q27968043", "festival occurrence"),
            ("Q132241", "festival"),
            ("Q1573906", "concert tour"),
        ],
    },
    "venue": {
        # Recording studios are deliberately included (Abbey Road Studios is a
        # first-class entity in this domain) but are not under Q17350442.
        # Q41176 (building) is deliberately NOT a root: it would admit any
        # structure at all.
        "instance_of": [
            ("Q17350442", "venue"),
            ("Q746369", "recording studio"),
        ],
    },
    "colour": {
        # Colours are modelled inconsistently as instances -- white is P31
        # color, black/red/blue/green/yellow are P31 primary color, orange is
        # P31 secondary color, gold is P31 web color -- but every one of those
        # classes is a subclass of color, so this single branch covers them all.
        # A bare wdt:P279* branch was tried and removed: it is redundant (it
        # changed no verdict) and it was by far the most expensive pattern in
        # the whole module, pushing batched colour validation past the 60s WDQS
        # limit at 13 candidates.
        "instance_of": [("Q1075", "color")],
    },
    "material": {
        # Materials are overwhelmingly classes (leather, velvet, denim, nacre
        # and snakeskin have no P31 at all), so the P279* branch does the work.
        # P31/P279* is NOT used here: Q214609's descendant tree is large enough
        # to time out WDQS. The exact-P31 branch is the cheap catch for the few
        # materials modelled as instances (e.g. paper P31 material).
        "subclass_of": [("Q214609", "material")],
        "instance_of_exact": [("Q214609", "material")],
    },
}


# ##############################################################################
# Query generation
# ##############################################################################

_ENTITY = "wd:{{entity_id}}"  # inlined, deliberately not BIND-ed (see module doc)

_BRANCH_PATTERNS = {
    "instance_of": "{e} wdt:P31/wdt:P279* wd:{qid} .",
    "subclass_of": "{e} wdt:P279* wd:{qid} .",
    "instance_of_exact": "{e} wdt:P31 wd:{qid} .",
    "occupation": "{e} wdt:P106/wdt:P279* wd:{qid} .",
}

# Order branches cheapest-first so a match short-circuits before the costly ones.
_BRANCH_ORDER = ("instance_of_exact", "instance_of", "occupation", "subclass_of")


def build_wd_ask(spec: dict) -> str:
    """Render a Wikidata ASK query from a type spec (see ``WD_TYPE_SPECS``)."""
    branches = []
    for kind in _BRANCH_ORDER:
        for qid, label in spec.get(kind, []):
            pattern = _BRANCH_PATTERNS[kind].format(e=_ENTITY, qid=qid)
            branches.append(f"    {{ {pattern} }}  # {label}")
    if not branches:
        raise ValueError("type spec has no branches")
    body = "\n    UNION\n".join(branches)
    return "ASK {\n" + body + "\n}\n"


# ##############################################################################
# DBpedia queries (unvalidated -- kept for API compatibility)
# ##############################################################################

_DBP = {
    "music_artist": """
        ASK {
            { dbr:{{entity_id}} a dbo:Band. }
            UNION
            { dbr:{{entity_id}} a dbo:MusicalArtist. }
        }
        """,
    "place": """
        ASK {
            { dbr:{{entity_id}} a dbo:Settlement. }
            UNION
            { dbr:{{entity_id}} a dbo:PopulatedPlace. }
            UNION
            { dbr:{{entity_id}} a dbo:City. }
            UNION
            { dbr:{{entity_id}} a dbo:Town. }
            UNION
            { dbr:{{entity_id}} a dbo:Village. }
            UNION
            { dbr:{{entity_id}} a dbo:AdministrativeRegion. }
        }
        """,
    "country": """
        ASK { dbr:{{entity_id}} a dbo:Country. }
        """,
    "genre": """
        ASK { dbr:{{entity_id}} a dbo:Genre. }
        """,
    "music_work": """
        ASK {
            { dbr:{{entity_id}} a dbo:MusicalWork. }
            UNION
            { dbr:{{entity_id}} a schema:MusicComposition. }
        }
        """,
    "music_album": """
        ASK {
            { dbr:{{entity_id}} a dbo:Album. }
            UNION
            { dbr:{{entity_id}} a dbo:Single. }
            UNION
            { dbr:{{entity_id}} a dbo:ExtendedPlay. }
        }
        """,
    "event": """
        ASK { dbr:{{entity_id}} rdf:type/rdfs:subClassOf* dbo:Event. }
        """,
    "venue": """
        ASK {
            { dbr:{{entity_id}} rdf:type/rdfs:subClassOf* dbo:Venue. }
            UNION
            { dbr:{{entity_id}} rdf:type/rdfs:subClassOf* dbo:Building. }
        }
        """,
    "colour": """
        ASK { dbr:{{entity_id}} a dbo:Colour. }
        """,
    "material": """
        ASK { dbr:{{entity_id}} rdf:type/rdfs:subClassOf* dbo:Material. }
        """,
}


def _eck(type_name: str) -> dict:
    return {
        "wd": build_wd_ask(WD_TYPE_SPECS[type_name]),
        "dbp": _DBP[type_name],
    }


# ##############################################################################
# Public ECKs
# ##############################################################################

MUSIC_ARTIST_ECK = _eck("music_artist")
LOCATION_ECK = _eck("place")
COUNTRY_ECK = _eck("country")
MUSIC_GENRE_ECK = _eck("genre")
MUSIC_WORK_ECK = _eck("music_work")
MUSIC_ALBUM_ECK = _eck("music_album")
GENERAL_EVENT_ECK = _eck("event")
VENUE_ECK = _eck("venue")
COLOUR_ECK = _eck("colour")
MATERIAL_ECK = _eck("material")

#: Maps the entity-type names used by the ingest pipeline (the ``fields_to_link``
#: in ``populate.py``) to their ECK, so a linker can look the check up from the
#: ``entity_type`` argument alone.
ECK_BY_TYPE = {
    "music_artist": MUSIC_ARTIST_ECK,
    "place": LOCATION_ECK,
    "country": COUNTRY_ECK,
    "genre": MUSIC_GENRE_ECK,
    "music_work": MUSIC_WORK_ECK,
    "music_album": MUSIC_ALBUM_ECK,
    "event": GENERAL_EVENT_ECK,
    "venue": VENUE_ECK,
    "colour": COLOUR_ECK,
    "material": MATERIAL_ECK,
}

#: Aliases for entity-type spellings that callers may pass instead of the
#: canonical pipeline field names.
ECK_ALIASES = {
    "artist": "music_artist",
    "musicartist": "music_artist",
    "music artist": "music_artist",
    "band": "music_artist",
    "musician": "music_artist",
    "location": "place",
    "city": "place",
    "music_genre": "genre",
    "music genre": "genre",
    "musicgenre": "genre",
    "work": "music_work",
    "song": "music_work",
    "music work": "music_work",
    "composition": "music_work",
    "album": "music_album",
    "music album": "music_album",
    "release": "music_album",
    "color": "colour",
}


def normalise_type(entity_type: str) -> str | None:
    """Map a free-form entity-type string onto a canonical ECK type name."""
    if not entity_type:
        return None
    key = entity_type.strip().lower().replace("-", "_")
    if key in ECK_BY_TYPE:
        return key
    if key in ECK_ALIASES:
        return ECK_ALIASES[key]
    spaced = key.replace("_", " ")
    return ECK_ALIASES.get(spaced)


def get_eck(entity_type: str) -> dict | None:
    """Return the ECK for an entity type, or None if the type is unknown."""
    canonical = normalise_type(entity_type)
    return ECK_BY_TYPE.get(canonical) if canonical else None
