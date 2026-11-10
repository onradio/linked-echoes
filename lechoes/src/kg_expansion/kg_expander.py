"""
kg_expander.py

Iterative Wikidata enrichment driven by type_mapping.json + entity_type_overrides.csv.
Processes one entity at a time. Never revisits a QID. Writes a report CSV at the end.

Usage:
    # Build frontier first (if not already done):
    python kg_expansion/build_frontier.py \
        --kg ../../assets/kg_triple_store/knowledge_graph.ttl \
        --output kg_expansion/frontier.json

    # Test run — 10 entities, Human + MusicGroup only:
    python kg_expansion/kg_expander.py \
        --frontier kg_expansion/frontier.json \
        --output-dir kg_expansion/ttl_expanded \
        --type-filter Human,MusicGroup \
        --max-entities 10 \
        --dry-run

    # Full run:
    python kg_expansion/kg_expander.py \
        --frontier kg_expansion/frontier.json \
        --output-dir kg_expansion/ttl_expanded
"""

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from wd_classifier import classify, Tier2Tracker, tier2_triples, TIER1
from shacl_validator import run_shacl, SHAPES_DEFAULT

warnings.filterwarnings("ignore")

SCHO = Namespace("http://schema.org/")
WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
MIMO = Namespace("http://www.mimo-db.eu/InstrumentsKeywords/")
WD_API = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

# ---------------------------------------------------------------------------
# Property map per normalized KG type → {WD PID: (schema.org predicate, value_type)}
# value_type: "uri_label" | "date" | "literal"
# ---------------------------------------------------------------------------
# Value types:
#   "literal"      — plain text / URL / number
#   "date"         — date value
#   "uri_expand"   — write triple + add object QID to frontier (round 0 AND round 1+)
#   "uri_label"    — write triple + add object QID to frontier (round 0 only)
#   "uri_nonstub"  — write triple, NEVER add object QID to frontier
#   "mimo"         — two-hop: WD instrument QID → P3763 → MIMO URI; fallback to WD URI
# ---------------------------------------------------------------------------
PROPERTY_MAP = {
    "Person": {
        "P1476": (SCHO.name,            "literal"),
        "P1477": (SCHO.alternateName,   "literal"),     # birth name
        "P569":  (SCHO.birthDate,       "date"),
        "P570":  (SCHO.deathDate,       "date"),
        "P19":   (SCHO.birthPlace,      "uri_nonstub"), # → City/Place terminal
        "P20":   (SCHO.deathPlace,      "uri_nonstub"), # → City/Place terminal
        "P27":   (SCHO.nationality,     "uri_nonstub"), # → Country terminal
        "P21":   (SCHO.gender,          "uri_nonstub"), # → DefinedTerm terminal
        "P106":  (SCHO.hasOccupation,   "uri_nonstub"), # → DefinedTerm terminal — main explosion source
        "P1303": (SCHO.instrument,      "mimo"),        # → MIMO URI via P3763 two-hop
        "P412":  (SCHO.knowsAbout,      "uri_nonstub"), # voice type → DefinedTerm terminal
        "P136":  (SCHO.genre,           "uri_nonstub"), # → MusicGenre terminal
        "P737":  (SCHO.isRelatedTo,     "uri_label"),   # influenced by → Person/MusicGroup (expand round 0)
        "P1066": (SCHO.knows,           "uri_label"),   # student of / mentor → Person (expand round 0)
        "P463":  (SCHO.memberOf,        "uri_label"),   # → MusicGroup (expand round 0)
        "P264":  (SCHO.recordLabel,     "uri_nonstub"), # → Organisation terminal
        "P166":  (SCHO.award,           "uri_nonstub"), # → Award terminal
        "P1411": (SCHO.nominatedFor,    "uri_nonstub"), # → Award terminal
        # P800 notable work — separate catalogue_expansion pass
        "P1344": (SCHO.performerIn,     "uri_nonstub"), # → Event (don't expand)
        "P434":  (SCHO.sameAs,          "uri_nonstub"), # external ID
        "P1953": (SCHO.sameAs,          "uri_nonstub"), # external ID
        "P213":  (SCHO.sameAs,          "literal"),
        "P856":  (SCHO.url,             "literal"),
        "P18":   (SCHO.image,           "literal"),
    },
    "MusicGroup": {
        "P1476": (SCHO.name,            "literal"),
        "P571":  (SCHO.foundingDate,    "date"),
        "P576":  (SCHO.dissolutionDate, "date"),
        "P740":  (SCHO.foundingLocation,"uri_nonstub"), # → Place terminal
        "P136":  (SCHO.genre,           "uri_nonstub"), # → MusicGenre terminal
        "P737":  (SCHO.isRelatedTo,     "uri_label"),   # influenced by → Person/MusicGroup (expand round 0)
        "P527":  (SCHO.member,          "uri_label"),   # → Person (expand round 0)
        "P264":  (SCHO.recordLabel,     "uri_nonstub"), # → Organisation terminal
        "P495":  (SCHO.countryOfOrigin, "uri_nonstub"), # → Country terminal
        "P166":  (SCHO.award,           "uri_nonstub"), # → Award terminal
        "P1411": (SCHO.nominatedFor,    "uri_nonstub"), # → Award terminal
        # P800 notable work — catalogue_expansion pass
        "P1344": (SCHO.performerIn,     "uri_nonstub"), # → Event (don't expand)
        "P434":  (SCHO.sameAs,          "uri_nonstub"), # external ID
        "P1953": (SCHO.sameAs,          "uri_nonstub"), # external ID
        "P213":  (SCHO.sameAs,          "literal"),
        "P856":  (SCHO.url,             "literal"),
        "P18":   (SCHO.image,           "literal"),
    },
    "MusicAlbum": {
        "P1476": (SCHO.name,            "literal"),
        "P577":  (SCHO.datePublished,   "date"),
        "P175":  (SCHO.byArtist,        "uri_label"),   # → Person/MusicGroup (expand round 0)
        "P86":   (SCHO.composer,        "uri_nonstub"), # → Person (don't re-expand from album)
        "P162":  (SCHO.producer,        "uri_nonstub"), # → Person (don't re-expand from album)
        "P264":  (SCHO.recordLabel,     "uri_nonstub"), # → Organisation terminal
        "P2635": (SCHO.numTracks,       "literal"),
        "P2047": (SCHO.duration,        "literal"),
        "P483":  (SCHO.recordedAt,      "uri_nonstub"), # → Venue (don't expand from album)
        "P658":  (SCHO.track,           "uri_label"),   # → MusicRecording stubs (expand round 0)
        "P136":  (SCHO.genre,           "uri_nonstub"), # → MusicGenre terminal
        "P407":  (SCHO.inLanguage,      "uri_nonstub"), # → DefinedTerm terminal
        "P495":  (SCHO.countryOfOrigin, "uri_nonstub"), # → Country terminal
        "P18":   (SCHO.image,           "literal"),
        "P436":  (SCHO.sameAs,          "uri_nonstub"), # external ID
        "P435":  (SCHO.sameAs,          "uri_nonstub"), # external ID
        "P1953": (SCHO.sameAs,          "uri_nonstub"), # external ID
    },
    "MusicComposition": {
        # Branch A — musical work
        "P86":   (SCHO.composer,        "uri_nonstub"), # → Person (don't re-expand from recording)
        "P676":  (SCHO.lyricist,        "uri_nonstub"), # → Person (don't re-expand from recording)
        "P407":  (SCHO.inLanguage,      "uri_nonstub"), # → DefinedTerm terminal
        "P571":  (SCHO.dateCreated,     "date"),
        # Branch B — audio recording
        "P175":  (SCHO.byArtist,        "uri_label"),   # → Person/MusicGroup (expand round 0)
        "P162":  (SCHO.producer,        "uri_nonstub"), # → Person (don't re-expand)
        "P2047": (SCHO.duration,        "literal"),
        "P577":  (SCHO.datePublished,   "date"),
        "P361":  (SCHO.inAlbum,         "uri_label"),   # → MusicAlbum (expand round 0)
        "P264":  (SCHO.recordLabel,     "uri_nonstub"), # → Organisation terminal
        # Common
        "P1476": (SCHO.name,            "literal"),
        "P136":  (SCHO.genre,           "uri_nonstub"), # → MusicGenre terminal
        "P435":  (SCHO.sameAs,          "uri_nonstub"), # external ID
        "P1243": (SCHO.sameAs,          "literal"),
    },
    "ConcertTour": {
        "P1476": (SCHO.name,            "literal"),
        "P580":  (SCHO.startDate,       "date"),
        "P582":  (SCHO.endDate,         "date"),
        "P175":  (SCHO.performer,       "uri_label"),   # → Person/MusicGroup (expand round 0)
        "P276":  (SCHO.location,        "uri_nonstub"), # → Venue/City (don't expand from tour)
        "P527":  (SCHO.subEvent,        "uri_nonstub"), # → Event stubs (don't expand)
        "P144":  (SCHO.isBasedOn,       "uri_label"),   # → MusicAlbum (expand round 0)
    },
    # Concert collapsed into Event
    "Event": {
        "P1476": (SCHO.name,            "literal"),
        "P580":  (SCHO.startDate,       "date"),
        "P582":  (SCHO.endDate,         "date"),
        "P585":  (SCHO.startDate,       "date"),        # point in time fallback
        "P571":  (SCHO.startDate,       "date"),        # inception fallback
        "P276":  (SCHO.location,        "uri_nonstub"), # → Venue (don't expand from event)
        "P175":  (SCHO.performer,       "uri_label"),   # → Person/MusicGroup (expand round 0)
        "P17":   (SCHO.location,        "uri_nonstub"), # country fallback
        "P361":  (SCHO.superEvent,      "uri_nonstub"), # → ConcertTour (don't expand)
    },
    # Venue covers NightClub and Theatre
    "Venue": {
        "P1476": (SCHO.name,                    "literal"),
        "P969":  (SCHO.address,                 "literal"),
        "P17":   (SCHO.addressCountry,          "uri_nonstub"), # → Country terminal
        "P625":  (SCHO.geo,                     "literal"),
        "P1619": (SCHO.foundingDate,            "date"),
        "P3999": (SCHO.dissolutionDate,         "date"),
        "P571":  (SCHO.foundingDate,            "date"),        # inception fallback
        "P576":  (SCHO.dissolutionDate,         "date"),        # dissolved fallback
        "P1083": (SCHO.maximumAttendeeCapacity, "literal"),
        "P856":  (SCHO.url,                     "literal"),
        "P18":   (SCHO.image,                   "literal"),
        "P1566": (SCHO.sameAs,                  "literal"),     # GeoNames ID
    },
}

# Wikidata classifies many songs as MusicRecording — alias to MusicComposition
PROPERTY_MAP["MusicRecording"] = PROPERTY_MAP["MusicComposition"]

# Types that should expand outward (add new stubs to frontier)
# Concert → Event, NightClub/Theatre → Venue (collapsed per parsing strategy)
EXPAND_TYPES = {"Person", "MusicGroup", "MusicAlbum", "MusicComposition",
                "ConcertTour", "Event", "Venue"}

# Types that are terminal — label+description only, never add stubs to frontier
TERMINAL_TYPES = {"Country", "City", "Place", "DefinedTerm", "Organisation",
                  "Award", "Film", "TVShow", "Island", "FederalState", "District",
                  "USState", "Gulf", "Restaurant", "ArtMuseum", "MusicGenre",
                  "Material", "NightClub", "Theatre", "Concert"}

# Schema.org type URIs
TYPE_URI_MAP = {t: getattr(SCHO, t) for t in list(EXPAND_TYPES) + list(TERMINAL_TYPES)}
# Wikidata may classify entities as MusicRecording — treat as MusicComposition throughout
TYPE_URI_MAP["MusicRecording"] = SCHO.MusicComposition


def load_type_mapping(path: Path) -> dict:
    return json.loads(path.read_text())


def load_overrides(path: Path) -> dict:
    """Returns {entity_qid: normalized_type}"""
    overrides = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row.get("Entity QID", "").strip()
            ntype = row.get("Assigned KG type", "").strip()
            if qid and ntype:
                overrides[qid] = ntype
    return overrides


def fetch_entity(qid: str, retries: int = 3) -> dict | None:
    url = WD_API.format(qid=qid)
    req = urllib.request.Request(url, headers={"User-Agent": "BME-KG-Expander/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            return data["entities"].get(qid)
        except Exception as e:
            wait = [5, 15, 30][attempt]
            print(f"  [warn] fetch {qid} attempt {attempt+1} failed: {e} — retry in {wait}s")
            time.sleep(wait)
    return None


def resolve_type(qid: str, entry: dict, type_mapping: dict, overrides: dict) -> tuple[str, str]:
    """
    Returns (normalized_type, role).
    Resolution order: entity override → class mapping (highest confidence) → Unknown.
    """
    if qid in overrides:
        return overrides[qid], "enrichment_target"

    # Collect P31 QIDs from frontier entry types list
    # types in frontier are schema.org labels from KG, not P31 QIDs —
    # so we resolve via type_mapping using the P31 claims fetched from WD.
    # For pre-dispatch (before fetch), fall back to KG-side type label.
    kg_types = entry.get("types", [])

    # Try direct match in type_mapping by KG label (approximate)
    for tm_entry in type_mapping.values():
        if tm_entry["normalized_type"] in kg_types:
            return tm_entry["normalized_type"], tm_entry["role"]

    # Default: use first KG type if known
    if kg_types:
        t = kg_types[0]
        if t in EXPAND_TYPES:
            return t, "enrichment_target"
        if t in TERMINAL_TYPES:
            return t, "terminal_context"

    return "Unknown", "review_needed"


def resolve_type_from_p31(p31_qids: set, qid: str, type_mapping: dict, overrides: dict) -> tuple[str, str]:
    """Resolve using actual P31 QIDs fetched from Wikidata."""
    if qid in overrides:
        return overrides[qid], "enrichment_target"

    # Pick highest-confidence match
    best_type, best_role, best_conf = None, None, 0
    conf_rank = {"high": 3, "medium": 2, "low": 1}
    for p31_qid in p31_qids:
        if p31_qid in type_mapping:
            entry = type_mapping[p31_qid]
            rank = conf_rank.get(entry["confidence"], 0)
            if rank > best_conf:
                best_type = entry["normalized_type"]
                best_role = entry["role"]
                best_conf = rank

    if best_type:
        return best_type, best_role
    return "Unknown", "review_needed"


LABEL_LANGS = ("en", "mul", "en-gb", "en-ca")


def pick_label(entity: dict, fallback: str = "") -> tuple[str, str]:
    """
    Best English-usable label for a Wikidata entity, with the source that
    supplied it. Wikidata migrates language-independent names (song and album
    titles especially) to the "mul" code, so an en-only read misses them.
    Returns ("", "") when nothing safe to assert as @en is available.
    """
    labels = entity.get("labels", {})
    for lang in LABEL_LANGS:
        value = labels.get(lang, {}).get("value")
        if value:
            return value, f"labels.{lang}"

    for claim in entity.get("claims", {}).get("P1476", []):
        try:
            value = claim["mainsnak"]["datavalue"]["value"]
            if value.get("text") and value.get("language", "en").startswith("en"):
                return value["text"], "P1476"
        except (KeyError, TypeError, AttributeError):
            pass

    sitelink = entity.get("sitelinks", {}).get("enwiki", {}).get("title")
    if sitelink:
        return sitelink, "sitelink.enwiki"

    for alias in entity.get("aliases", {}).get("en", []):
        if alias.get("value"):
            return alias["value"], "aliases.en"

    return (fallback, "frontier") if fallback else ("", "")


def get_p31_qids(entity: dict) -> set:
    qids = set()
    for claim in entity.get("claims", {}).get("P31", []):
        try:
            qids.add(claim["mainsnak"]["datavalue"]["value"]["id"])
        except (KeyError, TypeError):
            pass
    return qids


def get_claim_values(entity: dict, prop: str) -> list:
    results = []
    for claim in entity.get("claims", {}).get(prop, []):
        try:
            snak = claim["mainsnak"]
            dtype = snak.get("datatype")
            val = snak.get("datavalue", {}).get("value")
            if val is None:
                continue
            if dtype == "wikibase-item":
                qid = val["id"]
                results.append(("uri", f"http://www.wikidata.org/entity/{qid}", qid))
            elif dtype == "time":
                t = val.get("time", "")
                if t.startswith("+"):
                    t = t[1:]
                # WD dates can have out-of-range day values (e.g. 1969-02-30)
                # or years shorter than 4 digits (e.g. ancient dates like "850").
                # Use full date if valid, year-only if day invalid, skip if year < 4 digits.
                date_str = t[:10]
                year_str = t[:4]
                if len(year_str) < 4 or not year_str.lstrip("-").isdigit():
                    continue  # skip unparseable ancient dates
                try:
                    from datetime import date as _date
                    parts = date_str.split("-")
                    _date(int(parts[0]), int(parts[1]), int(parts[2]))
                    results.append(("date", date_str, ""))
                except (ValueError, IndexError):
                    results.append(("date", year_str, ""))   # year only
            elif dtype == "quantity":
                results.append(("literal", str(val.get("amount", "").lstrip("+")), ""))
            elif dtype == "string":
                results.append(("literal", val, ""))
            elif dtype == "url":
                results.append(("literal", val, ""))
            elif dtype == "globe-coordinate":
                lat = val.get("latitude", "")
                lon = val.get("longitude", "")
                results.append(("literal", f"{lat},{lon}", ""))
        except (KeyError, TypeError):
            pass
    return results


def resolve_mimo_uri(instrument_qid: str) -> str | None:
    """
    Given a Wikidata instrument QID, fetch P3763 (MIMO instrument ID) and
    return the MIMO URI, or None if not available.
    """
    entity = fetch_entity(instrument_qid)
    if entity is None:
        return None
    for claim in entity.get("claims", {}).get("P3763", []):
        try:
            val = claim["mainsnak"]["datavalue"]["value"]
            if isinstance(val, str) and val:
                return str(MIMO[val])
        except (KeyError, TypeError):
            pass
    return None


def enrich_entity(
    qid: str,
    entry: dict,
    output_dir: Path,
    type_mapping: dict,
    overrides: dict,
    seen: set,
    dry_run: bool = False,
    expand_rounds: int = 1,
) -> dict:
    """
    Enrich one entity. Returns a report row dict.
    `seen` is the global set of QIDs already processed or queued — prevents loops.
    """
    uri_str = entry.get("uri", f"http://www.wikidata.org/entity/{qid}")
    uri = URIRef(uri_str)

    report = {
        "qid": qid,
        "label": entry.get("label", ""),
        "kg_types": "|".join(entry.get("types", [])),
        "resolved_type": "",
        "role": "",
        "status": "pending",
        "triples_written": 0,
        "new_stubs": 0,
        "new_stub_qids": "",
        "error": "",
        "enriched_at": "",
    }

    entity = fetch_entity(qid)
    if entity is None:
        report["status"] = "fetch_failed"
        report["error"] = "Wikidata fetch returned None"
        return report

    p31_qids = list(get_p31_qids(entity))
    wd_label, _label_src = pick_label(entity, fallback=entry.get("label", ""))
    report["label"] = wd_label

    # Tier-1 / Tier-2 classification via SPARQL hierarchy walk
    cls = classify(entity_qid=qid, p31_qids=p31_qids, overrides=overrides,
                   type_mapping=type_mapping)
    norm_type = cls.normalized_type
    role = cls.role

    report["resolved_type"] = norm_type
    report["role"] = role

    # Tier 2 — write minimal terminal_reference TTL (label + desc + sameAs + P31 refs)
    if cls.tier == 2:
        if not dry_run:
            g2 = tier2_triples(qid, entity, p31_qids)
            out_path = output_dir / f"{qid}.ttl"
            out_path.write_text(g2.serialize(format="turtle"), encoding="utf-8")
        report["status"] = "terminal_reference"
        report["triples_written"] = 3 + len(p31_qids)  # approx
        return report

    if role == "review_needed":
        report["status"] = "review_needed"
        return report

    if norm_type not in PROPERTY_MAP:
        report["status"] = "no_property_map"
        report["error"] = f"No PROPERTY_MAP entry for type '{norm_type}'"
        return report

    # Build graph
    g = Graph()
    g.bind("schema", SCHO)
    g.bind("wd", WD)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # Use EventVenue if that's the KG-side type, otherwise look up TYPE_URI_MAP
    kg_type = (entry.get("types") or [None])[0]
    if kg_type == "EventVenue":
        g.add((uri, RDF.type, SCHO.EventVenue))
    elif norm_type in TYPE_URI_MAP:
        g.add((uri, RDF.type, TYPE_URI_MAP[norm_type]))

    label_val = report["label"]
    if label_val:
        g.add((uri, RDFS.label, Literal(label_val, lang="en")))

    desc = entity.get("descriptions", {}).get("en", {}).get("value", "")
    if desc:
        g.add((uri, SCHO.description, Literal(desc, lang="en")))

    prop_map = PROPERTY_MAP[norm_type]
    new_stub_qids = []

    g.bind("mimo", MIMO)

    for prop_id, (predicate, val_type) in prop_map.items():
        for vtype, value, stub_qid in get_claim_values(entity, prop_id):
            if val_type == "mimo" and vtype == "uri":
                # Two-hop: WD instrument QID → P3763 → MIMO URI
                mimo_uri = resolve_mimo_uri(stub_qid)
                if mimo_uri:
                    g.add((uri, predicate, URIRef(mimo_uri)))
                else:
                    # Fallback: assert WD URI if no MIMO ID available
                    g.add((uri, predicate, URIRef(value)))
            elif vtype == "uri":
                obj = URIRef(value)
                g.add((uri, predicate, obj))
                # Stub policy:
                #   uri_nonstub  — write triple, never stub
                #   uri_label    — stub only for round 0 (direct museum entities)
                #   uri_expand   — always stub
                if stub_qid and stub_qid not in seen:
                    entity_round = entry.get("round", 0)
                    if val_type == "uri_expand":
                        new_stub_qids.append(stub_qid)
                    elif val_type == "uri_label" and entity_round < expand_rounds:
                        new_stub_qids.append(stub_qid)
                    # uri_nonstub: triple written above, no stub
            elif vtype == "date":
                dtype = XSD.date if len(value) == 10 else XSD.gYear
                g.add((uri, predicate, Literal(value, datatype=dtype)))
            elif vtype == "literal":
                g.add((uri, predicate, Literal(value)))

    triple_count = len(g)

    if not dry_run:
        out_path = output_dir / f"{qid}.ttl"
        out_path.write_text(g.serialize(format="turtle"), encoding="utf-8")

    report["status"] = "enriched"
    report["triples_written"] = triple_count
    report["new_stubs"] = len(new_stub_qids)
    report["new_stub_qids"] = "|".join(new_stub_qids)
    report["enriched_at"] = datetime.now(timezone.utc).isoformat()

    return report, new_stub_qids


def label_completion_pass(output_dir: Path, delay: float = 0.5) -> int:
    """
    Post-expansion pass: find every WD entity URI mentioned in the output TTLs
    that carries no rdfs:label, and write a minimal stub TTL for each.
    Targets both entities with no TTL file of their own and entities whose file
    exists but was written without a label, which an existence check misses.
    Returns count of stubs written.
    """
    WD_PREFIX = "http://www.wikidata.org/entity/Q"

    mentioned = set()
    labelled = set()
    for ttl_path in output_dir.glob("*.ttl"):
        g = Graph()
        try:
            g.parse(ttl_path, format="turtle")
        except Exception:
            continue
        for s, p, o in g:
            for node in (s, o):
                if isinstance(node, URIRef) and str(node).startswith(WD_PREFIX):
                    mentioned.add(str(node).split("/")[-1])
            if p == RDFS.label and isinstance(s, URIRef) and str(s).startswith(WD_PREFIX):
                labelled.add(str(s).split("/")[-1])

    missing = mentioned - labelled
    print(f"[label-completion] {len(mentioned)} WD entities mentioned | "
          f"{len(labelled)} already labelled | {len(missing)} need labels")

    written = 0
    unresolved = []
    for i, qid in enumerate(sorted(missing), 1):
        print(f"  [{i}/{len(missing)}] {qid} ...", end=" ", flush=True)
        entity = fetch_entity(qid)
        if entity is None:
            print("fetch failed")
            continue

        label, label_src = pick_label(entity)
        desc = entity.get("descriptions", {}).get("en", {}).get("value", "")
        if not label:
            print("no English-safe label — skipped for review")
            unresolved.append(qid)
            time.sleep(delay)
            continue

        out_path = output_dir / f"{qid}.ttl"
        g = Graph()
        if out_path.exists():
            try:
                g.parse(out_path, format="turtle")
            except Exception:
                g = Graph()
        g.bind("schema", SCHO)
        g.bind("rdfs", RDFS)
        uri = URIRef(f"http://www.wikidata.org/entity/{qid}")
        g.add((uri, RDFS.label, Literal(label, lang="en")))
        if desc and (uri, SCHO.description, None) not in g:
            g.add((uri, SCHO.description, Literal(desc, lang="en")))

        out_path.write_text(g.serialize(format="turtle"), encoding="utf-8")
        written += 1
        print(f"✓ {label[:50]} [{label_src}]")
        time.sleep(delay)

    print(f"[label-completion] done — {written} labelled, {len(unresolved)} unresolved")
    if unresolved:
        review_path = output_dir / "unresolved_labels.txt"
        review_path.write_text("\n".join(sorted(unresolved)), encoding="utf-8")
        print(f"[label-completion] unresolved QIDs -> {review_path}")
    written += mimo_label_pass(output_dir)
    return written


def mimo_label_pass(output_dir: Path) -> int:
    """
    MIMO instrument URIs are minted from Wikidata P3763 and never carry a label
    of their own, so the WD-prefixed pass above cannot see them. Resolve each
    back through P3763 in one WDQS query and write stubs.
    """
    mentioned, labelled = set(), set()
    for ttl_path in output_dir.glob("*.ttl"):
        g = Graph()
        try:
            g.parse(ttl_path, format="turtle")
        except Exception:
            continue
        for s, p, o in g:
            for node in (s, o):
                if isinstance(node, URIRef) and str(node).startswith(str(MIMO)):
                    mentioned.add(str(node).split("/")[-1])
            if p == RDFS.label and isinstance(s, URIRef) and str(s).startswith(str(MIMO)):
                labelled.add(str(s).split("/")[-1])

    missing = sorted(mentioned - labelled)
    print(f"[mimo-labels] {len(mentioned)} MIMO ids mentioned | {len(missing)} need labels")
    if not missing:
        return 0

    values = " ".join(f'"{m}"' for m in missing)
    query = ('SELECT ?v ?l WHERE { VALUES ?v { ' + values + ' } '
             '?item wdt:P3763 ?v . ?item rdfs:label ?l . FILTER(LANG(?l) = "en") }')
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode(
        {"query": query, "format": "json"})
    req = urllib.request.Request(url, headers={
        "User-Agent": "BME-KG-Expander/1.0",
        "Accept": "application/sparql-results+json"})
    bindings = None
    for attempt, wait in enumerate((10, 65, 65), 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                bindings = json.loads(r.read())["results"]["bindings"]
            break
        except Exception as e:
            print(f"[mimo-labels] WDQS attempt {attempt} failed: {e}")
            if attempt < 3:
                print(f"[mimo-labels] retrying in {wait}s")
                time.sleep(wait)
    if bindings is None:
        print("[mimo-labels] WDQS unavailable — falling back to search API")
        resolved = mimo_labels_via_search(missing)
    else:
        resolved = {b["v"]["value"]: b["l"]["value"] for b in bindings}

    written = 0
    for mimo_id, label in sorted(resolved.items()):
        out_path = output_dir / f"mimo_{mimo_id}.ttl"
        g = Graph()
        if out_path.exists():
            try:
                g.parse(out_path, format="turtle")
            except Exception:
                g = Graph()
        g.bind("rdfs", RDFS)
        g.bind("mimo", MIMO)
        g.add((MIMO[mimo_id], RDFS.label, Literal(label, lang="en")))
        out_path.write_text(g.serialize(format="turtle"), encoding="utf-8")
        written += 1
    print(f"[mimo-labels] done — {written}/{len(missing)} resolved via P3763")
    return written


def mimo_labels_via_search(mimo_ids: list, delay: float = 0.3) -> dict:
    """
    WDQS-free resolution of MIMO ids: find the Wikidata item asserting
    P3763 = <id> through the search API, then read its label. One request per
    id, so only used when the bulk WDQS query is unavailable.
    """
    resolved = {}
    for mimo_id in mimo_ids:
        url = "https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query", "list": "search", "format": "json", "srlimit": 1,
            "srsearch": f"haswbstatement:P3763={mimo_id}"})
        req = urllib.request.Request(url, headers={"User-Agent": "BME-KG-Expander/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                hits = json.loads(r.read())["query"]["search"]
        except Exception as e:
            print(f"  [warn] MIMO {mimo_id} search failed: {e}")
            continue
        if not hits:
            continue
        entity = fetch_entity(hits[0]["title"])
        if entity:
            label, _ = pick_label(entity)
            if label:
                resolved[mimo_id] = label
        time.sleep(delay)
    return resolved


def write_report(report_path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = ["qid", "label", "kg_types", "resolved_type", "role",
                  "status", "triples_written", "new_stubs", "new_stub_qids",
                  "error", "enriched_at"]
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict], frontier: dict, total_in_frontier: int):
    from collections import Counter
    status_counts = Counter(r["status"] for r in rows)
    type_counts = Counter(r["resolved_type"] for r in rows if r["resolved_type"])
    unenriched = sum(1 for v in frontier.values() if not v.get("enriched"))

    print("\n" + "=" * 60)
    print("  EXPANSION RUN SUMMARY")
    print("=" * 60)
    print(f"  Entities processed this run : {len(rows)}")
    print(f"  Frontier total              : {total_in_frontier}")
    print(f"  Still unenriched            : {unenriched}")
    print()
    print("  Status breakdown:")
    for status, count in sorted(status_counts.items()):
        print(f"    {status:<20} {count}")
    print()
    print("  Resolved types:")
    for t, count in type_counts.most_common():
        print(f"    {t:<25} {count}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="KG Expansion — Wikidata enrichment")
    parser.add_argument("--frontier",    type=Path, default=Path("kg_expansion/frontier.json"))
    parser.add_argument("--output-dir",  type=Path, default=Path("kg_expansion/ttl_expanded"))
    parser.add_argument("--type-mapping",type=Path, default=Path("kg_expansion/type_mapping.json"))
    parser.add_argument("--overrides",   type=Path, default=Path("kg_expansion/entity_type_overrides.csv"))
    parser.add_argument("--report-dir",  type=Path, default=Path("kg_expansion/reports"))
    parser.add_argument("--type-filter", default=None,
                        help="Comma-separated normalized types to process e.g. Human,MusicGroup")
    parser.add_argument("--max-entities",type=int, default=None)
    parser.add_argument("--max-rounds",  type=int, default=2)
    parser.add_argument("--delay",       type=float, default=1.0)
    parser.add_argument("--expand-rounds", type=int, default=1,
                        help="Number of rounds where uri_label predicates add stubs. "
                             "Round 0..N-1 expand fully; round N+ terminates. "
                             "Default 1 = only direct museum entities expand.")
    parser.add_argument("--label-completion", action="store_true",
                        help="After expansion rounds, fetch labels for every WD entity "
                             "mentioned in the TTLs that has no rdfs:label, then resolve "
                             "MIMO instrument ids via P3763")
    parser.add_argument("--shacl", action="store_true", default=True,
                        help="Run SHACL validation on aggregated output after expansion (default: on)")
    parser.add_argument("--no-shacl", dest="shacl", action="store_false",
                        help="Skip SHACL validation")
    parser.add_argument("--shacl-shapes", type=Path, default=SHAPES_DEFAULT,
                        help="Path to SHACL shapes TTL")
    parser.add_argument("--shacl-strict", action="store_true",
                        help="Abort if SHACL violations are found")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Fetch and resolve but do not write TTL files")
    args = parser.parse_args()

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    type_mapping = load_type_mapping(args.type_mapping)
    overrides = load_overrides(args.overrides) if args.overrides.exists() else {}
    frontier = json.loads(args.frontier.read_text())

    total_in_frontier = len(frontier)
    filter_types = [t.strip() for t in args.type_filter.split(",")] if args.type_filter else None

    # Build candidate list — only unenriched, respecting round limit
    # Skip anything already resolved as review_needed or no_property_map — no point refetching
    SKIP_STATUSES = {"review_needed", "no_property_map", "terminal"}
    # KG-side type labels that are always terminal — no Wikidata fetch needed
    TERMINAL_KG_TYPES = {"DefinedTerm", "Country", "City", "Place", "core#Concept",
                         "MusicGenre"}

    # KG-side type label aliases → normalized PROPERTY_MAP type
    # GT KG uses different labels than our expander's normalized types
    KG_TYPE_ALIASES = {
        # EventVenue kept as-is — matches museum graph typing, maps to Venue PROPERTY_MAP
    }
    candidates = []
    for qid, entry in frontier.items():
        if entry.get("enriched"):
            continue
        if entry.get("status") in SKIP_STATUSES:
            continue
        # Pre-skip known terminal KG types — no fetch needed
        if any(t in TERMINAL_KG_TYPES for t in entry.get("types", [])):
            frontier[qid]["enriched"] = True
            frontier[qid]["status"] = "terminal"
            continue
        if entry.get("round", 0) >= args.max_rounds:
            continue
        # Remap KG-side type aliases to normalized expander types
        if entry.get("types"):
            entry["types"] = [KG_TYPE_ALIASES.get(t, t) for t in entry["types"]]
        if filter_types and entry.get("types"):
            if not any(t in entry["types"] for t in filter_types):
                continue
        candidates.append((qid, entry))
    # Persist terminal pre-skips
    if not args.dry_run:
        args.frontier.write_text(json.dumps(frontier, indent=2, ensure_ascii=False))

    if args.max_entities:
        candidates = candidates[:args.max_entities]

    # Seen set — all QIDs already in frontier (enriched or not) + anything we process
    # This is the loop guard: a QID already known is never added as a new stub
    seen = set(frontier.keys())

    print(f"[expand] {len(candidates)} candidates | "
          f"frontier total: {total_in_frontier} | "
          f"dry_run: {args.dry_run}")
    if filter_types:
        print(f"[expand] type filter: {filter_types}")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.report_dir / f"expansion_report_{run_ts}.csv"
    report_rows = []
    tier2_tracker = Tier2Tracker(threshold=10)

    for i, (qid, entry) in enumerate(candidates, 1):
        print(f"  [{i}/{len(candidates)}] {qid} ({entry.get('label', '')}) ...", end=" ", flush=True)

        result = enrich_entity(qid, entry, args.output_dir,
                               type_mapping, overrides, seen,
                               dry_run=args.dry_run,
                               expand_rounds=args.expand_rounds)

        # enrich_entity returns (report_row, new_stub_qids) on success,
        # or just report_row on terminal/error
        if isinstance(result, tuple):
            row, new_stub_qids = result
        else:
            row = result
            new_stub_qids = []

        report_rows.append(row)

        status_str = row["status"]
        if status_str == "enriched":
            print(f"✓ {row['resolved_type']} | {row['triples_written']} triples | {row['new_stubs']} new stubs")
        elif status_str == "terminal_reference":
            # Track P31 classes appearing in Tier 2 for promotion reporting
            p31_qids = [q for q in row.get("new_stub_qids", "").split("|") if q]
            tier2_tracker.record(p31_qids)
            print(f"— tier2 ({row['resolved_type']})")
        else:
            print(f"— {status_str}")

        # Mark as done in frontier — enriched, terminal_reference, and review_needed all count as
        # processed so they don't block subsequent runs. Only fetch_failed stays unenriched.
        frontier[qid]["enriched"] = status_str in ("enriched", "terminal_reference", "terminal",
                                                    "review_needed", "no_property_map")
        frontier[qid]["status"] = status_str
        frontier[qid]["enriched_at"] = row["enriched_at"]
        frontier[qid]["resolved_type"] = row["resolved_type"]
        frontier[qid]["new_stubs_added"] = new_stub_qids

        # Add new stubs to frontier (loop guard: only if not already seen)
        for stub_qid in new_stub_qids:
            if stub_qid not in seen:
                frontier[stub_qid] = {
                    "uri": f"http://www.wikidata.org/entity/{stub_qid}",
                    "label": "",
                    "types": [],
                    "enriched": False,
                    "round": entry.get("round", 0) + 1,
                    "added_by": qid,
                    "enriched_at": None,
                    "resolved_type": "",
                    "new_stubs_added": [],
                }
                seen.add(stub_qid)

        # Persist frontier after every entity — safe to interrupt
        if not args.dry_run:
            args.frontier.write_text(json.dumps(frontier, indent=2, ensure_ascii=False))

        time.sleep(args.delay)

    # Write report
    write_report(report_path, report_rows)
    print(f"\n[report] {report_path}")

    print_summary(report_rows, frontier, total_in_frontier)

    # Tier 2 promotion report
    promo = tier2_tracker.report()
    print(f"\n{promo}")

    # Label completion pass — runs after all expansion rounds
    if args.label_completion and not args.dry_run:
        print()
        label_completion_pass(args.output_dir, delay=args.delay)

    # SHACL validation — aggregate all per-entity TTLs and validate
    if args.shacl and not args.dry_run:
        print()
        print("[shacl] Aggregating TTLs for validation ...")
        from rdflib.namespace import OWL, XSD
        agg = Graph()
        agg.bind("schema", SCHO, override=True)
        agg.bind("wd",     WD,   override=True)
        agg.bind("rdfs",   RDFS, override=True)
        agg.bind("owl",    OWL,  override=True)
        agg.bind("xsd",    XSD,  override=True)
        ttl_files = sorted(args.output_dir.glob("*.ttl"))
        for f in ttl_files:
            agg.parse(f, format="turtle")
        print(f"[shacl] {len(ttl_files)} TTL files, {len(agg):,} triples")
        shacl_report = args.report_dir / "expansion_shacl.csv"
        run_shacl(
            agg,
            shapes_path=args.shacl_shapes,
            abort_on_violation=args.shacl_strict,
            report_path=shacl_report,
        )


if __name__ == "__main__":
    main()
