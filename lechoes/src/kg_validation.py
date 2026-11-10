"""
kg_validation.py — Validation script for the BME knowledge graph.

Usage:
    python kg_validation.py --kg path/to/knowledge_graph.ttl [--mode all|structural|schema] [--base path/to/base.ttl]

Modes:
    structural  — URI patterns, blank nodes, namespace declarations, parse integrity
    schema      — required predicates per type, type/classification mappings, entity link targets
    all         — both (default)

--base is used when validating an expansion TTL that references entities in the main KG.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Graph, URIRef, BNode, Literal, RDF, RDFS, Namespace
from rdflib.namespace import SKOS, XSD

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

CRM   = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
SCHO  = Namespace("http://schema.org/")
WD    = Namespace("http://www.wikidata.org/entity/")
AAT   = Namespace("http://vocab.getty.edu/aat/")
EUF   = Namespace("http://thesaurus.europeanafashion.eu/thesaurus/")
MIMO  = Namespace("http://www.mimo-db.eu/InstrumentsKeywords/")
BME   = Namespace("http://radio.liv.ac.uk/bme/")

BME_BASE   = "http://radio.liv.ac.uk/bme/"
WD_BASE    = "http://www.wikidata.org/entity/"
PBCORE_BASE = "http://pbcore.org/pbcore-controlled-vocabularies/titletype-vocabulary/#"

# URI patterns for minted entities
UUID_RE  = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
HASH_RE  = re.compile(r"^[0-9a-f]{8,10}$")
INT_RE   = re.compile(r"^\d+$")
CAMEL_RE = re.compile(r"^[A-Z][A-Za-z0-9]+$")
QCODE_RE = re.compile(r"^Q\d+$")

BME_PATH_ID_FORMAT: dict[str, str] = {
    "HumanMadeObject": "hash",
    "MusicAlbum":      "hash",
    "MusicComposition":"hash",
    "Event":           "hash",
    "Person":          "hash",
    "Place":           "hash",
    "Country":         "hash",
    "EventVenue":      "hash",
    "Concept":         "hash",
    "DefinedTerm":     "hash",
    "Zone":            "int",
    "Case":            "int",
    "Exhibition":      "camel",
    "Era":             "camel",
    # "Q214609":         "hash",  # replaced by schema:DefinedTerm
}

# Required predicates per rdf:type
REQUIRED_PREDICATES: dict[URIRef, list[URIRef]] = {
    CRM["E22_Human-Made_Object"]: [
        SCHO.name,
        SCHO.identifier,
        SCHO.description,
        SCHO.location,
        SCHO.additionalType,
    ],
}

# Predicates whose objects must be typed entities (not dangling)
ENTITY_LINK_PREDICATES = {
    SCHO.about,
    SCHO.location,
    SCHO.material,
    SCHO.genre,
    SCHO.category,
    SCHO.additionalType,
}

# Predicates whose objects must be typed as one of the allowed types
TYPED_OBJECT_CONSTRAINTS: dict[URIRef, set[URIRef]] = {
    SCHO.genre:    {SCHO.DefinedTerm, SKOS.Concept, SCHO.MusicComposition},
    SCHO.category: {SCHO.DefinedTerm, SKOS.Concept},
}

# Known external vocabulary bases that are valid additionalType targets
KNOWN_VOCAB_BASES = [AAT_BASE := "http://vocab.getty.edu/aat/",
                     EUF_BASE := "http://thesaurus.europeanafashion.eu/thesaurus/",
                     MIMO_BASE := "http://www.mimo-db.eu/InstrumentsKeywords/",
                     PBCORE_BASE]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    severity: str       # "error" | "warning"
    category: str
    subject: str
    message: str
    row: int | None = None  # source row index if known

    def __str__(self):
        row_part = f" [row {self.row}]" if self.row is not None else ""
        return f"[{self.severity.upper()}] [{self.category}] {self.subject}{row_part}: {self.message}"


@dataclass
class ValidationResult:
    issues: list[Issue] = field(default_factory=list)
    hmo_row_index: dict[str, int] = field(default_factory=dict)

    def _row(self, subject: str) -> int | None:
        local = subject.split("/")[-1]
        return self.hmo_row_index.get(local)

    def error(self, category: str, subject: str, message: str):
        self.issues.append(Issue("error", category, subject, message, self._row(subject)))

    def warning(self, category: str, subject: str, message: str):
        self.issues.append(Issue("warning", category, subject, message, self._row(subject)))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> dict:
        by_category: dict[str, dict] = defaultdict(lambda: {"errors": 0, "warnings": 0})
        for issue in self.issues:
            by_category[issue.category][issue.severity + "s"] += 1
        return {
            "passed": self.passed,
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "by_category": dict(by_category),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def short(uri: str, max_len: int = 80) -> str:
    return uri if len(uri) <= max_len else "..." + uri[-max_len + 3:]


def is_bme_uri(uri: str) -> bool:
    return uri.startswith(BME_BASE)


def parse_bme_uri(uri: str) -> tuple[str, str] | None:
    """Returns (path_segment, local_id) for BME URIs, or None."""
    tail = uri[len(BME_BASE):]
    parts = tail.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None


def validate_id_format(path_segment: str, local_id: str) -> str | None:
    """Returns an error message if the ID format is wrong, else None."""
    expected = BME_PATH_ID_FORMAT.get(path_segment)
    if expected is None:
        return None  # Unknown segment — reported separately
    if expected == "hash" and not HASH_RE.match(local_id):
        return f"expected short hash, got '{local_id}'"
    if expected == "uuid" and not UUID_RE.match(local_id):
        return f"expected UUID, got '{local_id}'"
    if expected == "int" and not INT_RE.match(local_id):
        return f"expected integer, got '{local_id}'"
    if expected == "camel" and not CAMEL_RE.match(local_id):
        return f"expected CamelCase name, got '{local_id}'"
    return None


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def validate_structural(g: Graph, result: ValidationResult, ttl_path: Path,
                        declared_ns: dict | None = None):
    if declared_ns is None:
        declared_ns = {}
    subjects = set(g.subjects())
    all_uris = set()

    # Blank node check
    bnode_subjects = [s for s in subjects if isinstance(s, BNode)]
    for bn in bnode_subjects:
        result.error("blank_node", str(bn), "Blank node used as subject — all entities should be named URIs")

    for s, p, o in g:
        if isinstance(o, BNode):
            result.error("blank_node", str(s), f"Blank node used as object of {short(str(p))}")
        if isinstance(s, URIRef):
            all_uris.add(str(s))
        if isinstance(o, URIRef):
            all_uris.add(str(o))

    # URI pattern validation for BME-minted URIs
    known_paths = set(BME_PATH_ID_FORMAT.keys())
    for uri in sorted(all_uris):
        if not is_bme_uri(uri):
            continue
        parsed = parse_bme_uri(uri)
        if parsed is None:
            result.warning("uri_pattern", uri, "BME URI has no path segment after base")
            continue
        path_seg, local_id = parsed
        if path_seg not in known_paths:
            result.warning("uri_pattern", uri, f"Unknown BME path segment '/{path_seg}/'")
            continue
        msg = validate_id_format(path_seg, local_id)
        if msg:
            result.error("uri_pattern", uri, f"URI ID format invalid — {msg}")

    # Wikidata URI check
    for uri in sorted(all_uris):
        if uri.startswith(WD_BASE):
            local = uri[len(WD_BASE):]
            if not QCODE_RE.match(local):
                result.error("uri_pattern", uri, f"Wikidata URI has unexpected local name '{local}' (expected Q-code)")

    # PBCore uses hash-fragment URIs (e.g. ...vocabulary/#VinylRecording) which cannot be
    # shortened as Turtle prefixed names — rdflib will always write them as full URIs.
    # This is expected and not an error.

    # Check for namespace URIs used in the graph that aren't declared
    used_ns_bases: set[str] = set()
    for uri in all_uris:
        for known in [str(CRM), str(SCHO), str(WD), str(AAT), str(EUF), str(MIMO),
                      str(RDFS), str(SKOS), str(XSD), BME_BASE, PBCORE_BASE,
                      "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                      "http://www.w3.org/2000/01/rdf-schema#"]:
            if uri.startswith(known):
                used_ns_bases.add(known)
                break

    # These bases are declared in the TTL under project-specific prefixes (e.g. scho:, crm:, wd:)
    # but rdflib's namespace_manager may not surface them under those exact URIs because it
    # registers its own built-in aliases (e.g. schema: -> https://schema.org/).
    # They are genuinely declared — suppress the warning.
    always_ok = {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/2004/02/skos/core#",
        BME_BASE,
        PBCORE_BASE,                                      # hash-fragment URIs cannot be prefixed in Turtle
        str(SCHO),                                        # declared as scho: in TTL
        str(CRM),                                         # declared as crm: in TTL
        str(WD),                                          # declared as wd: in TTL
        str(AAT),                                         # declared as aat: in TTL
        str(EUF),                                         # declared as euf: in TTL
        str(MIMO),                                        # declared as mimo: in TTL
    }
    for base in sorted(used_ns_bases):
        if base not in declared_ns and base not in always_ok:
            result.warning("namespace", base, f"Namespace base '{base}' is used but not declared as @prefix")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_schema(g: Graph, base_g: Graph | None, result: ValidationResult):
    # Merge base graph for cross-graph entity lookups
    lookup = Graph()
    for triple in g:
        lookup.add(triple)
    if base_g is not None:
        for triple in base_g:
            lookup.add(triple)

    all_subjects = set(g.subjects(RDF.type, None))

    # -----------------------------------------------------------------------
    # 1. Required predicates per type
    # -----------------------------------------------------------------------
    for rdf_type, required_preds in REQUIRED_PREDICATES.items():
        for s in g.subjects(RDF.type, rdf_type):
            for pred in required_preds:
                if not any(True for _ in g.objects(s, pred)):
                    result.error(
                        "required_predicate",
                        str(s),
                        f"Missing required predicate {pred.n3(g.namespace_manager)}",
                    )

    # -----------------------------------------------------------------------
    # 2. scho:identifier matches URI hash
    # -----------------------------------------------------------------------
    for s in g.subjects(RDF.type, CRM["E22_Human-Made_Object"]):
        uri = str(s)
        identifier_vals = list(g.objects(s, SCHO.identifier))
        if not identifier_vals:
            continue  # already caught by required_predicate check
        parsed = parse_bme_uri(uri)
        if parsed is None:
            continue
        _, local_id = parsed
        for id_val in identifier_vals:
            if str(id_val) != local_id:
                result.error(
                    "identifier_mismatch",
                    uri,
                    f"scho:identifier value '{id_val}' does not match URI local ID '{local_id}'",
                )

    # -----------------------------------------------------------------------
    # 3. scho:caption — warn on "nan" (unfilled placeholder)
    # -----------------------------------------------------------------------
    for s in g.subjects(RDF.type, CRM["E22_Human-Made_Object"]):
        for caption in g.objects(s, SCHO.caption):
            if str(caption).strip().lower() == "nan":
                result.warning("data_quality", str(s), "scho:caption is 'nan' (unfilled placeholder)")

    # -----------------------------------------------------------------------
    # 4. Entity link targets must exist and be typed
    # -----------------------------------------------------------------------
    for s, p, o in g:
        if p not in ENTITY_LINK_PREDICATES:
            continue
        if not isinstance(o, URIRef):
            continue

        uri_str = str(o)

        # Known external vocab terms (aat, euf, mimo, pbcore) are valid untyped targets
        if any(uri_str.startswith(base) for base in KNOWN_VOCAB_BASES):
            continue

        # Check target exists in graph (or base graph)
        target_types = set(lookup.objects(o, RDF.type))
        if not target_types:
            result.warning(
                "dangling_reference",
                str(s),
                f"Object of {p.n3(g.namespace_manager)} has no rdf:type: <{short(uri_str)}>",
            )

    # -----------------------------------------------------------------------
    # 5. scho:material targets must be typed as schema:DefinedTerm
    # -----------------------------------------------------------------------
    for s in g.subjects(RDF.type, CRM["E22_Human-Made_Object"]):
        for mat in g.objects(s, SCHO.material):
            if not isinstance(mat, URIRef):
                result.error("material_type", str(s), f"scho:material value is not a URI: {mat!r}")
                continue
            mat_str = str(mat)
            target_types = set(lookup.objects(mat, RDF.type))
            if SCHO.DefinedTerm not in target_types:
                result.error(
                    "material_type",
                    str(s),
                    f"scho:material target <{short(mat_str)}> is not typed as schema:DefinedTerm",
                )

    # -----------------------------------------------------------------------
    # 6. scho:about targets must be typed
    # -----------------------------------------------------------------------
    for s in g.subjects(RDF.type, CRM["E22_Human-Made_Object"]):
        for about in g.objects(s, SCHO.about):
            if not isinstance(about, URIRef):
                continue
            target_types = set(lookup.objects(about, RDF.type))
            if not target_types:
                result.warning(
                    "entity_link",
                    str(s),
                    f"scho:about target has no rdf:type: <{short(str(about))}>",
                )

    # -----------------------------------------------------------------------
    # 7. scho:genre and scho:category targets must have an acceptable type
    # -----------------------------------------------------------------------
    for pred, allowed_types in TYPED_OBJECT_CONSTRAINTS.items():
        for s, _, o in g.triples((None, pred, None)):
            if not isinstance(o, URIRef):
                continue
            uri_str = str(o)
            if any(uri_str.startswith(base) for base in KNOWN_VOCAB_BASES):
                continue
            target_types = set(lookup.objects(o, RDF.type))
            if target_types and not target_types.intersection(allowed_types):
                result.warning(
                    "classification_type",
                    str(s),
                    f"Object of {pred.n3(g.namespace_manager)} <{short(uri_str)}> "
                    f"has unexpected types: {[short(str(t)) for t in target_types]}",
                )

    # -----------------------------------------------------------------------
    # 8. rdfs:label on supporting entities
    # -----------------------------------------------------------------------
    supporting_types = {
        SCHO.Person, SCHO.MusicAlbum, SCHO.MusicComposition,
        SCHO.Event, SCHO.Place, SCHO.Country, SCHO.EventVenue,
        SCHO.DefinedTerm,
    }
    for entity_type in supporting_types:
        for s in g.subjects(RDF.type, entity_type):
            labels = list(g.objects(s, RDFS.label))
            if not labels:
                result.warning(
                    "missing_label",
                    str(s),
                    f"Entity typed as {entity_type.n3(g.namespace_manager)} has no rdfs:label",
                )

    # -----------------------------------------------------------------------
    # 9. Wikidata entities should have rdfs:label when they appear as typed nodes
    # -----------------------------------------------------------------------
    for s in g.subjects(RDF.type, None):
        if not str(s).startswith(WD_BASE):
            continue
        labels = list(g.objects(s, RDFS.label))
        if not labels:
            result.warning(
                "missing_label",
                str(s),
                "Wikidata entity has no rdfs:label",
            )

    # -----------------------------------------------------------------------
    # 10. scho:location targets must be typed as scho:Place
    # -----------------------------------------------------------------------
    for s in g.subjects(RDF.type, CRM["E22_Human-Made_Object"]):
        for loc in g.objects(s, SCHO.location):
            if not isinstance(loc, URIRef):
                result.error("location_type", str(s), f"scho:location value is not a URI: {loc!r}")
                continue
            target_types = set(lookup.objects(loc, RDF.type))
            if target_types and SCHO.Place not in target_types:
                result.error(
                    "location_type",
                    str(s),
                    f"scho:location target <{short(str(loc))}> is not typed as scho:Place "
                    f"(got: {[short(str(t)) for t in target_types]})",
                )

    # -----------------------------------------------------------------------
    # 11. Artist label present in HMO text → must have scho:about link
    # -----------------------------------------------------------------------
    BME_PERSON_PREFIX = f"{BME_BASE}Person/"
    persons: dict[str, str] = {}
    for s, _, o in g.triples((None, RDFS.label, None)):
        if str(s).startswith(BME_PERSON_PREFIX):
            label = str(o).strip()
            if len(label) >= 4:  # skip very short labels prone to false positives
                persons[str(s)] = label

    # label → set of all person URIs sharing that label (for duplicate-aware lookup)
    label_to_uris: dict[str, set[str]] = defaultdict(set)
    for uri, label in persons.items():
        label_to_uris[label].add(uri)

    unique_labels = {label: re.compile(rf"\b{re.escape(label)}\b", re.IGNORECASE)
                     for label in label_to_uris}

    for hmo in g.subjects(RDF.type, CRM["E22_Human-Made_Object"]):
        text_parts = []
        for pred in (SCHO.description, SCHO.caption):
            for _, _, o in g.triples((hmo, pred, None)):
                text_parts.append(str(o))
        text = " ".join(text_parts)
        if not text:
            continue

        linked_persons = {str(o) for _, _, o in g.triples((hmo, SCHO.about, None))
                          if str(o).startswith(BME_PERSON_PREFIX)}

        for label, pattern in unique_labels.items():
            if not pattern.search(text):
                continue
            # label is covered if ANY person URI with this label is already linked
            if label_to_uris[label] & linked_persons:
                continue
            result.warning(
                "artist_link",
                str(hmo),
                f"Label {label!r} found in HMO text but no scho:about link to any person with this label",
            )

    # -----------------------------------------------------------------------
    # 12. Duplicate bme:Person entities with the same rdfs:label
    # -----------------------------------------------------------------------
    label_to_persons: dict[str, list[str]] = defaultdict(list)
    for s, _, o in g.triples((None, RDFS.label, None)):
        if str(s).startswith(BME_PERSON_PREFIX):
            label_to_persons[str(o).strip()].append(str(s))

    for label, uris in label_to_persons.items():
        if len(uris) > 1:
            result.warning(
                "duplicate_person",
                uris[0],
                f"Label {label!r} shared by {len(uris)} bme:Person entities: "
                + ", ".join(f"<{short(u)}>" for u in uris),
            )

    # -----------------------------------------------------------------------
    # 13. bme:Person minted for an entity that already exists as a wd: node
    # -----------------------------------------------------------------------
    wd_labels: dict[str, str] = {}
    for s, _, o in g.triples((None, RDFS.label, None)):
        if str(s).startswith(WD_BASE):
            wd_labels[str(o).strip()] = str(s)

    for s, _, o in g.triples((None, RDFS.label, None)):
        if not str(s).startswith(BME_PERSON_PREFIX):
            continue
        label = str(o).strip()
        if label in wd_labels:
            result.warning(
                "redundant_bme_person",
                str(s),
                f"bme:Person minted for {label!r} but wd entity already exists: <{short(wd_labels[label])}>",
            )


    # -----------------------------------------------------------------------
    # 14. Missing scho:temporalCoverage — only warn when text contains year
    #     patterns (1900-2099) but no temporalCoverage triples are asserted
    # -----------------------------------------------------------------------
    _YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
    for s in g.subjects(RDF.type, CRM["E22_Human-Made_Object"]):
        if any(True for _ in g.objects(s, SCHO.temporalCoverage)):
            continue  # years present — all good
        # Check if description or caption mentions a year
        text = " ".join(
            str(o) for pred in (SCHO.description, SCHO.caption)
            for o in g.objects(s, pred)
        )
        if _YEAR_RE.search(text):
            years_in_text = ", ".join(sorted(set(_YEAR_RE.findall(text))))
            result.warning(
                "missing_year",
                str(s),
                f"Text mentions year(s) {years_in_text} but no scho:temporalCoverage asserted",
            )


# ---------------------------------------------------------------------------
# Namespace inventory
# ---------------------------------------------------------------------------

def report_namespaces(g: Graph) -> dict:
    declared = {prefix: str(ns) for prefix, ns in g.namespace_manager.namespaces() if prefix}
    used_bases: set[str] = set()
    for s, p, o in g:
        for node in (s, p, o):
            if isinstance(node, URIRef):
                uri = str(node)
                for base in declared.values():
                    if uri.startswith(base):
                        used_bases.add(base)
                        break
    return {
        "declared": declared,
        "used_declared_bases": sorted(used_bases),
    }


# ---------------------------------------------------------------------------
# Row index lookup (from ingest report CSVs)
# ---------------------------------------------------------------------------

def build_hmo_row_index(reports_dir: Path) -> dict[str, int]:
    """
    Scans all ingest_report_*.csv files in reports_dir and returns a mapping
    of HMO hash → row_index, derived from the ttl_file column.
    e.g. "20260629_134720_000_0261fe56ed.ttl" → hash "0261fe56ed" → row 0
    """
    hmo_to_row: dict[str, int] = {}
    ttl_re = re.compile(r"_(\d+)_([0-9a-f]+)\.ttl$")
    for report in sorted(reports_dir.glob("ingest_report_*.csv")):
        with report.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ttl = row.get("ttl_file", "").strip()
                row_idx = row.get("row_index", "").strip()
                m = ttl_re.search(ttl)
                if m and row_idx.isdigit():
                    hmo_hash = m.group(2)
                    hmo_to_row[hmo_hash] = int(row_idx)
    return hmo_to_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> Graph:
    g = Graph()
    g.parse(str(path), format="turtle")
    return g


def run_validation(
    ttl_path: Path,
    mode: str = "all",
    base_path: Path | None = None,
    source_csv: Path | None = None,
    ttl_dir: Path | None = None,
) -> ValidationResult:
    # Build HMO hash → row_index from ingest reports if available
    reports_dir = ttl_path.parent / "reports"
    hmo_row_index: dict[str, int] = {}
    if reports_dir.is_dir():
        hmo_row_index = build_hmo_row_index(reports_dir)
        if hmo_row_index:
            print(f"Row index loaded: {len(hmo_row_index)} HMO entries from reports/")

    result = ValidationResult(hmo_row_index=hmo_row_index)

    print(f"Parsing {ttl_path} ...", flush=True)
    try:
        g = load_graph(ttl_path)
    except Exception as e:
        result.error("parse", str(ttl_path), f"TTL parse failed: {e}")
        return result

    print(f"  {len(g)} triples loaded", flush=True)

    base_g: Graph | None = None
    if base_path is not None:
        print(f"Parsing base graph {base_path} ...", flush=True)
        try:
            base_g = load_graph(base_path)
            print(f"  {len(base_g)} base triples loaded", flush=True)
        except Exception as e:
            result.warning("parse", str(base_path), f"Base graph parse failed: {e}")

    ns_info = report_namespaces(g)
    print(f"\nDeclared namespaces: {list(ns_info['declared'].keys())}")

    if mode in ("structural", "all"):
        print("\nRunning structural validation ...", flush=True)
        validate_structural(g, result, ttl_path, declared_ns=ns_info["declared"])

    if mode in ("schema", "all"):
        print("Running schema validation ...", flush=True)
        validate_schema(g, base_g, result)

    if source_csv is not None and ttl_dir is not None:
        print("Running source vs TTL check ...", flush=True)
        validate_source(result, source_csv, ttl_dir)

    return result


def print_report(result: ValidationResult, verbose: bool = False):
    summary = result.summary()
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(f"Result:   {'PASSED' if result.passed else 'FAILED'}")
    print(f"Errors:   {summary['total_errors']}")
    print(f"Warnings: {summary['total_warnings']}")

    if summary["by_category"]:
        print("\nBy category:")
        for cat, counts in sorted(summary["by_category"].items()):
            parts = []
            if counts.get("errors"):
                parts.append(f"{counts['errors']} error(s)")
            if counts.get("warnings"):
                parts.append(f"{counts['warnings']} warning(s)")
            print(f"  {cat:<30} {', '.join(parts)}")

    if result.errors or verbose:
        print("\nErrors:")
        for issue in result.errors:
            print(f"  {issue}")

    if result.warnings and verbose:
        print("\nWarnings:")
        for issue in result.warnings:
            print(f"  {issue}")
    elif result.warnings and not verbose:
        print(f"\n(Use --verbose to see {len(result.warnings)} warning(s))")


def write_json_report(result: ValidationResult, output_path: Path):
    data = result.summary()
    data["issues"] = [
        {"severity": i.severity, "category": i.category, "subject": i.subject,
         "row": i.row, "message": i.message}
        for i in result.issues
    ]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report written to {output_path}")


_LEADING_NUMBER_RE = re.compile(r"^\d+\s+")

def _strip_position_prefix(caption: str) -> str:
    return _LEADING_NUMBER_RE.sub("", caption.strip())

_ROW_TTL_RE = re.compile(r"^(\d{8}_\d{6})_(\d+)_([0-9a-f]+)\.ttl$")

SCHO_NS = Namespace("http://schema.org/")


def validate_source(
    result: ValidationResult,
    source_csv: Path,
    ttl_dir: Path,
):
    """Check each source CSV row against its corresponding per-row TTL file."""
    # Load source rows
    source_rows: dict[int, dict] = {}
    with source_csv.open(encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            source_rows[i] = {k.strip(): v.strip() for k, v in row.items()}

    # Build TTL lookup: row_number -> latest path
    ttl_by_row: dict[int, tuple[str, Path]] = {}
    for f in ttl_dir.iterdir():
        m = _ROW_TTL_RE.match(f.name)
        if not m:
            continue
        ts, row_str, _ = m.groups()
        row_num = int(row_str)
        if row_num not in ttl_by_row or ts > ttl_by_row[row_num][0]:
            ttl_by_row[row_num] = (ts, f)

    for row_number, src in source_rows.items():
        subject_id = f"row:{row_number}"
        ttl_entry = ttl_by_row.get(row_number)
        if ttl_entry is None:
            result.warning("source_check", subject_id,
                           f"No TTL file found for source row {row_number} — may not have been processed yet")
            continue

        _, ttl_path = ttl_entry
        g = Graph()
        g.parse(str(ttl_path), format="turtle")

        # Find HumanMadeObject subject
        hmo_subjects = list(g.subjects(RDF.type, CRM["E22_Human-Made_Object"]))
        if not hmo_subjects:
            result.error("source_check", subject_id,
                         f"No crm:E22_Human-Made_Object in {ttl_path.name}")
            continue
        s = hmo_subjects[0]
        subject_id = str(s)

        ttl_desc = next((str(o) for o in g.objects(s, SCHO_NS.description)), "")
        ttl_cap  = next((str(o) for o in g.objects(s, SCHO_NS.caption)),     "")

        src_desc = src.get("item_description", "").strip()
        src_cap  = src.get("caption", "").strip()

        if src_desc and not ttl_desc.strip().lower().startswith(src_desc.lower()):
            result.warning(
                "source_check", subject_id,
                f"scho:description mismatch (row {row_number}, {ttl_path.name}): "
                f"source={src_desc!r} | ttl={ttl_desc!r}",
            )

        if src_cap and src_cap.lower() != "nan":
            normalised_cap = _strip_position_prefix(src_cap)
            if ttl_cap.strip().lower() != normalised_cap.lower():
                result.warning(
                    "source_check", subject_id,
                    f"scho:caption mismatch (row {row_number}, {ttl_path.name}): "
                    f"source={normalised_cap[:80]!r} | ttl={ttl_cap[:80]!r}",
                )


def main():
    parser = argparse.ArgumentParser(description="Validate a BME knowledge graph TTL file")
    parser.add_argument("--kg", required=True, help="Path to TTL file to validate")
    parser.add_argument(
        "--mode", choices=["all", "structural", "schema"], default="all",
        help="Validation mode (default: all)",
    )
    parser.add_argument(
        "--base", default=None,
        help="Optional base KG TTL for cross-graph entity lookups (used when validating expansion files)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print all warnings")
    parser.add_argument("--report", default=None, help="Write JSON report to this path")
    parser.add_argument("--source-csv", default=None,
                        help="Source input CSV (item_descriptions_with_captions.csv) for source vs TTL check")
    parser.add_argument("--ttl-dir", default=None,
                        help="Directory of per-row TTL files (used with --source-csv)")
    args = parser.parse_args()

    ttl_path = Path(args.kg)
    if not ttl_path.exists():
        print(f"Error: file not found: {ttl_path}", file=sys.stderr)
        sys.exit(1)

    base_path  = Path(args.base)       if args.base       else None
    source_csv = Path(args.source_csv) if args.source_csv else None
    ttl_dir    = Path(args.ttl_dir)    if args.ttl_dir    else None

    result = run_validation(ttl_path, mode=args.mode, base_path=base_path,
                            source_csv=source_csv, ttl_dir=ttl_dir)
    print_report(result, verbose=args.verbose)

    if args.report:
        report_path = Path(args.report)
        write_json_report(result, report_path)
        txt_path = report_path.with_suffix(".txt")
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_report(result, verbose=True)
        txt_path.write_text(buf.getvalue(), encoding="utf-8")
        print(f"Text report written to {txt_path}")

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
