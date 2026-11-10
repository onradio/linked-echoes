"""
ingest_pipeline.py

End-to-end ingestion pipeline: raw descriptions → knowledge graph.
  1. Metadata extraction  — calls Gemini to extract structured metadata from
                            raw item descriptions and captions.
  2. Populate pipeline    — classifies, entity-links, and triplifies each item
                            into the knowledge graph (mirrors populate.py).

Run from the lechoes/src/ directory (classification.py's VOCAB_PATHS are relative
to it — see INGEST_PIPELINE.md):
    python ingest_pipeline.py path/to/input.csv --api-key YOUR_KEY

Input CSV must have at minimum:
    - item_description : free-text description of the museum object
    - caption          : physical/visual description (can be empty)

All other fields understood by the Item model (name, case_number, exhibition,
year, …) can optionally be present and will be forwarded as-is.
"""

import argparse
import csv
import json
import logging
import re
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from tqdm import tqdm

# lechoes/__init__.py initialises FastAPI, SPARQL, and DB — none of which are
# needed here. Register a lightweight stub *before* any lechoes.* import so
# that __init__.py never executes, while sub-packages (lechoes.src.*) still
# load normally through the regular package machinery.
import types as _types

_PACKAGE_DIR = Path(__file__).parent.parent          # .../lechoes/
_PACKAGE_PARENT = _PACKAGE_DIR.parent                # repository root

if "lechoes" not in sys.modules:
    _stub = _types.ModuleType("lechoes")
    _stub.__path__ = [str(_PACKAGE_DIR)]
    _stub.__package__ = "lechoes"
    sys.modules["lechoes"] = _stub

if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

# Make metadata_curation importable without package prefix
_MC_DIR = Path(__file__).parent / "metadata_curation"
if str(_MC_DIR) not in sys.path:
    sys.path.insert(0, str(_MC_DIR))

from metadata_extractor import MetadataExtractor

from lechoes.src.metamodel import Item, LinkedEntity
from lechoes.src.classification import HierarchicalObjectClassifier
from lechoes.src.linker import RAGEntityLinker
from lechoes.src.scho_triplifier import Triplifier
from lechoes.src.namespaces import bind_namespaces
from lechoes.src.populate import (
    process_single_item,
    aggregate_graphs,
    DEFAULT_MAPPINGS_PATH,
    DEFAULT_VOCAB_FILE_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_AGGREGATE_FILE,
)
from lechoes.src.kg_validation import run_validation, print_report as print_validation_report
from lechoes.src.kg_stats import load_rdf, extract_stats, print_stats
from lechoes.src.populate import process_dict_row
import io
import contextlib

# ---------------------------------------------------------------------------
# Post-aggregation validation + stats
# ---------------------------------------------------------------------------

def run_kg_inspection(kg_path: Path, reports_dir: Path, run_ts: str,
                      source_csv: Path | None = None, ttl_dir: Path | None = None):
    """Run validation and stats against a freshly aggregated KG. Non-blocking."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    validation_report = reports_dir / f"kg_validation_{run_ts}.json"
    stats_report      = reports_dir / f"kg_stats_{run_ts}.json"

    print(f"\n=== KG Validation [{datetime.now().strftime('%H:%M:%S')}] ===")
    logger.info(f"KG validation starting: kg={kg_path}")
    try:
        result = run_validation(kg_path, mode="all",
                                source_csv=source_csv, ttl_dir=ttl_dir)
        status = "PASSED" if result.passed else "FAILED"
        summary = result.summary()
        print(f"Validation {status} — {summary['total_errors']} error(s), {summary['total_warnings']} warning(s)")
        if result.errors:
            for issue in result.errors:
                print(f"  [ERROR] [{issue.category}] {issue.subject}: {issue.message}")
        with validation_report.open("w", encoding="utf-8") as f:
            data = summary
            data["issues"] = [
                {"severity": i.severity, "category": i.category,
                 "subject": i.subject, "message": i.message}
                for i in result.issues
            ]
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Validation report → {validation_report}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_validation_report(result, verbose=True)
        validation_txt = validation_report.with_suffix(".txt")
        validation_txt.write_text(buf.getvalue(), encoding="utf-8")
        print(f"Validation report → {validation_txt}")
        logger.info(f"KG validation complete: status={status}, report={validation_report}")
    except Exception as exc:
        logger.warning(f"KG validation failed unexpectedly: {exc}")
        print(f"KG validation skipped (unexpected error): {exc}")

    print(f"\n=== KG Stats [{datetime.now().strftime('%H:%M:%S')}] ===")
    logger.info(f"KG stats starting: kg={kg_path}")
    try:
        rdf = load_rdf(kg_path)
        stats = extract_stats(rdf)
        print_stats(stats)
        with stats_report.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Stats report → {stats_report}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_stats(stats)
        stats_txt = stats_report.with_suffix(".txt")
        stats_txt.write_text(buf.getvalue(), encoding="utf-8")
        print(f"Stats report → {stats_txt}")
        logger.info(f"KG stats complete: report={stats_report}")
    except Exception as exc:
        logger.warning(f"KG stats failed unexpectedly: {exc}")
        print(f"KG stats skipped (unexpected error): {exc}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field mapping: AnnotationItem → Item-compatible dict
# ---------------------------------------------------------------------------
# Keys are AnnotationItem field names; values are Item field names.
# List fields stay as lists (process_dict_row handles them).
_ANNOTATION_TO_ITEM: Dict[str, str] = {
    "base_type":     "category",       # single string → LinkedEntity
    "specific_type": "type",           # single string → LinkedEntity
    "item_title":    "name",           # required by Item
    "music_genres":  "genre",
    "music_works":   "music_work",
    "music_albums":  "music_album",
    "music_artists": "music_artist",
    "events":        "event",
    "countries":     "country",
    "places":        "place",
    "venues":        "venue",
    "colours":       "colour",
    "materials":     "material",
    "years":         "year",
}

# Item fields that hold a single LinkedEntity (not a list)
_SINGLE_ENTITY_FIELDS = {"category", "type"}


def _clean_label(value: Any) -> Any:
    """Strip rogue newlines and excess whitespace from a label string or list of strings."""
    if isinstance(value, list):
        return [" ".join(str(v).split()) for v in value if v]
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def _annotation_to_item_dict(
    extracted: Dict[str, Any],
    passthrough: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert an AnnotationItem.model_dump() result into a dict compatible
    with populate.process_dict_row().

    Args:
        extracted:   Output of MetadataExtractor.extract() (AnnotationItem fields).
        passthrough: Optional extra fields already in Item format (e.g. case_number,
                     exhibition, caption) that come straight from the input CSV.
    """
    item_dict: Dict[str, Any] = {}

    for ann_field, item_field in _ANNOTATION_TO_ITEM.items():
        value = extracted.get(ann_field)
        if value is None:
            continue
        value = _clean_label(value)
        if item_field in _SINGLE_ENTITY_FIELDS:
            item_dict[item_field] = str(value)
        else:
            item_dict[item_field] = value  # keep as list or str

    # Fallback: if the model didn't produce a title, build one from type/category
    if not item_dict.get("name"):
        fallback = (
            item_dict.get("type")
            or item_dict.get("category")
            or "Untitled Item"
        )
        item_dict["name"] = str(fallback)

    # Merge passthrough fields (they take lower priority than extracted data)
    if passthrough:
        for k, v in passthrough.items():
            if k not in item_dict and v is not None and v == v:  # v == v rejects NaN without crashing on lists
                item_dict[k] = v

    return item_dict


# ---------------------------------------------------------------------------
# Passthrough fields: columns in the CSV that map directly to Item fields
# without going through the extractor.
# Covers the actual CSV column names used in testset_flat_nt.csv and similar.
# ---------------------------------------------------------------------------
_PASSTHROUGH_FIELDS = {
    "name", "case_number", "exhibition", "category", "type",
    "genre", "music_work", "music_album", "music_artist",
    "event", "country", "place", "venue", "year",
    "colour", "material", "image", "audio", "video",
    "lender", "lend_start", "lend_end",
}

# Columns that feed the extractor — excluded from passthrough
_EXTRACTOR_INPUT_COLS = {"description", "item_description", "caption"}

_ITEM_MODEL_FIELDS = set(Item.model_fields.keys())


def _build_passthrough(row: pd.Series) -> Dict[str, Any]:
    """Extract any Item-compatible fields present in the CSV row."""
    result = {}
    for col, value in row.items():
        if col in _EXTRACTOR_INPUT_COLS:
            continue
        if col in _PASSTHROUGH_FIELDS and col in _ITEM_MODEL_FIELDS:
            result[col] = value
    return result


def _ground_truth_to_extracted(row: pd.Series) -> Dict[str, Any]:
    """
    Build an extracted dict from a hand-filled ground truth CSV row,
    matching the shape returned by MetadataExtractor.extract().
    List fields are split on commas; empty/NaN values become empty lists or "".
    """
    def val(col: str) -> str:
        v = row.get(col, "")
        return "" if v != v or str(v).strip().lower() == "nan" else str(v).strip()

    def listval(col: str) -> List[str]:
        v = val(col)
        if not v:
            return []
        # Support both comma and pipe as separators
        sep = "|" if "|" in v else ","
        return [x.strip() for x in v.split(sep) if x.strip()]

    return {
        "base_type":     val("base_type"),
        "specific_type": val("specific_type"),
        "item_title":    val("item_title"),
        "music_genres":  listval("music_genres"),
        "music_works":   listval("music_works"),
        "music_albums":  listval("music_albums"),
        "music_artists": listval("music_artists"),
        "events":        listval("events"),
        "countries":     listval("countries"),
        "places":        listval("places"),
        "venues":        listval("venues"),
        "years":         listval("years"),
        "colours":       listval("colours"),
        "materials":     listval("materials"),
        "cardinality":   val("cardinality"),
        "item_description": val("item_description"),
    }


# ---------------------------------------------------------------------------
# Gemini circuit breaker
# ---------------------------------------------------------------------------
MAX_CONSECUTIVE_GEMINI_FAILURES = 3
MAX_TOTAL_GEMINI_FAILURES = 5

gemini_consecutive_failures = 0
gemini_total_failures = 0


class GeminiCircuitBreakerOpen(Exception):
    pass


def is_gemini_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    indicators = [
        "429", "resource_exhausted", "quota", "rate limit",
        "too many requests", "503", "unavailable",
        "deadline exceeded", "timeout", "internal error", "500",
        "400", "api_key_invalid", "api key not valid",
    ]
    return any(i in text for i in indicators)


def _extract_with_circuit_breaker(extractor: MetadataExtractor, raw: Dict[str, Any]) -> Any:
    """Call extractor.extract(raw) and apply Gemini circuit breaker logic."""
    global gemini_consecutive_failures, gemini_total_failures
    try:
        result = extractor.extract(raw)
        gemini_consecutive_failures = 0
        return result
    except Exception as exc:
        if is_gemini_failure(exc):
            gemini_consecutive_failures += 1
            gemini_total_failures += 1
            logger.warning(
                f"Gemini failure: consecutive={gemini_consecutive_failures}, "
                f"total={gemini_total_failures}"
            )
            if (gemini_consecutive_failures >= MAX_CONSECUTIVE_GEMINI_FAILURES or
                    gemini_total_failures >= MAX_TOTAL_GEMINI_FAILURES):
                logger.critical(
                    f"Gemini circuit breaker triggered: consecutive={gemini_consecutive_failures}, "
                    f"total={gemini_total_failures}"
                )
                raise GeminiCircuitBreakerOpen(
                    f"Circuit breaker open: consecutive={gemini_consecutive_failures}, "
                    f"total={gemini_total_failures}"
                ) from exc
        raise


# ---------------------------------------------------------------------------
# Core TTL generation
# ---------------------------------------------------------------------------
# The collection's physical layout — zone_definitions.csv and zone_case_mappings.csv —
# is institution data, not workflow code, so it is read from a directory the caller
# names (--layout-dir), defaulting to the input CSV's own directory: the layout files
# travel with the collection export they describe. They used to be read from this
# repository's assets/data/, which tied every run to one museum's files.
_CORE_TTL_DEST_NAME = "core.ttl"

# Static object-category definitions (Wikidata URIs → rdfs:label)
_OBJECT_CATEGORIES = [
    ("http://www.wikidata.org/entity/Q34379",   "Musical Instrument"),
    ("http://www.wikidata.org/entity/Q11460",    "Clothing"),
    ("http://www.wikidata.org/entity/Q758901",   "Audiovisual Media"),
    ("http://www.wikidata.org/entity/Q49848",    "Textual and Graphic Works"),
    ("http://www.wikidata.org/entity/Q28877",    "Merchandise"),
    ("http://www.wikidata.org/entity/Q115866017","Personal Effects"),
    ("http://www.wikidata.org/entity/Q190573",   "Award and Recognition"),
]

_BME_OPENING_DATE = "2017-03-09"


def _to_camel(name: str) -> str:
    """Convert a zone name to CamelCase for use in URIs, e.g. 'Punk & New Romantics' → 'PunkAndNewRomantics'."""
    return re.sub(r"[^a-zA-Z0-9]", " ", name).title().replace(" ", "")


def generate_core_ttl(output_dir: Path, layout_dir: Path) -> None:
    """Generate core.ttl in output_dir from zone_definitions.csv and zone_case_mappings.csv,
    both read from layout_dir.

    Produces:
    - Object category definitions (static Wikidata terms)
    - Exhibition entries (one per zone, CamelCase URI, museum opening date)
    - Era Event nodes (historical period each zone covers)
    - Zone Place nodes (bme:Zone/{n})
    - Case Place nodes (bme:Case/{n}, containedInPlace → Zone)

    Overwrites any existing core.ttl in output_dir so it stays current with the data files.
    """
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, RDFS, SKOS, XSD

    SCHO = Namespace("http://schema.org/")
    BME = Namespace("http://radio.liv.ac.uk/bme/")

    g = Graph()
    g.bind("scho",   SCHO)
    g.bind("skos",   SKOS)
    g.bind("rdfs",   RDFS)
    g.bind("xsd",    XSD)
    g.bind("wd",     Namespace("http://www.wikidata.org/entity/"))
    g.bind("bme",    BME)
    g.bind("crm",    Namespace("http://www.cidoc-crm.org/cidoc-crm/"))
    g.bind("aat",    Namespace("http://vocab.getty.edu/aat/"))
    g.bind("mimo",   Namespace("http://www.mimo-db.eu/InstrumentsKeywords/"))
    g.bind("euf",    Namespace("http://thesaurus.europeanafashion.eu/thesaurus/"))
    g.bind("pbcore", Namespace("http://pbcore.org/pbcore-controlled-vocabularies/titletype-vocabulary#"))

    # --- Object categories ---
    for uri_str, label in _OBJECT_CATEGORIES:
        uri = URIRef(uri_str)
        g.add((uri, RDF.type,    SCHO.DefinedTerm))
        g.add((uri, RDF.type,    SKOS.Concept))
        g.add((uri, RDFS.label,  Literal(label, lang="en")))

    # --- Zone definitions ---
    zone_defs_path = layout_dir / "zone_definitions.csv"
    zone_defs: Dict[str, Any] = {}
    if zone_defs_path.exists():
        import pandas as _pd
        for _, row in _pd.read_csv(zone_defs_path, dtype=str).iterrows():
            z = str(row.get("zone", "")).strip()
            if z:
                zone_defs[z] = {
                    "name":        str(row.get("name", f"Zone {z}")).strip(),
                    "era_start":   str(row.get("era_start", "")).strip(),
                    "era_end":     "" if row.get("era_end") != row.get("era_end") else str(row.get("era_end", "")).strip(),
                    "description": str(row.get("description", "")).strip(),
                }
    else:
        logger.warning(f"zone_definitions.csv not found at {zone_defs_path} (set --layout-dir); "
                       "zone entries will be omitted.")

    for zone_num, zd in sorted(zone_defs.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        name        = zd["name"]
        camel       = _to_camel(name)
        zone_uri    = BME[f"Zone/{zone_num}"]
        exhibit_uri = BME[f"Exhibition/{camel}"]
        era_uri     = BME[f"Era/{camel}"]

        # Zone Place node
        g.add((zone_uri, RDF.type,   SCHO.Place))
        g.add((zone_uri, RDFS.label, Literal(f"Zone {zone_num}", lang="en")))

        # ExhibitionEvent
        g.add((exhibit_uri, RDF.type,          SCHO.ExhibitionEvent))
        g.add((exhibit_uri, SCHO.name,          Literal(name, lang="en")))
        g.add((exhibit_uri, RDFS.label,         Literal(f"Exhibition: {name}", lang="en")))
        g.add((exhibit_uri, SCHO.startDate,     Literal(_BME_OPENING_DATE, datatype=XSD.date)))
        g.add((exhibit_uri, SCHO.location,       zone_uri))
        g.add((exhibit_uri, SCHO["about"],      era_uri))
        if zd["description"]:
            g.add((exhibit_uri, SCHO.description, Literal(zd["description"], lang="en")))

        # Era Event
        g.add((era_uri, RDF.type,   SCHO.Event))
        g.add((era_uri, SCHO.name,  Literal(name, lang="en")))
        g.add((era_uri, RDFS.label, Literal(name, lang="en")))
        if zd["era_start"]:
            g.add((era_uri, SCHO.startDate, Literal(zd["era_start"], datatype=XSD.gYear)))
        if zd["era_end"]:
            g.add((era_uri, SCHO.endDate,   Literal(zd["era_end"],   datatype=XSD.gYear)))

    # --- Case nodes (from zone_case_mappings.csv) ---
    zone_case_path = layout_dir / "zone_case_mappings.csv"
    if zone_case_path.exists():
        import pandas as _pd
        for _, row in _pd.read_csv(zone_case_path, dtype=str).iterrows():
            cn   = str(row.get("case_number", "")).strip()
            zn   = str(row.get("zone", "")).strip()
            cname = str(row.get("case_name", f"Case {cn}")).strip()
            if not cn:
                continue
            case_uri = BME[f"Case/{cn}"]
            zone_uri = BME[f"Zone/{zn}"]
            g.add((case_uri, RDF.type,                           SCHO.Place))
            g.add((case_uri, RDFS.label,                         Literal(cname, lang="en")))
            g.add((case_uri, URIRef("http://schema.org/containedInPlace"), zone_uri))
    else:
        logger.warning(f"zone_case_mappings.csv not found at {zone_case_path} (set --layout-dir); "
                       "case entries will be omitted.")

    dest = output_dir / _CORE_TTL_DEST_NAME
    g.serialize(destination=str(dest), format="turtle")
    logger.info(f"Generated core.ttl → {dest} ({len(g)} triples)")


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------
def run_stage(stage_name: str, fn, row_context: dict, *args, **kwargs):
    logger.info(f"Stage starting: {stage_name}")
    try:
        result = fn(*args, **kwargs)
        row_context["stage_results"].append({
            "stage": stage_name,
            "status": "success",
            "message": "",
        })
        row_context["last_successful_stage"] = stage_name
        logger.info(f"Stage complete: {stage_name}")
        return result
    except Exception as exc:
        row_context["stage_results"].append({
            "stage": stage_name,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
        row_context["status"] = "failed"
        row_context["failed_stage"] = stage_name
        row_context["failure_type"] = type(exc).__name__
        row_context["failure_message"] = str(exc)
        row_context["failure_trace"] = traceback.format_exc()
        logger.error(
            f"Stage failed: {stage_name} — {type(exc).__name__}: {exc}"
        )
        raise


# ---------------------------------------------------------------------------
# Report CSV
# ---------------------------------------------------------------------------
REPORT_COLUMNS = [
    "source_file", "row_index", "record_id", "status", "last_successful_stage",
    "failed_stage", "failure_type", "failure_message", "failure_trace",
    "output_file", "ttl_file", "processed_at", "retryable",
]


def write_report_row(report_writer: csv.DictWriter, report_file, row_context: dict) -> None:
    row = {col: row_context.get(col, "") for col in REPORT_COLUMNS}
    row["retryable"] = str(row_context.get("retryable", False)).lower()
    report_writer.writerow(row)
    report_file.flush()
    logger.debug(
        f"Report row written: row_index={row_context.get('row_index')}, "
        f"status={row_context.get('status')}"
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def update_checkpoint(
    checkpoint_path: Path,
    source_file: str,
    row_index: int,
    status: str = "in_progress",
) -> None:
    checkpoint: Dict[str, Any] = {}
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            checkpoint = json.load(f)
    checkpoint[source_file] = {
        "last_processed_row_index": row_index,
        "last_processed_at": datetime.utcnow().isoformat(),
        "status": status,
    }
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    logger.debug(
        f"Checkpoint written: {source_file} row_index={row_index} status={status}"
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    api_key: str,
    output_dir: Path,
    aggregate_file: str,
    vocab_path: Path,
    base_uri: str,
    uri_strategy: str,
    embedding_model: str,
    generator_model: str,
    classifier_top_types: int,
    linker_top_entities: int,
    extractor_config_overrides: Optional[Dict] = None,
    n_rows: Optional[int] = None,
    debug: bool = False,
    resume: bool = False,
    row_filter: Optional[Set[int]] = None,
    report_suffix: str = "",
    ground_truth_rows: Optional[Dict[int, pd.Series]] = None,
    linker_cache_path: Optional[Path] = None,
    reuse_extractions_dir: Optional[Path] = None,
    layout_dir: Optional[Path] = None,
):
    layout_dir = layout_dir if layout_dir is not None else input_path.parent

    # --- Load input CSV ---
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    # Accept either 'description' (actual CSV) or 'item_description' as the text column
    if "description" in df.columns:
        df = df.rename(columns={"description": "item_description"})
    elif "item_description" not in df.columns:
        raise ValueError("Input CSV must contain a 'description' or 'item_description' column.")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(exist_ok=True, parents=True)
    ttl_dir = output_dir / "ttl"
    ttl_dir.mkdir(exist_ok=True)
    generate_core_ttl(output_dir, layout_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    extractions_dir = output_dir / "extractions"
    extractions_dir.mkdir(exist_ok=True)

    # Build row_index → extraction dict from a prior run's extractions dir
    reuse_extraction_map: Dict[int, Dict] = {}
    if reuse_extractions_dir and reuse_extractions_dir.is_dir():
        import re as _re
        for f in reuse_extractions_dir.glob("*.json"):
            m = _re.search(r"_(\d+)\.json$", f.name)
            if m:
                idx = int(m.group(1))
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    reuse_extraction_map[idx] = data
                except Exception:
                    pass
        logger.info(f"Reuse extractions: {len(reuse_extraction_map)} entries loaded from {reuse_extractions_dir}")
    checkpoint_path = output_dir / "checkpoints" / "ingest_checkpoint.json"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

    manifest: Dict[str, Any] = {
        "run_timestamp": run_ts,
        "input_file": str(input_path),
        "config": {
            "n_rows": n_rows,
            "debug": debug,
            "generator_model": generator_model,
            "embedding_model": embedding_model,
            "uri_strategy": uri_strategy,
            "base_uri": base_uri,
        },
        "items": [],
    }

    if debug and n_rows is None:
        df = df.head(5)
        print("Debug mode: processing first 5 rows only.")

    print(f"Loaded {len(df)} rows from {input_path}.")

    # --- Resume from checkpoint ---
    last_processed_row_index = -1
    if resume:
        if checkpoint_path.exists():
            with checkpoint_path.open("r", encoding="utf-8") as f:
                existing_checkpoint = json.load(f)
            file_checkpoint = existing_checkpoint.get(str(input_path), {})
            last_processed_row_index = file_checkpoint.get("last_processed_row_index", -1)
            logger.info(
                f"Resuming from checkpoint: skipping rows 0..{last_processed_row_index}, "
                f"resuming from row {last_processed_row_index + 1}"
            )
        else:
            logger.info("--resume specified but no checkpoint found; starting from beginning.")

    logger.info(
        f"Pipeline starting: input_file={input_path}, resume={resume}, "
        f"row_filter={'set of ' + str(len(row_filter)) + ' rows' if row_filter is not None else 'none'}"
    )

    # --- Open report CSV ---
    report_filename = (
        f"ingest_report_retry_{run_ts}.csv" if report_suffix == "retry"
        else f"ingest_report_{run_ts}.csv"
    )
    report_path = reports_dir / report_filename
    report_file = open(report_path, "w", newline="", encoding="utf-8")
    report_writer = csv.DictWriter(report_file, fieldnames=REPORT_COLUMNS)
    report_writer.writeheader()
    report_file.flush()

    # --- Initialise services ---
    print(f"\n=== Initialising services ===")
    extractor = MetadataExtractor(config_overrides=extractor_config_overrides, api_key=api_key)

    with open(DEFAULT_MAPPINGS_PATH, "r") as f:
        mappings = json.load(f)
    vocab_df = pd.read_csv(vocab_path)

    # --- Load zone/case location lookup ---
    _zone_case_path = layout_dir / "zone_case_mappings.csv"
    location_lookup: Dict[str, Any] = {}
    if _zone_case_path.exists():
        _zone_df = pd.read_csv(_zone_case_path, dtype=str)
        for _, _zrow in _zone_df.iterrows():
            _cn = str(_zrow.get("case_number", "")).strip()
            if _cn:
                location_lookup[_cn] = {
                    "zone": str(_zrow.get("zone", "")).strip(),
                    "case_name": str(_zrow.get("case_name", f"Case {_cn}")).strip(),
                }
        logger.info(f"Loaded location lookup: {len(location_lookup)} cases from {_zone_case_path}")
    else:
        logger.warning(f"zone_case_mappings.csv not found at {_zone_case_path} (set --layout-dir); "
                       "location info will be omitted.")

    print("Initialising services (Classifier, Linker, Triplifier)...")
    classifier = HierarchicalObjectClassifier(
        generator_api_key=api_key,
        embedding_model=embedding_model,
        generator_model=generator_model,
    )
    linker = RAGEntityLinker(
        generator_api_key=api_key,
        generator_model=generator_model,
        cache_path=linker_cache_path,
    )
    triplifier = Triplifier(
        mappings_path=str(DEFAULT_MAPPINGS_PATH),
        base_uri=base_uri,
        uri_strategy=uri_strategy,
        location_lookup=location_lookup,
    )

    # --- Unified per-row processing loop ---
    last_row_index = -1
    rows_skipped = 0
    rows_processed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        row_index = int(idx)

        # Skip rows based on row_filter (retry mode)
        if row_filter is not None and row_index not in row_filter:
            continue

        # Skip rows already processed per checkpoint (resume mode)
        if resume and row_index <= last_processed_row_index:
            rows_skipped += 1
            logger.info(f"Skipping row {row_index} (already processed per checkpoint).")
            continue

        if rows_skipped > 0 and row_index == last_processed_row_index + 1:
            logger.info(
                f"Skipped {rows_skipped} rows due to checkpoint; "
                f"resuming from row {row_index}."
            )
            rows_skipped = 0

        # Apply n_rows limit after skipping checkpointed rows
        if n_rows is not None and rows_processed >= n_rows:
            logger.info(f"Reached --n-rows limit of {n_rows}; stopping.")
            break

        rows_processed += 1
        logger.info(f"Row starting: row_index={row_index}")

        _raw_caption = row.get("caption", "")
        caption = "" if _raw_caption != _raw_caption else str(_raw_caption or "")  # NaN != NaN
        _case_match = re.match(r"^(\d+)\s+", caption)
        item_number = _case_match.group(1) if _case_match else ""
        caption = caption[_case_match.end():] if _case_match else caption
        record_id = str(row.get("case_number", "") or "")

        row_context: Dict[str, Any] = {
            "source_file": str(input_path),
            "row_index": row_index,
            "record_id": record_id,
            "status": "pending",
            "last_successful_stage": "",
            "failed_stage": "",
            "failure_type": "",
            "failure_message": "",
            "failure_trace": "",
            "output_file": "",
            "ttl_file": "",
            "processed_at": datetime.utcnow().isoformat(),
            "retryable": False,
            "stage_results": [],
        }

        manifest_entry: Dict[str, Any] = {
            "row": row_index,
            "description_snippet": str(row.get("item_description", ""))[:80],
        }

        try:
            # Stage 1: extract (Gemini call site) — bypassed for ground truth rows
            raw = {
                "item_description": row.get("item_description", ""),
                "caption": caption,
            }

            gt_row = (ground_truth_rows or {}).get(row_index)
            reuse_data = reuse_extraction_map.get(row_index)
            if gt_row is not None:
                extracted = _ground_truth_to_extracted(gt_row)
                logger.info(f"Row {row_index}: using ground truth extraction (skipping Gemini)")
                row_context["stage_results"].append({
                    "stage": "extract",
                    "status": "ground_truth",
                    "message": "Bypassed — ground truth data used",
                })
                row_context["last_successful_stage"] = "extract"
            elif reuse_data is not None:
                extracted = reuse_data["extracted"]
                logger.info(f"Row {row_index}: reusing prior extraction (skipping Gemini)")
                row_context["stage_results"].append({
                    "stage": "extract",
                    "status": "reused",
                    "message": "Bypassed — prior extraction reused",
                })
                row_context["last_successful_stage"] = "extract"
            else:
                extracted = run_stage(
                    "extract",
                    _extract_with_circuit_breaker,
                    row_context,
                    extractor, raw,
                )

            manifest_entry["status"] = "extracted"
            manifest_entry["name"] = extracted.get("item_title", "")
            manifest_entry["category"] = extracted.get("base_type", "")
            manifest_entry["type"] = extracted.get("specific_type", "")

            # Persist extraction JSON for flow review (ground truth or LLM)
            extraction_record = {
                "input": raw,
                "extracted": extracted,
                "source": "ground_truth" if gt_row is not None else ("reused" if reuse_data is not None else "gemini"),
            }
            extraction_filename = f"{run_ts}_{row_index:03d}.json"
            extraction_path = extractions_dir / extraction_filename
            with extraction_path.open("w", encoding="utf-8") as _ef:
                json.dump(extraction_record, _ef, indent=2, ensure_ascii=False)
            row_context["output_file"] = extraction_filename
            logger.debug(f"Extraction JSON written: {extraction_path}")
            print(
                f"  [{row_index}] {'[GT] ' if gt_row is not None else '[RE] ' if reuse_data is not None else ''}"
                f"{manifest_entry['name']!r:40s} | "
                f"{manifest_entry['category']!r} → {manifest_entry['type']!r}"
            )

            # Stage 2: process_dict_row
            passthrough = _build_passthrough(row)
            item_dict = _annotation_to_item_dict(extracted, passthrough)
            item_dict["caption"] = raw["caption"]
            item_dict["description"] = raw["item_description"]
            if item_number:
                item_dict["item_number"] = item_number
            unknown = [k for k in item_dict if k not in _ITEM_MODEL_FIELDS]
            if unknown:
                warnings.warn(f"Row {row_index}: unknown fields {unknown} will be ignored.")
            item = run_stage("process_dict_row", process_dict_row, row_context, item_dict)

            manifest_entry["identifier"] = item.identifier

            # Stage 3: process_single_item
            run_stage(
                "process_single_item",
                process_single_item,
                row_context,
                item, classifier, linker, triplifier,
                mappings, output_dir,
                vocab_df=vocab_df,
                classifier_top_types=classifier_top_types,
                linker_top_entities=linker_top_entities,
                ttl_dir=ttl_dir,
            )

            # Rename TTL to include run timestamp and row number
            old_ttl = ttl_dir / f"{item.identifier}.ttl"
            new_ttl = ttl_dir / f"{run_ts}_{row_index:03d}_{item.identifier}.ttl"
            if old_ttl.exists():
                old_ttl.rename(new_ttl)
            row_context["ttl_file"] = new_ttl.name
            manifest_entry["status"] = "triplified"
            manifest_entry["ttl"] = new_ttl.name

            row_context["status"] = "success"

        except GeminiCircuitBreakerOpen:
            row_context["retryable"] = True
            write_report_row(report_writer, report_file, row_context)
            update_checkpoint(checkpoint_path, str(input_path), row_index)
            manifest_entry.setdefault("status", row_context.get("failed_stage", "extract") + "_failed")
            manifest_entry["error"] = row_context["failure_message"]
            manifest["items"].append(manifest_entry)
            report_file.close()
            logger.critical("Gemini circuit breaker open — halting pipeline.")
            raise

        except Exception:
            row_context["retryable"] = True
            write_report_row(report_writer, report_file, row_context)
            update_checkpoint(checkpoint_path, str(input_path), row_index)
            manifest_entry.setdefault("status", row_context.get("failed_stage", "unknown") + "_failed")
            manifest_entry["error"] = row_context["failure_message"]
            manifest["items"].append(manifest_entry)
            continue

        write_report_row(report_writer, report_file, row_context)
        update_checkpoint(checkpoint_path, str(input_path), row_index)
        manifest["items"].append(manifest_entry)
        last_row_index = row_index

    report_file.close()

    # Mark checkpoint as completed for this file
    if last_row_index >= 0:
        update_checkpoint(checkpoint_path, str(input_path), last_row_index, "completed")

    print(f"\nReport CSV → {report_path}")

    # --- Step 4: Aggregate into single knowledge graph ---
    _step_start = datetime.now()
    logger.info(f"TTL aggregation starting: ttl_dir={ttl_dir}")
    ttl_files = list(ttl_dir.glob("*.ttl"))
    logger.info(f"TTL aggregation: {len(ttl_files)} files found.")
    print(f"\n=== Step 4: Aggregating Knowledge Graph  [{_step_start.strftime('%H:%M:%S')}] ===")
    final_output = output_dir / aggregate_file
    aggregate_graphs(ttl_dir, final_output)
    elapsed_agg = (datetime.now() - _step_start).total_seconds()
    logger.info(f"TTL aggregation complete: output={final_output}")
    print(f"Aggregation complete. ({elapsed_agg:.1f}s)")

    run_kg_inspection(final_output, output_dir / "reports", run_ts,
                      source_csv=input_path, ttl_dir=ttl_dir)

    # --- Save manifest + human-readable report ---
    run_end = datetime.now()
    total_elapsed = (run_end - datetime.strptime(run_ts, "%Y%m%d_%H%M%S")).total_seconds()

    manifest["summary"] = {
        "total_rows": len(df),
        "extracted": sum(1 for e in manifest["items"] if e.get("status") == "extracted"),
        "triplified": sum(1 for e in manifest["items"] if e.get("status") == "triplified"),
        "failed": sum(1 for e in manifest["items"] if "failed" in e.get("status", "")),
        "no_type": sum(1 for e in manifest["items"] if e.get("status") == "triplified" and not e.get("type")),
        "knowledge_graph": str(final_output),
        "total_elapsed_s": round(total_elapsed, 1),
    }
    manifest_path = logs_dir / f"run_manifest_{run_ts}.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    _write_report(manifest, output_dir, logs_dir, run_ts)


def _write_report(manifest: Dict[str, Any], output_dir: Path, logs_dir: Path, run_ts: str) -> None:
    s = manifest["summary"]
    cfg = manifest["config"]
    items = manifest["items"]
    report_path = logs_dir / f"run_report_{run_ts}.txt"

    lines = [
        "=" * 70,
        "  BME KNOWLEDGE GRAPH PIPELINE — RUN REPORT",
        "=" * 70,
        f"  Date / time  : {run_ts[:4]}-{run_ts[4:6]}-{run_ts[6:8]}  "
        f"{run_ts[9:11]}:{run_ts[11:13]}:{run_ts[13:15]}",
        f"  Input file   : {manifest['input_file']}",
        f"  Output dir   : {output_dir}",
        f"  Elapsed      : {s['total_elapsed_s']}s",
        "",
        "  Configuration",
        "  -------------",
        f"  Generator model  : {cfg['generator_model']}",
        f"  Embedding model  : {cfg['embedding_model']}",
        f"  URI strategy     : {cfg['uri_strategy']}",
        f"  Base URI         : {cfg['base_uri']}",
        f"  Row limit        : {cfg['n_rows'] or 'none'}",
        "",
        "  Summary",
        "  -------",
        f"  Total rows       : {s['total_rows']}",
        f"  Extracted        : {s['extracted']}",
        f"  Triplified       : {s['triplified']}",
        f"  Failed           : {s['failed']}",
        f"  No type assigned : {s['no_type']}  (classifier returned 0, no vocab match)",
        f"  Knowledge graph  : {s['knowledge_graph']}",
        "",
        "  Item Detail",
        "  -----------",
    ]

    for e in items:
        status = e.get("status", "unknown")
        name = e.get("name", "—")
        cat = e.get("category", "")
        typ = e.get("type", "")
        identifier = e.get("identifier", "")
        snippet = e.get("description_snippet", "")[:60]
        if status == "triplified" and not typ:
            status_mark = "NT"  # no type
        elif status == "triplified":
            status_mark = "OK"
        else:
            status_mark = "!!"
        lines.append(
            f"  [{status_mark}] row {e['row']:>3}  {name:<35s}  "
            f"{cat} → {typ or '(no type)'}"
        )
        if identifier:
            lines.append(f"          id={identifier}  \"{snippet}\"")
        if "error" in e:
            lines.append(f"          ERROR: {e['error']}")

    lines += [
        "",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nReport   → {report_path}")
    print(f"Manifest → {logs_dir / f'run_manifest_{run_ts}.json'}")
    print(f"Graph    → {s['knowledge_graph']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: metadata extraction → classify → link → triplify."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        # PRIMARY pipeline input — required for normal ingestion runs.
        # Not used when --aggregate-ttl-only is set (aggregation reads existing TTL files instead).
        # For aggregate-only runs that also need source vs TTL validation, pass --source-csv instead.
        help="Path to input CSV with 'item_description' (and optionally 'caption') columns. "
             "Required for normal pipeline runs; omit only when using --aggregate-ttl-only.",
    )
    parser.add_argument("--api-key", required=True, help="Gemini API key.")
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for TTL output files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--aggregate-file", default=DEFAULT_AGGREGATE_FILE,
        help=f"Name of the aggregated knowledge graph file. Default: {DEFAULT_AGGREGATE_FILE}",
    )
    parser.add_argument(
        "--vocab-path", default=str(DEFAULT_VOCAB_FILE_PATH),
        help=f"Path to vocabulary CSV. Default: {DEFAULT_VOCAB_FILE_PATH}",
    )
    parser.add_argument(
        "--base-uri", default="http://radio.liv.ac.uk/bme/",
        help="Base URI for minting new resource URIs.",
    )
    parser.add_argument(
        "--uri-strategy", choices=["uuid", "slug","hash"], default="hash",
        help="URI minting strategy.",
    )
    parser.add_argument(
        "--embedding-model", default="all-MiniLM-L6-v2",
        help="Embedding model for similarity tasks.",
    )
    parser.add_argument(
        "--generator-model", default="gemini-2.5-flash",
        help="Generator model for classification and linking.",
    )
    parser.add_argument(
        "--linker-top-entities", type=int, default=10,
        help="Top-k entities for linking.",
    )
    parser.add_argument(
        "--classifier-top-types", type=int, default=20,
        help="Top-k types for classification.",
    )
    parser.add_argument(
        "--n-rows", type=int, default=None,
        help="Limit processing to the first N rows.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Process only the first 5 rows.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint, skipping already-processed rows.",
    )
    parser.add_argument(
        "--retry-report", metavar="PATH",
        help="Path to a report CSV; reprocess only failed+retryable rows.",
    )
    parser.add_argument(
        "--aggregate-ttl-only", action="store_true",
        help="Skip ingestion; only aggregate existing TTL files into a knowledge graph.",
    )
    parser.add_argument(
        "--ground-truth-csv", metavar="PATH", default=None,
        help="Path to ground truth CSV (with row_index column). Rows present in this file "
             "bypass Gemini extraction and use hand-filled data directly.",
    )
    parser.add_argument(
        "--reuse-extractions-dir", metavar="PATH", default=None,
        help="Path to a prior run's extractions/ directory. Rows with matching "
             "row_index JSON files skip Gemini and reuse the prior extraction.",
    )
    parser.add_argument(
        "--linker-cache", metavar="PATH", default=None,
        help="Path to linker cache JSON file. Cache hits skip Wikidata retrieval and Gemini. "
             "Created automatically on first run.",
    )
    parser.add_argument(
        "--source-csv", metavar="PATH",
        # AGGREGATE-ONLY companion — only needed when running --aggregate-ttl-only without a csv_path.
        # In normal pipeline runs csv_path IS the source CSV and is wired through automatically;
        # this flag exists so the post-aggregation source vs TTL validation can find the original
        # input when no ingestion is happening and csv_path is not provided.
        help="Source input CSV for source vs TTL validation (used with --aggregate-ttl-only when "
             "csv_path is not supplied). In normal runs the pipeline uses csv_path automatically.",
    )
    parser.add_argument(
        "--layout-dir", metavar="DIR", default=None,
        help="Directory holding the collection layout files (zone_definitions.csv, "
             "zone_case_mappings.csv) used for core.ttl and object locations. "
             "Default: the input CSV's directory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # --- Aggregate-only mode ---
    if args.aggregate_ttl_only:
        ttl_dir = output_dir / "ttl"
        out_path = output_dir / args.aggregate_file
        ttl_files = list(ttl_dir.glob("*.ttl"))
        logger.info(f"TTL aggregation starting: ttl_dir={ttl_dir}")
        logger.info(f"TTL aggregation: {len(ttl_files)} files found.")
        print(f"Aggregating {len(ttl_files)} TTL files from {ttl_dir} → {out_path}")
        aggregate_graphs(ttl_dir, out_path)
        logger.info(f"TTL aggregation complete: output={out_path}")
        print(f"Done. Aggregated graph → {out_path}")
        _agg_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _source_csv = Path(args.source_csv) if args.source_csv else None
        run_kg_inspection(out_path, output_dir / "reports", _agg_ts,
                          source_csv=_source_csv, ttl_dir=ttl_dir)
        return

    # Load ground truth rows if provided (needed for both retry and normal mode)
    ground_truth_rows: Optional[Dict[int, pd.Series]] = None
    if args.ground_truth_csv:
        gt_path = Path(args.ground_truth_csv)
        if not gt_path.exists():
            parser.error(f"--ground-truth-csv file not found: {gt_path}")
        gt_df = pd.read_csv(gt_path)
        if "row_index" not in gt_df.columns:
            parser.error("--ground-truth-csv must have a 'row_index' column")
        ground_truth_rows = {int(r["row_index"]): r for _, r in gt_df.iterrows()}
        print(f"Ground truth loaded: {len(ground_truth_rows)} rows will bypass Gemini extraction.")

    # --- Retry mode ---
    if args.retry_report:
        retry_report_path = Path(args.retry_report)
        if not retry_report_path.exists():
            print(f"Error: retry report not found: {retry_report_path}")
            sys.exit(1)

        retry_df = pd.read_csv(retry_report_path)
        retryable_rows = retry_df[
            (retry_df["status"] == "failed") & (retry_df["retryable"].astype(str).str.lower() == "true")
        ]

        if retryable_rows.empty:
            print("No retryable rows found in report.")
            return

        # Group by source_file and process each
        for source_file, group in retryable_rows.groupby("source_file"):
            row_indices: Set[int] = set(group["row_index"].astype(int).tolist())
            print(
                f"Retrying {len(row_indices)} rows from {source_file}: "
                f"row indices {sorted(row_indices)}"
            )
            run_pipeline(
                input_path=Path(source_file),
                api_key=args.api_key,
                output_dir=output_dir,
                aggregate_file=args.aggregate_file,
                vocab_path=Path(args.vocab_path),
                base_uri=args.base_uri,
                uri_strategy=args.uri_strategy,
                embedding_model=args.embedding_model,
                generator_model=args.generator_model,
                classifier_top_types=args.classifier_top_types,
                linker_top_entities=args.linker_top_entities,
                n_rows=args.n_rows,
                debug=args.debug,
                row_filter=row_indices,
                report_suffix="retry",
                linker_cache_path=Path(args.linker_cache) if args.linker_cache else None,
                ground_truth_rows=ground_truth_rows,
                reuse_extractions_dir=Path(args.reuse_extractions_dir) if args.reuse_extractions_dir else None,
                layout_dir=Path(args.layout_dir) if args.layout_dir else None,
            )
        return

    # --- Normal mode ---
    if not args.csv_path:
        parser.error("csv_path is required unless --aggregate-ttl-only is set.")

    run_pipeline(
        input_path=Path(args.csv_path),
        api_key=args.api_key,
        output_dir=output_dir,
        aggregate_file=args.aggregate_file,
        vocab_path=Path(args.vocab_path),
        base_uri=args.base_uri,
        uri_strategy=args.uri_strategy,
        embedding_model=args.embedding_model,
        generator_model=args.generator_model,
        classifier_top_types=args.classifier_top_types,
        linker_top_entities=args.linker_top_entities,
        n_rows=args.n_rows,
        debug=args.debug,
        resume=args.resume,
        ground_truth_rows=ground_truth_rows,
        linker_cache_path=Path(args.linker_cache) if args.linker_cache else None,
        reuse_extractions_dir=Path(args.reuse_extractions_dir) if args.reuse_extractions_dir else None,
        layout_dir=Path(args.layout_dir) if args.layout_dir else None,
    )


if __name__ == "__main__":
    main()
