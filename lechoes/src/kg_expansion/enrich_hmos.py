"""
enrich_hmos.py

Post-merge HMO enrichment pass. Run this AFTER aggregate_expansion.py has
produced knowledge_graph_full_2hop.ttl.

Three enrichment tasks:
  1. Genre fill     — propagate artist genre → HMOs missing schema:genre
  2. Instrument     — propagate artist instrument → HMOs (association, not identity)
  3. relatedTo      — link HMO siblings sharing the same artist AND case

Usage:
    # In-place (modifies the KG file directly):
    python kg_expansion/enrich_hmos.py \\
        --kg ../../assets/kg_triple_store_gt/knowledge_graph_full_2hop.ttl

    # Write patch TTLs instead (keeps source KG unchanged):
    python kg_expansion/enrich_hmos.py \\
        --kg ../../assets/kg_triple_store_gt/knowledge_graph_full_2hop.ttl \\
        --patch-dir kg_expansion/patches

    # Skip individual tasks:
    python kg_expansion/enrich_hmos.py \\
        --kg ../../assets/kg_triple_store_gt/knowledge_graph_full_2hop.ttl \\
        --no-genre --no-instrument

NOTE: This script must be run after every expansion pipeline run that updates
knowledge_graph_full_2hop.ttl. See PIPELINE_OVERVIEW.md.
"""

import argparse
import logging
import re
import warnings
from pathlib import Path

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SCHO = Namespace("http://schema.org/")
CRM  = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
HMO_TYPE = URIRef("http://www.cidoc-crm.org/cidoc-crm/E22_Human-Made_Object")

# Maximum same-case siblings to link per HMO (Task 3)
MAX_RELATED = 3


def fix_schema_prefix(ttl: str) -> str:
    ttl = re.sub(
        r"@prefix schema\d+: <http://schema\.org/> \.",
        "@prefix schema: <http://schema.org/> .",
        ttl,
    )
    ttl = re.sub(r"\bschema\d+:", "schema:", ttl)
    return ttl


def load_graph(kg_path: Path) -> Graph:
    g = Graph()
    g.bind("schema", SCHO, override=True)
    g.bind("rdfs",   RDFS,  override=True)
    g.bind("owl",    OWL,   override=True)
    g.bind("xsd",    XSD,   override=True)
    g.bind("crm",    CRM,   override=True)
    g.parse(kg_path, format="turtle")
    logger.info(f"[load] {kg_path.name} — {len(g):,} triples")
    return g


def task_genre(g: Graph) -> Graph:
    """Propagate schema:genre from linked artists onto HMOs that have none."""
    patch = Graph()
    patch.bind("schema", SCHO, override=True)

    hmos = set(g.subjects(RDF.type, HMO_TYPE))
    filled = 0
    for hmo in hmos:
        if list(g.objects(hmo, SCHO.genre)):
            continue  # already has genre
        genres = set()
        for artist in g.objects(hmo, SCHO.about):
            for genre in g.objects(artist, SCHO.genre):
                genres.add(genre)
        for genre in genres:
            patch.add((hmo, SCHO.genre, genre))
        if genres:
            filled += 1

    triples = len(patch)
    logger.info(
        f"[genre]      {filled} HMOs filled — {triples} new triples"
        + (f" (target >480 total)" if filled else "")
    )
    return patch


def task_instrument(g: Graph) -> Graph:
    """Propagate schema:instrument from linked artists onto HMOs.

    Distinction to maintain downstream:
      schema:additionalType <MIMO URI>  →  the object IS an instrument
      schema:instrument     <MIMO URI>  →  the object is ASSOCIATED WITH a player of that instrument
    """
    patch = Graph()
    patch.bind("schema", SCHO, override=True)

    hmos = set(g.subjects(RDF.type, HMO_TYPE))
    filled = 0
    for hmo in hmos:
        instruments = set()
        for artist in g.objects(hmo, SCHO.about):
            for inst in g.objects(artist, SCHO.instrument):
                instruments.add(inst)
        for inst in instruments:
            patch.add((hmo, SCHO.instrument, inst))
        if instruments:
            filled += 1

    triples = len(patch)
    logger.info(f"[instrument] {filled} HMOs filled — {triples} new triples")
    return patch


def task_related_to(g: Graph, max_related: int = MAX_RELATED) -> Graph:
    """Link HMO siblings that share the same artist AND the same case.

    Edges are bidirectional but stored as directed pairs to avoid duplicates.
    Capped at max_related siblings per HMO to avoid combinatorial explosion.
    """
    patch = Graph()
    patch.bind("schema", SCHO, override=True)

    hmos = list(g.subjects(RDF.type, HMO_TYPE))

    # Build index: (artist, case) → [hmo, ...]
    from collections import defaultdict
    bucket: dict = defaultdict(list)
    for hmo in hmos:
        cases   = list(g.objects(hmo, SCHO.location))
        artists = list(g.objects(hmo, SCHO.about))
        for artist in artists:
            for case in cases:
                bucket[(str(artist), str(case))].append(hmo)

    edges: set = set()
    for siblings in bucket.values():
        if len(siblings) < 2:
            continue
        for hmo in siblings:
            added = 0
            for sibling in siblings:
                if sibling == hmo:
                    continue
                pair = tuple(sorted([str(hmo), str(sibling)]))
                if pair not in edges:
                    edges.add(pair)
                    patch.add((hmo, SCHO.relatedTo, sibling))
                    patch.add((sibling, SCHO.relatedTo, hmo))
                    added += 1
                if added >= max_related:
                    break

    logger.info(f"[relatedTo]  {len(edges)} pairs — {len(patch)} new triples")
    return patch


def enrich(kg_path: Path, patch_dir: Path | None,
           do_genre: bool, do_instrument: bool, do_related: bool) -> None:
    g = load_graph(kg_path)
    patches: list[tuple[str, Graph]] = []

    if do_genre:
        patches.append(("genre", task_genre(g)))
    if do_instrument:
        patches.append(("instrument", task_instrument(g)))
    if do_related:
        patches.append(("relatedto", task_related_to(g)))

    total_new = sum(len(p) for _, p in patches)
    if total_new == 0:
        logger.info("[done] No new triples — KG already enriched.")
        return

    if patch_dir:
        patch_dir.mkdir(parents=True, exist_ok=True)
        for name, patch in patches:
            out = patch_dir / f"kg_patch_{name}_{kg_path.stem}.ttl"
            out.write_text(
                fix_schema_prefix(patch.serialize(format="turtle")),
                encoding="utf-8",
            )
            logger.info(f"[write] {out} ({len(patch):,} triples)")
    else:
        # Merge directly into the KG file
        for _, patch in patches:
            g += patch
        kg_path.write_text(
            fix_schema_prefix(g.serialize(format="turtle")),
            encoding="utf-8",
        )
        logger.info(f"[write] {kg_path.name} — {len(g):,} triples (+{total_new:,} new)")

    # Validation summary — merge patches into g temporarily for accurate counts
    g_check = g
    for _, patch in patches:
        g_check = g_check + patch
    hmos = set(g_check.subjects(RDF.type, HMO_TYPE))
    n = len(hmos)
    with_genre  = sum(1 for h in hmos if list(g_check.objects(h, SCHO.genre)))
    with_inst   = sum(1 for h in hmos if list(g_check.objects(h, SCHO.instrument)))
    with_rel    = sum(1 for h in hmos if list(g_check.objects(h, SCHO.relatedTo)))
    logger.info(
        f"[validate] HMOs with genre={with_genre}/{n} | "
        f"instrument={with_inst}/{n} | relatedTo={with_rel}/{n}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-merge HMO enrichment pass")
    parser.add_argument(
        "--kg", dest="kg_paths", type=Path, action="append", required=True,
        help="Merged KG file(s) to enrich (1-hop and/or 2-hop); repeat for multiple files",
    )
    parser.add_argument(
        "--patch-dir", type=Path, default=None,
        help="Write patch TTLs here instead of modifying KG in place",
    )
    parser.add_argument("--no-genre",      action="store_true", help="Skip genre fill")
    parser.add_argument("--no-instrument", action="store_true", help="Skip instrument propagation")
    parser.add_argument("--no-related",    action="store_true", help="Skip relatedTo edges")
    parser.add_argument(
        "--max-related", type=int, default=MAX_RELATED,
        help=f"Max sibling links per HMO (default {MAX_RELATED})",
    )
    args = parser.parse_args()

    for kg_path in args.kg_paths:
        if not kg_path.exists():
            logger.error(f"Not found: {kg_path}")
            continue
        logger.info(f"\n{'='*60}")
        logger.info(f"Enriching: {kg_path.name}")
        logger.info(f"{'='*60}")
        enrich(
            kg_path=kg_path,
            patch_dir=args.patch_dir,
            do_genre=not args.no_genre,
            do_instrument=not args.no_instrument,
            do_related=not args.no_related,
        )


if __name__ == "__main__":
    main()
