"""
aggregate_expansion.py

Aggregates all per-entity TTL files in kg_expansion/ttl/ into a single
expansion.ttl, then optionally merges with the main knowledge_graph.ttl.

Usage:
    python kg_expansion/aggregate_expansion.py \
        --ttl-dir kg_expansion/ttl \
        --output kg_expansion/expansion.ttl

    # Merge with main KG
    python kg_expansion/aggregate_expansion.py \
        --ttl-dir kg_expansion/ttl \
        --output kg_expansion/expansion.ttl \
        --merge ../../assets/kg_triple_store_gt/knowledge_graph.ttl \
        --merged-output ../../assets/kg_triple_store_gt/knowledge_graph_expanded.ttl
"""

import argparse
import logging
import re
import warnings
from pathlib import Path

from rdflib import Graph, Namespace
from rdflib.namespace import RDFS, OWL, XSD

from shacl_validator import run_shacl, SHAPES_DEFAULT

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

SCHO = Namespace("http://schema.org/")
WD   = Namespace("http://www.wikidata.org/entity/")
MIMO = Namespace("http://www.mimo-db.eu/InstrumentsKeywords/")


from rdflib.namespace import RDF

# When merging, if an entity already has a type, prefer the more specific one.
# Order = (type to DROP, type to KEEP when both present)
TYPE_PRIORITY = [
    (SCHO.DefinedTerm,      SCHO.MusicAlbum),
    (SCHO.DefinedTerm,      SCHO.MusicComposition),
    (SCHO.DefinedTerm,      SCHO.MusicGroup),
    (SCHO.DefinedTerm,      SCHO.Person),
    (SCHO.Person,           SCHO.MusicGroup),
    (SCHO.Venue,            SCHO.EventVenue),
    (SCHO.Place,            SCHO.EventVenue),
    (SCHO.Place,            SCHO.Country),
    (SCHO.Event,            SCHO.ConcertTour),
    (SCHO.Place,            SCHO.Event),
    (SCHO.MusicComposition, SCHO.MusicAlbum),
]


def deduplicate_types(g: Graph) -> int:
    """Remove lower-priority rdf:type triples when a higher-priority type exists."""
    removed = 0
    for drop_type, keep_type in TYPE_PRIORITY:
        for e in list(g.subjects(RDF.type, drop_type)):
            if (e, RDF.type, keep_type) in g:
                g.remove((e, RDF.type, drop_type))
                removed += 1
    return removed


def fix_schema_prefix(ttl: str) -> str:
    """Normalise any schemaX: prefix rdflib may introduce when merging graphs."""
    ttl = re.sub(r'@prefix schema\d+: <http://schema\.org/> \.', '@prefix schema: <http://schema.org/> .', ttl)
    ttl = re.sub(r'\bschema\d+:', 'schema:', ttl)
    return ttl


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl-dir", type=Path, default=Path("kg_expansion/ttl"))
    parser.add_argument("--output", type=Path, default=Path("kg_expansion/expansion.ttl"))
    parser.add_argument("--merge", type=Path, default=None,
                        help="Main KG to merge expansion into")
    parser.add_argument("--merged-output", type=Path, default=None)
    parser.add_argument("--shacl", action="store_true", default=True,
                        help="Run SHACL validation after aggregation/merge (default: on)")
    parser.add_argument("--no-shacl", dest="shacl", action="store_false",
                        help="Skip SHACL validation")
    parser.add_argument("--shacl-shapes", type=Path, default=SHAPES_DEFAULT,
                        help="Path to SHACL shapes TTL")
    parser.add_argument("--shacl-report", type=Path, default=None,
                        help="Write SHACL results to this CSV path")
    parser.add_argument("--shacl-strict", action="store_true",
                        help="Abort if SHACL violations are found")
    args = parser.parse_args()

    ttl_files = sorted(args.ttl_dir.glob("*.ttl"))
    print(f"[aggregate] {len(ttl_files)} expansion TTL files")

    g = Graph()
    g.bind("schema", SCHO, override=True)
    g.bind("wd",     WD,   override=True)
    g.bind("rdfs",   RDFS, override=True)
    g.bind("owl",    OWL,  override=True)
    g.bind("xsd",    XSD,  override=True)
    g.bind("mimo",   MIMO, override=True)
    for f in ttl_files:
        g.parse(f, format="turtle")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ttl = fix_schema_prefix(g.serialize(format="turtle"))
    args.output.write_text(ttl, encoding="utf-8")
    print(f"[done] {len(g):,} triples → {args.output}")

    if args.shacl:
        shacl_report = args.shacl_report or args.output.parent / "reports" / f"{args.output.stem}_shacl.csv"
        print(f"[shacl] Validating expansion graph ...")
        run_shacl(g, shapes_path=args.shacl_shapes, abort_on_violation=args.shacl_strict, report_path=shacl_report)

    if args.merge:
        print(f"[merge] Loading {args.merge} ...")
        kg = Graph()
        kg.bind("schema", SCHO, override=True)
        kg.bind("wd",     WD,   override=True)
        kg.bind("rdfs",   RDFS, override=True)
        kg.bind("owl",    OWL,  override=True)
        kg.bind("xsd",    XSD,  override=True)
        kg.bind("mimo",   MIMO, override=True)
        kg.parse(args.merge, format="turtle")
        print(f"  KG: {len(kg):,} triples")
        before = len(kg)
        kg += g
        added = len(kg) - before
        deduped = deduplicate_types(kg)
        if deduped:
            print(f"[merge] removed {deduped} duplicate type triples")
        out = args.merged_output or args.merge.parent / "knowledge_graph_expanded.ttl"
        merged_ttl = fix_schema_prefix(kg.serialize(format="turtle"))
        Path(out).write_text(merged_ttl, encoding="utf-8")
        print(f"[merge] +{added:,} new triples → {out}")

        if args.shacl:
            merged_report = args.shacl_report or Path(out).parent / "reports" / f"{Path(out).stem}_shacl.csv"
            print(f"[shacl] Validating merged graph ...")
            run_shacl(kg, shapes_path=args.shacl_shapes, abort_on_violation=args.shacl_strict, report_path=merged_report)


if __name__ == "__main__":
    main()
