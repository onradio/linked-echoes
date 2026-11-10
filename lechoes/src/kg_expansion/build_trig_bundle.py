"""
build_trig_bundle.py

Assembles the BME KG layer bundle into a single TriG file with named graphs.
Each layer gets its own named graph URI so consumers can load selectively.

Named graph URIs:
    urn:bme:kg:base                  — pipeline output (knowledge_graph.ttl)
    urn:bme:kg:expansion-1hop        — 1-hop Wikidata enrichment
    urn:bme:kg:expansion-2hop-delta  — 2nd-hop new entities only
    urn:bme:kg:patch-genre           — genre fill on museum objects
    urn:bme:kg:patch-instrument      — instrument association on museum objects
    urn:bme:kg:patch-relatedto       — same-case sibling links

Output: assets/kg_triple_store_gt/bme_knowledge_graph.trig

Usage:
    python kg_expansion/build_trig_bundle.py
    python kg_expansion/build_trig_bundle.py --no-2hop   # 1-hop only bundle
    python kg_expansion/build_trig_bundle.py --out path/to/output.trig
"""

import argparse
import logging
import re
import warnings
from pathlib import Path

from rdflib import ConjunctiveGraph, Dataset, Graph, Namespace, URIRef
from rdflib.namespace import RDFS, OWL, XSD

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SCHO = Namespace("http://schema.org/")
WD   = Namespace("http://www.wikidata.org/entity/")
MIMO = Namespace("http://www.mimo-db.eu/InstrumentsKeywords/")
CRM  = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
BME  = Namespace("http://radio.liv.ac.uk/bme/")

# Named graph URIs
GRAPHS = {
    "base":                 URIRef("urn:bme:kg:base"),
    "expansion-1hop":       URIRef("urn:bme:kg:expansion-1hop"),
    "expansion-2hop-delta": URIRef("urn:bme:kg:expansion-2hop-delta"),
    "patch-genre":          URIRef("urn:bme:kg:patch-genre"),
    "patch-instrument":     URIRef("urn:bme:kg:patch-instrument"),
    "patch-relatedto":      URIRef("urn:bme:kg:patch-relatedto"),
}

GT_DIR = Path("../../assets/bme_kg_bundle")

LAYERS = [
    ("base",                 GT_DIR / "knowledge_graph.ttl"),
    ("expansion-1hop",       GT_DIR / "expansion_1hop.ttl"),
    ("expansion-2hop-delta", GT_DIR / "expansion_2hop_delta.ttl"),
    ("patch-genre",          GT_DIR / "patches/kg_patch_genre.ttl"),
    ("patch-instrument",     GT_DIR / "patches/kg_patch_instrument.ttl"),
    ("patch-relatedto",      GT_DIR / "patches/kg_patch_relatedto.ttl"),
]


def bind_prefixes(g):
    g.bind("schema", SCHO)
    g.bind("wd",     WD)
    g.bind("rdfs",   RDFS)
    g.bind("owl",    OWL)
    g.bind("xsd",    XSD)
    g.bind("mimo",   MIMO)
    g.bind("crm",    CRM)
    g.bind("bme",    BME)
    return g


def build(out_path: Path, include_2hop: bool) -> None:
    ds = Dataset()
    bind_prefixes(ds)

    total = 0
    for name, ttl_path in LAYERS:
        if name == "expansion-2hop-delta" and not include_2hop:
            logger.info(f"[skip]  {name} (--no-2hop)")
            continue
        if not ttl_path.exists():
            logger.warning(f"[missing] {ttl_path} — skipping {name}")
            continue

        # Load into a plain graph then add to named graph in dataset
        g = Graph()
        g.parse(ttl_path, format="turtle")

        named = ds.graph(GRAPHS[name])
        for triple in g:
            named.add(triple)

        logger.info(f"[graph]  urn:bme:kg:{name:25} {len(named):>7,} triples  ← {ttl_path.name}")
        total += len(named)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.serialize(destination=str(out_path), format="trig")
    logger.info(f"\n[done]  {out_path}  ({total:,} triples across {len(list(ds.contexts()))} named graphs)")
    logger.info(
        "\nLoad in SPARQL / rdflib:\n"
        "    ds = Dataset()\n"
        f'    ds.parse("{out_path}", format="trig")\n\n'
        "    # Query across all graphs:\n"
        "    ds.query('SELECT * WHERE { GRAPH ?g { ?s ?p ?o } }')\n\n"
        "    # Query a single layer:\n"
        "    g = ds.graph(URIRef('urn:bme:kg:base'))\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Build BME KG TriG bundle")
    parser.add_argument(
        "--out", type=Path,
        default=GT_DIR / "bme_knowledge_graph.trig",
        help="Output TriG file path",
    )
    parser.add_argument(
        "--no-2hop", action="store_true",
        help="Exclude 2-hop delta (1-hop bundle only)",
    )
    args = parser.parse_args()
    build(args.out, include_2hop=not args.no_2hop)


if __name__ == "__main__":
    main()
