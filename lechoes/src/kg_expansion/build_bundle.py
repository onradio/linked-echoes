"""
build_bundle.py

Regenerates the full bme_kg_bundle/ from scratch in one command.

Steps:
  1. Copy source KG (knowledge_graph.ttl) into bundle
  2. Aggregate ttl_gt/ → expansion_1hop.ttl
  3. Build expansion_2hop_delta.ttl (new round-2 entities only)
  4. Run HMO enrichment → patches/ (genre, instrument, relatedTo)
  5. Build merged convenience file (knowledge_graph_full_2hop.ttl)
  6. Build TriG bundle (bme_knowledge_graph.trig) with named graphs
  7. Generate void.ttl (VoID dataset descriptor)
  8. Append to CHANGELOG.md

Usage:
    python kg_expansion/build_bundle.py
    python kg_expansion/build_bundle.py --no-2hop
    python kg_expansion/build_bundle.py --version 2026.07.28
    python kg_expansion/build_bundle.py --bundle-dir ../../assets/bme_kg_bundle
"""

import argparse
import logging
import re
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, FOAF, RDF, RDFS, OWL, XSD

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------
SCHO  = Namespace("http://schema.org/")
WD    = Namespace("http://www.wikidata.org/entity/")
MIMO  = Namespace("http://www.mimo-db.eu/InstrumentsKeywords/")
CRM   = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
BME   = Namespace("http://radio.liv.ac.uk/bme/")
VOID  = Namespace("http://rdfs.org/ns/void#")
PROV  = Namespace("http://www.w3.org/ns/prov#")

HMO_TYPE = URIRef("http://www.cidoc-crm.org/cidoc-crm/E22_Human-Made_Object")

# Named graph URIs
GRAPH_URIS = {
    "base":                 URIRef("urn:bme:kg:base"),
    "expansion-1hop":       URIRef("urn:bme:kg:expansion-1hop"),
    "expansion-2hop-delta": URIRef("urn:bme:kg:expansion-2hop-delta"),
    "patch-genre":          URIRef("urn:bme:kg:patch-genre"),
    "patch-instrument":     URIRef("urn:bme:kg:patch-instrument"),
    "patch-relatedto":      URIRef("urn:bme:kg:patch-relatedto"),
}

LAYER_DESCRIPTIONS = {
    "base":                 "Pipeline output — 502 museum objects with stubs for every referenced entity",
    "expansion-1hop":       "1-hop Wikidata enrichment — structured facts for all directly referenced entities",
    "expansion-2hop-delta": "2nd-hop delta — new entities discovered via round-1 enrichment (band members, labels, etc.)",
    "patch-genre":          "Genre fill — propagates artist genre onto museum objects missing schema:genre",
    "patch-instrument":     "Instrument association — propagates artist instruments onto linked museum objects",
    "patch-relatedto":      "Sibling links — schema:relatedTo between museum objects sharing the same case and artist",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def fix_schema_prefix(ttl: str) -> str:
    ttl = re.sub(r"@prefix schema\d+: <http://schema\.org/> \.", "@prefix schema: <http://schema.org/> .", ttl)
    ttl = re.sub(r"\bschema\d+:", "schema:", ttl)
    return ttl


def bind(g):
    for prefix, ns in [
        ("schema", SCHO), ("wd", WD), ("rdfs", RDFS), ("owl", OWL),
        ("xsd", XSD), ("mimo", MIMO), ("crm", CRM), ("bme", BME),
    ]:
        g.bind(prefix, ns, override=True)
    return g


def load(path: Path) -> Graph:
    g = bind(Graph())
    g.parse(path, format="turtle")
    return g


def save(g: Graph, path: Path) -> None:
    path.write_text(fix_schema_prefix(g.serialize(format="turtle")), encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 1 — copy source KG
# ---------------------------------------------------------------------------
def copy_source(src_kg: Path, bundle_dir: Path) -> Path:
    dst = bundle_dir / "knowledge_graph.ttl"
    dst.write_bytes(src_kg.read_bytes())
    g = load(dst)
    logger.info(f"[1] knowledge_graph.ttl copied — {len(g):,} triples")
    return dst


# ---------------------------------------------------------------------------
# Step 2 — expansion_1hop.ttl
# ---------------------------------------------------------------------------
def build_1hop(ttl_gt_dir: Path, bundle_dir: Path) -> Graph:
    files = sorted(ttl_gt_dir.glob("*.ttl"))
    g = bind(Graph())
    for f in files:
        g.parse(f, format="turtle")
    out = bundle_dir / "expansion_1hop.ttl"
    save(g, out)
    logger.info(f"[2] expansion_1hop.ttl — {len(g):,} triples ({len(files)} entities)")
    return g


# ---------------------------------------------------------------------------
# Step 3 — expansion_2hop_delta.ttl
# ---------------------------------------------------------------------------
def build_2hop_delta(ttl_gt_dir: Path, ttl_gt_2hop_dir: Path, bundle_dir: Path) -> Graph:
    hop1_stems = {p.stem for p in ttl_gt_dir.glob("*.ttl")}
    new_files = [p for p in sorted(ttl_gt_2hop_dir.glob("*.ttl")) if p.stem not in hop1_stems]
    g = bind(Graph())
    for f in new_files:
        g.parse(f, format="turtle")
    out = bundle_dir / "expansion_2hop_delta.ttl"
    save(g, out)
    logger.info(f"[3] expansion_2hop_delta.ttl — {len(g):,} triples ({len(new_files)} new entities)")
    return g


# ---------------------------------------------------------------------------
# Step 4 — HMO enrichment → patches
# ---------------------------------------------------------------------------
def build_patches(base_g: Graph, expansion_g: Graph, patches_dir: Path) -> dict[str, Graph]:
    from collections import defaultdict

    patches_dir.mkdir(exist_ok=True)
    g = bind(Graph())
    g += base_g
    g += expansion_g

    hmos = set(g.subjects(RDF.type, HMO_TYPE))

    # genre
    pg = bind(Graph())
    for hmo in hmos:
        if list(g.objects(hmo, SCHO.genre)):
            continue
        for artist in g.objects(hmo, SCHO.about):
            for genre in g.objects(artist, SCHO.genre):
                pg.add((hmo, SCHO.genre, genre))
    save(pg, patches_dir / "kg_patch_genre.ttl")
    logger.info(f"[4a] kg_patch_genre.ttl      — {len(pg):,} triples")

    # instrument
    pi = bind(Graph())
    for hmo in hmos:
        for artist in g.objects(hmo, SCHO.about):
            for inst in g.objects(artist, SCHO.instrument):
                pi.add((hmo, SCHO.instrument, inst))
    save(pi, patches_dir / "kg_patch_instrument.ttl")
    logger.info(f"[4b] kg_patch_instrument.ttl — {len(pi):,} triples")

    # relatedTo
    pr = bind(Graph())
    bucket: dict = defaultdict(list)
    for hmo in hmos:
        for artist in g.objects(hmo, SCHO.about):
            for case in g.objects(hmo, SCHO.location):
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
                    pr.add((hmo, SCHO.relatedTo, sibling))
                    pr.add((sibling, SCHO.relatedTo, hmo))
                    added += 1
                if added >= 3:
                    break
    save(pr, patches_dir / "kg_patch_relatedto.ttl")
    logger.info(f"[4c] kg_patch_relatedto.ttl  — {len(pr):,} triples ({len(edges)} pairs)")

    return {"genre": pg, "instrument": pi, "relatedto": pr}


# ---------------------------------------------------------------------------
# Step 5 — merged convenience files (copy as-is from pipeline output)
# ---------------------------------------------------------------------------
def build_merged(bundle_dir: Path, source_dir: Path, include_2hop: bool) -> None:
    """Copy pre-built merged KGs from the pipeline output directory unchanged.

    Re-serialising through rdflib would normalise prefixes (e.g. scho: → schema:)
    and alter the files. We copy verbatim to preserve whatever the pipeline produced.
    """
    import shutil

    if include_2hop:
        src = source_dir / "knowledge_graph_full_2hop.ttl"
        dst = bundle_dir / "knowledge_graph_full_2hop.ttl"
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"[5] knowledge_graph_full_2hop.ttl — copied from {src.name}")
        else:
            logger.warning(f"[5] {src} not found — skipping")


# ---------------------------------------------------------------------------
# Step 6 — TriG bundle
# ---------------------------------------------------------------------------
def build_trig(bundle_dir: Path, include_2hop: bool) -> dict[str, int]:
    layers = [
        ("base",                 bundle_dir / "knowledge_graph.ttl"),
        ("expansion-1hop",       bundle_dir / "expansion_1hop.ttl"),
        ("expansion-2hop-delta", bundle_dir / "expansion_2hop_delta.ttl"),
        ("patch-genre",          bundle_dir / "patches/kg_patch_genre.ttl"),
        ("patch-instrument",     bundle_dir / "patches/kg_patch_instrument.ttl"),
        ("patch-relatedto",      bundle_dir / "patches/kg_patch_relatedto.ttl"),
    ]

    ds = Dataset()
    bind(ds)
    counts = {}
    for name, path in layers:
        if name == "expansion-2hop-delta" and not include_2hop:
            continue
        if not path.exists():
            logger.warning(f"[6] missing {path.name} — skipping")
            continue
        g = Graph()
        g.parse(path, format="turtle")
        named = ds.graph(GRAPH_URIS[name])
        for triple in g:
            named.add(triple)
        counts[name] = len(named)
        logger.info(f"[6] urn:bme:kg:{name:25} {len(named):>7,} triples")

    out = bundle_dir / "bme_knowledge_graph.trig"
    ds.serialize(destination=str(out), format="trig")
    total = sum(counts.values())
    logger.info(f"[6] bme_knowledge_graph.trig — {total:,} triples across {len(counts)} named graphs")
    return counts


# ---------------------------------------------------------------------------
# Step 7 — void.ttl
# ---------------------------------------------------------------------------
def build_void(bundle_dir: Path, layer_counts: dict[str, int], version: str) -> None:
    g = Graph()
    g.bind("void",    VOID)
    g.bind("dcterms", DCTERMS)
    g.bind("foaf",    FOAF)
    g.bind("prov",    PROV)
    g.bind("schema",  SCHO)
    g.bind("xsd",     XSD)

    now = Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime)

    # Root dataset
    root = URIRef("urn:bme:kg:dataset")
    g.add((root, RDF.type,           VOID.DatasetDescription))
    g.add((root, DCTERMS.title,      Literal("BME Knowledge Graph Bundle", lang="en")))
    g.add((root, DCTERMS.description, Literal(
        "Wikidata-enriched knowledge graph for the British Music Experience. "
        "Covers 502 museum objects with linked artist, album, venue, event and song data.",
        lang="en"
    )))
    g.add((root, DCTERMS.created,    now))
    g.add((root, DCTERMS.modified,   now))
    g.add((root, OWL.versionInfo,    Literal(version)))
    g.add((root, DCTERMS.publisher,  Literal("British Music Experience / University of Liverpool", lang="en")))
    g.add((root, DCTERMS.license,    URIRef("https://creativecommons.org/licenses/by/4.0/")))
    g.add((root, VOID.rootResource,  URIRef("http://radio.liv.ac.uk/bme/")))

    # Named graph / sub-dataset per layer
    for name, count in layer_counts.items():
        ds_uri = GRAPH_URIS[name]
        g.add((root,   VOID.subset,          ds_uri))
        g.add((ds_uri, RDF.type,             VOID.Dataset))
        g.add((ds_uri, DCTERMS.title,        Literal(f"BME KG — {name}", lang="en")))
        g.add((ds_uri, DCTERMS.description,  Literal(LAYER_DESCRIPTIONS.get(name, ""), lang="en")))
        g.add((ds_uri, VOID.triples,         Literal(count, datatype=XSD.integer)))
        g.add((ds_uri, DCTERMS.modified,     now))
        g.add((ds_uri, VOID.dataDump,        URIRef(f"urn:bme:kg:bundle:{name}")))
        g.add((ds_uri, PROV.wasDerivedFrom,  URIRef("urn:bme:kg:base") if name != "base" else root))

    # Key predicates used across the bundle
    for pred in [SCHO.about, SCHO.genre, SCHO.instrument, SCHO.member,
                 SCHO.birthDate, SCHO.nationality, RDFS.label, SCHO.relatedTo]:
        g.add((root, VOID.property, pred))

    out = bundle_dir / "void.ttl"
    out.write_text(g.serialize(format="turtle"), encoding="utf-8")
    logger.info(f"[7] void.ttl — {len(g)} triples")


# ---------------------------------------------------------------------------
# Step 8 — CHANGELOG.md
# ---------------------------------------------------------------------------
def update_changelog(bundle_dir: Path, version: str, layer_counts: dict[str, int],
                     include_2hop: bool) -> None:
    total = sum(layer_counts.values())
    base  = layer_counts.get("base", 0)
    hop1  = layer_counts.get("expansion-1hop", 0)
    hop2  = layer_counts.get("expansion-2hop-delta", 0)

    entry_lines = [
        f"## {version}  ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})\n",
        f"\n",
        f"| Layer | Triples |\n",
        f"|-------|---------|\n",
    ]
    for name, count in layer_counts.items():
        entry_lines.append(f"| `urn:bme:kg:{name}` | {count:,} |\n")
    entry_lines += [
        f"| **Total (TriG)** | **{total:,}** |\n",
        f"\n",
        f"2-hop included: {'yes' if include_2hop else 'no'}\n",
        f"\n",
        f"---\n",
        f"\n",
    ]

    changelog = bundle_dir / "CHANGELOG.md"
    if changelog.exists():
        existing = changelog.read_text(encoding="utf-8")
        # Insert after the header (first line)
        lines = existing.splitlines(keepends=True)
        header = lines[:2] if len(lines) >= 2 else lines
        rest   = lines[2:] if len(lines) >= 2 else []
        changelog.write_text("".join(header) + "\n" + "".join(entry_lines) + "".join(rest), encoding="utf-8")
    else:
        changelog.write_text(
            "# BME Knowledge Graph Bundle — Changelog\n\n" + "".join(entry_lines),
            encoding="utf-8",
        )
    logger.info(f"[8] CHANGELOG.md updated — version {version}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Regenerate the full bme_kg_bundle/")
    parser.add_argument(
        "--bundle-dir", type=Path,
        default=Path("../../assets/bme_kg_bundle"),
        help="Output bundle directory",
    )
    parser.add_argument(
        "--source-kg", type=Path,
        default=Path("../../assets/kg_triple_store_gt/knowledge_graph.ttl"),
        help="Source knowledge_graph.ttl",
    )
    parser.add_argument(
        "--ttl-gt", type=Path,
        default=Path("kg_expansion/ttl_gt"),
        help="Directory of 1-hop per-entity TTLs",
    )
    parser.add_argument(
        "--ttl-gt-2hop", type=Path,
        default=Path("kg_expansion/ttl_gt_2hop"),
        help="Directory of 2-hop per-entity TTLs",
    )
    parser.add_argument(
        "--no-2hop", action="store_true",
        help="Omit 2-hop delta from bundle",
    )
    parser.add_argument(
        "--version",
        default=datetime.now(timezone.utc).strftime("%Y.%m.%d"),
        help="Bundle version string (default: YYYY.MM.DD)",
    )
    args = parser.parse_args()

    bundle_dir  = args.bundle_dir
    patches_dir = bundle_dir / "patches"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"  BME KG Bundle — version {args.version}")
    logger.info(f"  Output: {bundle_dir}")
    logger.info(f"{'='*60}\n")

    # 1. Source KG
    copy_source(args.source_kg, bundle_dir)
    base_g = load(bundle_dir / "knowledge_graph.ttl")

    # 2. 1-hop expansion
    hop1_g = build_1hop(args.ttl_gt, bundle_dir)

    # 3. 2-hop delta
    if not args.no_2hop:
        hop2_g = build_2hop_delta(args.ttl_gt, args.ttl_gt_2hop, bundle_dir)
    else:
        hop2_g = bind(Graph())
        logger.info("[3] expansion_2hop_delta skipped (--no-2hop)")

    # 4. Patches
    patches = build_patches(base_g, hop1_g, patches_dir)

    # 5. Merged convenience files (copied verbatim from pipeline output)
    build_merged(bundle_dir, args.source_kg.parent, not args.no_2hop)

    # 6. TriG
    layer_counts = build_trig(bundle_dir, not args.no_2hop)

    # 7. VoID
    build_void(bundle_dir, layer_counts, args.version)

    # 8. Changelog
    update_changelog(bundle_dir, args.version, layer_counts, not args.no_2hop)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Bundle complete: {bundle_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
