"""
kg_stats.py — Structural snapshot and graph statistics for the BME knowledge graph.

Usage:
    python kg_stats.py --kg path/to/knowledge_graph.ttl [--report path/to/report.json]

Produces:
    - Console summary: node/edge counts, type distribution, predicate frequency,
      connectivity stats, isolated nodes, hub nodes
    - Optional JSON report for tracking over time
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdflib import Graph, URIRef, Literal, RDF, RDFS, Namespace
from rdflib.namespace import SKOS

try:
    import networkx as nx
except ImportError:
    print("networkx is required: pip install networkx", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

CRM  = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
SCHO = Namespace("http://schema.org/")
WD   = Namespace("http://www.wikidata.org/entity/")
BME  = "http://radio.liv.ac.uk/bme/"

# Predicates to use as edges in the graph (meaningful relationships)
EDGE_PREDICATES = {
    SCHO.about,
    SCHO.material,
    SCHO.genre,
    SCHO.category,
    SCHO.location,
    SCHO.additionalType,
    SCHO.containedInPlace,
}

# Human-readable labels for types
TYPE_LABELS = {
    str(CRM["E22_Human-Made_Object"]): "HumanMadeObject",
    str(SCHO.Person):                  "Person",
    str(SCHO.MusicAlbum):              "MusicAlbum",
    str(SCHO.MusicComposition):        "MusicComposition",
    str(SCHO.Event):                   "Event",
    str(SCHO.Place):                   "Place",
    str(SCHO.Country):                 "Country",
    str(SCHO.EventVenue):              "EventVenue",
    str(SCHO.DefinedTerm):             "DefinedTerm",
    str(SCHO.MusicGroup):              "MusicGroup",
    str(RDFS.Class):                   "Class",
    str(SKOS.Concept):                 "Concept",
}


def type_label(uri: str) -> str:
    return TYPE_LABELS.get(uri, uri.split("/")[-1] or uri.split("#")[-1])


def node_namespace(uri: str) -> str:
    if uri.startswith(BME):
        tail = uri[len(BME):]
        seg = tail.split("/")[0]
        return f"bme:{seg}"
    if uri.startswith("http://www.wikidata.org/entity/"):
        return "wd:"
    if uri.startswith("http://vocab.getty.edu/aat/"):
        return "aat:"
    if uri.startswith("http://thesaurus.europeanafashion.eu/thesaurus/"):
        return "euf:"
    if uri.startswith("http://www.mimo-db.eu/InstrumentsKeywords/"):
        return "mimo:"
    if uri.startswith("http://pbcore.org/"):
        return "pbcore:"
    if uri.startswith("http://schema.org/"):
        return "scho:"
    return "other:"


# ---------------------------------------------------------------------------
# RDF graph loading
# ---------------------------------------------------------------------------

def load_rdf(path: Path) -> Graph:
    g = Graph()
    g.parse(str(path), format="turtle")
    return g


# ---------------------------------------------------------------------------
# Stats extraction
# ---------------------------------------------------------------------------

def extract_stats(rdf: Graph) -> dict:
    stats: dict = {}

    # ------------------------------------------------------------------
    # 1. Triple counts
    # ------------------------------------------------------------------
    stats["total_triples"] = len(rdf)

    subjects   = set(rdf.subjects())
    predicates = set(rdf.predicates())
    objects    = set(rdf.objects())

    uri_subjects = {s for s in subjects if isinstance(s, URIRef)}
    uri_objects  = {o for o in objects  if isinstance(o, URIRef)}
    all_uris     = uri_subjects | uri_objects

    stats["unique_subjects"]   = len(subjects)
    stats["unique_predicates"] = len(predicates)
    stats["unique_objects"]    = len(objects)
    stats["unique_uri_nodes"]  = len(all_uris)
    stats["literal_objects"]   = sum(1 for o in objects if isinstance(o, Literal))

    # ------------------------------------------------------------------
    # 2. Type distribution
    # ------------------------------------------------------------------
    type_counts: Counter = Counter()
    node_types: dict[str, set[str]] = defaultdict(set)
    for s, _, t in rdf.triples((None, RDF.type, None)):
        if isinstance(s, URIRef) and isinstance(t, URIRef):
            label = type_label(str(t))
            type_counts[label] += 1
            node_types[str(s)].add(str(t))

    stats["type_distribution"] = dict(type_counts.most_common())

    # Nodes with multiple types
    multi_typed = {uri: list(types) for uri, types in node_types.items() if len(types) > 1}
    stats["multi_typed_node_count"] = len(multi_typed)

    # ------------------------------------------------------------------
    # 3. Predicate frequency
    # ------------------------------------------------------------------
    pred_counts: Counter = Counter()
    for _, p, _ in rdf:
        pred_counts[str(p)] += 1

    # Shorten predicate URIs for readability
    pred_short: Counter = Counter()
    for uri, count in pred_counts.items():
        short = uri.replace("http://schema.org/", "scho:")\
                   .replace("http://www.w3.org/2000/01/rdf-schema#", "rdfs:")\
                   .replace("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:")\
                   .replace("http://www.w3.org/2004/02/skos/core#", "skos:")\
                   .replace("http://www.cidoc-crm.org/cidoc-crm/", "crm:")\
                   .replace("http://www.w3.org/ns/prov#", "prov:")
        pred_short[short] = count

    stats["predicate_frequency"] = dict(pred_short.most_common())

    # ------------------------------------------------------------------
    # 4. Namespace distribution of URI nodes
    # ------------------------------------------------------------------
    ns_counts: Counter = Counter()
    for uri in all_uris:
        ns_counts[node_namespace(str(uri))] += 1

    stats["namespace_distribution"] = dict(ns_counts.most_common())

    # ------------------------------------------------------------------
    # 5. NetworkX graph — directed, URI nodes only, edge predicates only
    # ------------------------------------------------------------------
    G = nx.MultiDiGraph()

    for s, p, o in rdf:
        if p not in EDGE_PREDICATES:
            continue
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue
        G.add_node(str(s))
        G.add_node(str(o))
        G.add_edge(str(s), str(o), predicate=str(p))

    # Also add all URI nodes that appear as subjects (even if no edges)
    for s in uri_subjects:
        G.add_node(str(s))

    stats["graph_nodes"]    = G.number_of_nodes()
    stats["graph_edges"]    = G.number_of_edges()

    # ------------------------------------------------------------------
    # 6. Connectivity
    # ------------------------------------------------------------------
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    stats["connected_components"] = len(components)
    stats["largest_component_size"] = max(len(c) for c in components) if components else 0
    stats["isolated_nodes"] = sum(1 for n in G.nodes() if G.degree(n) == 0)

    # ------------------------------------------------------------------
    # 7. Degree stats
    # ------------------------------------------------------------------
    in_degrees  = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    def degree_summary(d: dict) -> dict:
        vals = list(d.values())
        if not vals:
            return {}
        vals_sorted = sorted(vals, reverse=True)
        return {
            "max":    vals_sorted[0],
            "mean":   round(sum(vals) / len(vals), 2),
            "median": vals_sorted[len(vals) // 2],
        }

    stats["in_degree_summary"]  = degree_summary(in_degrees)
    stats["out_degree_summary"] = degree_summary(out_degrees)

    # ------------------------------------------------------------------
    # 8. Hub nodes — top 10 by in-degree and out-degree
    # ------------------------------------------------------------------
    def top_nodes(degree_dict: dict, n: int = 10) -> list[dict]:
        top = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:n]
        return [
            {
                "uri": uri,
                "namespace": node_namespace(uri),
                "types": [type_label(t) for t in node_types.get(uri, [])],
                "degree": deg,
            }
            for uri, deg in top if deg > 0
        ]

    stats["top_in_degree_nodes"]  = top_nodes(in_degrees)
    stats["top_out_degree_nodes"] = top_nodes(out_degrees)

    # ------------------------------------------------------------------
    # 9. Centrality measures
    # ------------------------------------------------------------------
    simple_undirected = nx.Graph(undirected)  # collapse multi-edges for centrality

    stats["graph_density"] = round(nx.density(simple_undirected), 6)

    stats["average_clustering"] = round(nx.average_clustering(simple_undirected), 4)

    # Degree centrality
    deg_centrality = nx.degree_centrality(simple_undirected)
    top_deg_central = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    stats["top_degree_centrality"] = [
        {"uri": uri, "namespace": node_namespace(uri),
         "types": [type_label(t) for t in node_types.get(uri, [])],
         "centrality": round(score, 4)}
        for uri, score in top_deg_central if score > 0
    ]

    # Betweenness centrality (undirected, normalised)
    between_centrality = nx.betweenness_centrality(simple_undirected, normalized=True)
    top_between = sorted(between_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    stats["top_betweenness_centrality"] = [
        {"uri": uri, "namespace": node_namespace(uri),
         "types": [type_label(t) for t in node_types.get(uri, [])],
         "centrality": round(score, 4)}
        for uri, score in top_between if score > 0
    ]

    # PageRank
    try:
        pagerank = nx.pagerank(G, alpha=0.85)
        top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        stats["top_pagerank"] = [
            {"uri": uri, "namespace": node_namespace(uri),
             "types": [type_label(t) for t in node_types.get(uri, [])],
             "score": round(score, 4)}
            for uri, score in top_pr
        ]
    except nx.PowerIterationFailedConvergence:
        stats["top_pagerank"] = []

    # ------------------------------------------------------------------
    # 11. Wikidata entity summary
    # ------------------------------------------------------------------
    wd_entities = [u for u in all_uris if str(u).startswith("http://www.wikidata.org/entity/")]
    wd_typed    = [u for u in wd_entities if node_types.get(str(u))]
    stats["wikidata_entities_total"]  = len(wd_entities)
    stats["wikidata_entities_typed"]  = len(wd_typed)
    stats["wikidata_entities_untyped"] = len(wd_entities) - len(wd_typed)

    # ------------------------------------------------------------------
    # 10. scho:about link stats
    # ------------------------------------------------------------------
    about_counts: Counter = Counter()
    for s, _, o in rdf.triples((None, SCHO.about, None)):
        if isinstance(o, URIRef):
            about_counts[node_namespace(str(o))] += 1

    stats["about_link_targets_by_namespace"] = dict(about_counts.most_common())

    # ------------------------------------------------------------------
    # 11. HumanMadeObject completeness
    # ------------------------------------------------------------------
    hmo_type = CRM["E22_Human-Made_Object"]
    hmo_subjects = list(rdf.subjects(RDF.type, hmo_type))
    hmo_count = len(hmo_subjects)

    optional_preds = [SCHO.genre, SCHO.material, SCHO.color, SCHO.about]
    hmo_coverage: dict[str, int] = {}
    for pred in optional_preds:
        short_pred = str(pred).replace("http://schema.org/", "scho:")
        hmo_coverage[short_pred] = sum(
            1 for s in hmo_subjects if any(True for _ in rdf.objects(s, pred))
        )

    stats["humanmadeobject_count"] = hmo_count
    stats["humanmadeobject_field_coverage"] = {
        k: {"count": v, "pct": round(100 * v / hmo_count, 1) if hmo_count else 0}
        for k, v in hmo_coverage.items()
    }

    return stats


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_stats(stats: dict):
    W = 60

    def section(title: str):
        print(f"\n{'─' * W}")
        print(f"  {title}")
        print(f"{'─' * W}")

    def row(label: str, value):
        print(f"  {label:<38} {value}")

    print("=" * W)
    print("  KG STRUCTURAL SNAPSHOT")
    print("=" * W)

    section("Triples")
    row("Total triples",       stats["total_triples"])
    row("Unique subjects",     stats["unique_subjects"])
    row("Unique predicates",   stats["unique_predicates"])
    row("Unique URI nodes",    stats["unique_uri_nodes"])
    row("Literal objects",     stats["literal_objects"])

    section("Type Distribution")
    for t, count in stats["type_distribution"].items():
        row(t, count)

    section("Predicate Frequency")
    for p, count in list(stats["predicate_frequency"].items())[:20]:
        row(p, count)
    remaining = len(stats["predicate_frequency"]) - 20
    if remaining > 0:
        print(f"  ... and {remaining} more predicates")

    section("Namespace Distribution (URI nodes)")
    for ns, count in stats["namespace_distribution"].items():
        row(ns, count)

    section("Graph Structure (relationship edges only)")
    row("Nodes",                    stats["graph_nodes"])
    row("Edges",                    stats["graph_edges"])
    row("Connected components",     stats["connected_components"])
    row("Largest component size",   stats["largest_component_size"])
    row("Isolated nodes",           stats["isolated_nodes"])
    row("Multi-typed nodes",        stats["multi_typed_node_count"])

    section("Degree Summary")
    ind = stats["in_degree_summary"]
    outd = stats["out_degree_summary"]
    if ind:
        row("In-degree  max / mean / median",
            f"{ind['max']} / {ind['mean']} / {ind['median']}")
    if outd:
        row("Out-degree max / mean / median",
            f"{outd['max']} / {outd['mean']} / {outd['median']}")

    section("Centrality")
    row("Graph density",              stats["graph_density"])
    row("Average clustering coeff.",  stats["average_clustering"])

    section("Top 10 by Degree Centrality")
    for node in stats["top_degree_centrality"]:
        types = ", ".join(node["types"]) or "untyped"
        short_uri = node["uri"].replace("http://radio.liv.ac.uk/bme/", "bme:")\
                               .replace("http://www.wikidata.org/entity/", "wd:")
        print(f"  [{node['centrality']:.4f}]  {short_uri:<45}  {types}")

    section("Top 10 by Betweenness Centrality (bridge nodes)")
    for node in stats["top_betweenness_centrality"]:
        types = ", ".join(node["types"]) or "untyped"
        short_uri = node["uri"].replace("http://radio.liv.ac.uk/bme/", "bme:")\
                               .replace("http://www.wikidata.org/entity/", "wd:")
        print(f"  [{node['centrality']:.4f}]  {short_uri:<45}  {types}")

    section("Top 10 by PageRank (influential nodes)")
    for node in stats["top_pagerank"]:
        types = ", ".join(node["types"]) or "untyped"
        short_uri = node["uri"].replace("http://radio.liv.ac.uk/bme/", "bme:")\
                               .replace("http://www.wikidata.org/entity/", "wd:")
        print(f"  [{node['score']:.4f}]  {short_uri:<45}  {types}")

    section("Top 10 Nodes by In-Degree (most referenced)")
    for node in stats["top_in_degree_nodes"]:
        types = ", ".join(node["types"]) or "untyped"
        short_uri = node["uri"].replace("http://radio.liv.ac.uk/bme/", "bme:")\
                               .replace("http://www.wikidata.org/entity/", "wd:")
        print(f"  [{node['degree']:>3}]  {short_uri:<45}  {types}")

    section("Top 10 Nodes by Out-Degree (most connections out)")
    for node in stats["top_out_degree_nodes"]:
        types = ", ".join(node["types"]) or "untyped"
        short_uri = node["uri"].replace("http://radio.liv.ac.uk/bme/", "bme:")\
                               .replace("http://www.wikidata.org/entity/", "wd:")
        print(f"  [{node['degree']:>3}]  {short_uri:<45}  {types}")

    section("Wikidata Entities")
    row("Total wd: entities",   stats["wikidata_entities_total"])
    row("Typed",                stats["wikidata_entities_typed"])
    row("Untyped",              stats["wikidata_entities_untyped"])

    section("scho:about Link Targets by Namespace")
    for ns, count in stats["about_link_targets_by_namespace"].items():
        row(ns, count)

    section("HumanMadeObject Field Coverage")
    row("Total HumanMadeObjects", stats["humanmadeobject_count"])
    for field, cov in stats["humanmadeobject_field_coverage"].items():
        row(field, f"{cov['count']}  ({cov['pct']}%)")

    print(f"\n{'=' * W}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Structural snapshot and stats for a BME KG TTL file")
    parser.add_argument("--kg", required=True, help="Path to TTL file")
    parser.add_argument("--report", default=None, help="Write JSON stats report to this path")
    args = parser.parse_args()

    ttl_path = Path(args.kg)
    if not ttl_path.exists():
        print(f"Error: file not found: {ttl_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {ttl_path} ...", flush=True)
    rdf = load_rdf(ttl_path)
    print(f"  {len(rdf)} triples loaded", flush=True)

    print("Computing stats ...", flush=True)
    stats = extract_stats(rdf)

    print_stats(stats)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nJSON report written to {report_path}")
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_stats(stats)
        txt_path = report_path.with_suffix(".txt")
        txt_path.write_text(buf.getvalue(), encoding="utf-8")
        print(f"Text report written to {txt_path}")


if __name__ == "__main__":
    main()
