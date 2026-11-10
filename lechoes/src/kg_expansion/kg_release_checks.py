"""
kg_release_checks.py

Release gate for the published knowledge graph. Every check encodes an invariant
that has been violated at least once, so a failure here means a real regression
rather than a style preference.

Usage:
    python kg_expansion/kg_release_checks.py \
        --kg path/to/knowledge_graph_full_2hop.ttl \
        [--previous path/to/releases/kg-v<previous>.json]

Exit code 0 = releasable, 1 = blocked.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

SCHO = "http://schema.org/"
NAME = URIRef(SCHO + "name")
HMO = URIRef("http://www.cidoc-crm.org/cidoc-crm/E22_Human-Made_Object")
YEAR_RE = re.compile(r"^\d{4}$")

RETYPED_PREDICATES = ("temporalCoverage", "duration", "numTracks",
                      "maximumAttendeeCapacity")


class Result:
    def __init__(self):
        self.checks = []
        self.stats = {}

    def add(self, name, passed, detail, blocking=True):
        self.checks.append({"check": name, "passed": bool(passed),
                            "detail": detail, "blocking": blocking})

    @property
    def blocked(self):
        return any(c["blocking"] and not c["passed"] for c in self.checks)


def entity_nodes(g):
    subs = {s for s in g.subjects() if isinstance(s, URIRef)}
    objs = {o for o in g.objects() if isinstance(o, URIRef)}
    classes = {o for _, _, o in g.triples((None, RDF.type, None))}
    return (subs | objs) - set(g.predicates()) - classes


def is_named(g, x):
    return (x, RDFS.label, None) in g or (x, NAME, None) in g


def measure(g):
    hmos = set(g.subjects(RDF.type, HMO))
    nodes = entity_nodes(g)
    by_type = {}
    for _, _, t in g.triples((None, RDF.type, None)):
        key = str(t).rsplit("/", 1)[-1].split("#")[-1]
        by_type[key] = by_type.get(key, 0) + 1
    return {
        "triples": len(g),
        "objects": len(hmos),
        "entities": len(nodes),
        "unnamed_nodes": sum(1 for x in nodes if not is_named(g, x)),
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
    }


def run_checks(g, previous=None, drift_pct=5.0):
    r = Result()
    hmos = set(g.subjects(RDF.type, HMO))
    nodes = entity_nodes(g)
    r.stats = measure(g)

    unnamed = [x for x in nodes if not is_named(g, x)]
    r.add("every entity is nameable", not unnamed,
          f"{len(unnamed)} unnamed" + (f" e.g. {str(unnamed[0])}" if unnamed else ""))

    missing_label = [h for h in hmos if (h, RDFS.label, None) not in g]
    r.add("every object has rdfs:label", not missing_label,
          f"{len(missing_label)}/{len(hmos)} objects missing rdfs:label")

    missing_name = [h for h in hmos if (h, NAME, None) not in g]
    r.add("every object has schema:name", not missing_name,
          f"{len(missing_name)}/{len(hmos)} objects missing schema:name")

    divergent = [h for h in hmos
                 if (h, RDFS.label, None) in g and (h, NAME, None) in g
                 and {str(o) for o in g.objects(h, RDFS.label)}
                 != {str(o) for o in g.objects(h, NAME)}]
    r.add("rdfs:label matches schema:name", not divergent,
          f"{len(divergent)} objects where the two have drifted apart")

    blank = [(s, p) for s, p, o in g
             if isinstance(o, Literal) and not str(o).strip()]
    r.add("no blank literals", not blank, f"{len(blank)} empty-string literals")

    stale = []
    for pred in RETYPED_PREDICATES:
        for s, _, o in g.triples((None, URIRef(SCHO + pred), None)):
            if not isinstance(o, Literal) or o.datatype or o.language:
                continue
            v = str(o).strip()
            if pred == "temporalCoverage" and not YEAR_RE.match(v):
                continue  # decade strings are deliberately left plain
            if pred != "temporalCoverage" and not v.isdigit():
                continue
            stale.append((s, pred, v))
    r.add("numeric and date fields are typed", not stale,
          f"{len(stale)} plain-string values that should carry a datatype")

    dup = sum(1 for s, p, o in g
              if isinstance(o, Literal) and o.datatype
              and any(isinstance(x, Literal) and x.datatype is None
                      and x.language is None and str(x).strip() == str(o).strip()
                      for x in g.objects(s, p)))
    r.add("no typed/plain duplicates", not dup,
          f"{dup} values present both typed and untyped")

    orphan_hmos = [h for h in hmos if (h, URIRef(SCHO + "location"), None) not in g]
    r.add("every object has a location", not orphan_hmos,
          f"{len(orphan_hmos)} objects with no schema:location")

    if previous:
        before = previous.get("stats", {}).get("triples")
        if before:
            drift = abs(len(g) - before) / before * 100
            r.add(f"triple count within {drift_pct}% of last release",
                  drift <= drift_pct,
                  f"{before:,} -> {len(g):,} ({drift:+.1f}%)")
        prev_objs = previous.get("stats", {}).get("objects")
        if prev_objs:
            r.add("object count did not shrink", len(hmos) >= prev_objs,
                  f"{prev_objs} -> {len(hmos)}")
    return r


def run_shacl_if_available(g, shapes_path):
    try:
        from shacl_validator import run_shacl
    except ImportError:
        return None, "pyshacl not installed — SHACL skipped"
    try:
        conforms, violations, warnings = run_shacl(g, shapes_path)
        return bool(conforms), f"{violations} violations, {warnings} warnings"
    except Exception as e:
        return None, f"SHACL run failed: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", type=Path, required=True)
    ap.add_argument("--previous", type=Path,
                    help="previous releases/<version>.json for drift comparison")
    ap.add_argument("--drift-pct", type=float, default=5.0)
    ap.add_argument("--shapes", type=Path,
                    default=Path(__file__).parent / "shacl_shapes.ttl")
    ap.add_argument("--json", type=Path, help="write machine-readable result here")
    ap.add_argument("--skip-shacl", action="store_true")
    ap.add_argument("--require-shacl", action="store_true",
                    help="treat an unavailable validator as a failure")
    args = ap.parse_args()

    g = Graph()
    g.parse(args.kg, format="turtle")
    previous = json.loads(args.previous.read_text()) if args.previous and args.previous.exists() else None

    r = run_checks(g, previous, args.drift_pct)

    if not args.skip_shacl:
        conforms, detail = run_shacl_if_available(g, args.shapes)
        if conforms is None:
            # could not run — report as a warning, never as a pass, so an
            # absent validator cannot be mistaken for a clean bill of health
            r.add("SHACL", False, detail, blocking=args.require_shacl)
        else:
            r.add("SHACL conforms", conforms, detail)

    width = max(len(c["check"]) for c in r.checks)
    print(f"\nRelease gate — {args.kg.name}\n")
    for c in r.checks:
        mark = "PASS" if c["passed"] else ("FAIL" if c["blocking"] else "WARN")
        print(f"  [{mark}] {c['check']:<{width}}  {c['detail']}")
    s = r.stats
    print(f"\n  {s['triples']:,} triples | {s['objects']} objects | "
          f"{s['entities']:,} entities | {s['unnamed_nodes']} unnamed")

    if args.json:
        args.json.write_text(json.dumps(
            {"kg": str(args.kg), "stats": r.stats, "checks": r.checks}, indent=2))

    if r.blocked:
        print("\nBLOCKED — fix the failures above before releasing.\n")
        return 1
    print("\nReleasable.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
