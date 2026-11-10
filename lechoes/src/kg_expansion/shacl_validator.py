"""
shacl_validator.py

Utility for running SHACL validation against the BME knowledge graph.
Used by aggregate_expansion.py and kg_expander.py.

Usage (standalone):
    python kg_expansion/shacl_validator.py \
        --graph ../../assets/kg_triple_store_gt/knowledge_graph_full_2hop.ttl \
        --shapes kg_expansion/shacl_shapes.ttl \
        --report kg_expansion/reports/shacl_report.csv
"""

import argparse
import csv
import logging
from pathlib import Path

from pyshacl import validate
from rdflib import Graph

logger = logging.getLogger(__name__)

SHAPES_DEFAULT = Path(__file__).parent / "shacl_shapes.ttl"


def run_shacl(
    data_graph: Graph,
    shapes_path: Path = SHAPES_DEFAULT,
    abort_on_violation: bool = False,
    report_path: Path | None = None,
) -> tuple[bool, int, int]:
    """
    Run SHACL validation on data_graph.

    Returns:
        (conforms, violation_count, warning_count)

    If abort_on_violation=True, raises RuntimeError on any sh:Violation.
    If report_path is given, writes a CSV of all results.
    """
    shapes_graph = Graph()
    shapes_graph.parse(str(shapes_path), format="turtle")

    conforms, results_graph, results_text = validate(
        data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        abort_on_first=False,
        meta_shacl=False,
        advanced=False,
        js=False,
        debug=False,
    )

    # Parse results graph for structured output
    from rdflib.namespace import RDF, SH
    SH_NS = "http://www.w3.org/ns/shacl#"

    violations = []
    warnings = []

    for result in results_graph.subjects(RDF.type, SH.ValidationResult):
        severity  = str(results_graph.value(result, SH.resultSeverity) or "")
        focus     = str(results_graph.value(result, SH.focusNode) or "")
        path      = str(results_graph.value(result, SH.resultPath) or "")
        message   = str(results_graph.value(result, SH.resultMessage) or "")
        src_shape = str(results_graph.value(result, SH.sourceShape) or "")
        src_constraint = str(results_graph.value(result, SH.sourceConstraintComponent) or "")

        row = {
            "severity": severity.split("#")[-1] if "#" in severity else severity,
            "focus_node": focus.split("/")[-1] if "/" in focus else focus,
            "focus_uri": focus,
            "path": path.split("/")[-1] if "/" in path else path,
            "message": message,
            "source_shape": src_shape.split("/")[-1] if "/" in src_shape else src_shape,
            "constraint": src_constraint.split("#")[-1] if "#" in src_constraint else src_constraint,
        }

        if "Violation" in severity:
            violations.append(row)
        else:
            warnings.append(row)

    all_results = violations + warnings

    # Log summary
    logger.info(f"[shacl] conforms={conforms} violations={len(violations)} warnings={len(warnings)}")

    if violations:
        logger.warning(f"[shacl] {len(violations)} violation(s):")
        for r in violations[:10]:
            logger.warning(f"  [{r['severity']}] {r['focus_node']} — {r['message']}")
        if len(violations) > 10:
            logger.warning(f"  ... and {len(violations) - 10} more")

    if warnings:
        logger.info(f"[shacl] {len(warnings)} warning(s) (first 5):")
        for r in warnings[:5]:
            logger.info(f"  [{r['severity']}] {r['focus_node']} — {r['message']}")

    # Write report CSV
    if report_path and all_results:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["severity", "focus_node", "focus_uri", "path", "message", "source_shape", "constraint"])
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"[shacl] report written → {report_path}")

    if abort_on_violation and violations:
        raise RuntimeError(
            f"SHACL validation failed: {len(violations)} violation(s). "
            f"See {report_path or 'log output'} for details."
        )

    return conforms, len(violations), len(warnings)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph",   type=Path, required=True, help="TTL file to validate")
    parser.add_argument("--shapes",  type=Path, default=SHAPES_DEFAULT)
    parser.add_argument("--report",  type=Path, default=None, help="Output CSV report path")
    parser.add_argument("--strict",  action="store_true", help="Exit non-zero on violations")
    args = parser.parse_args()

    g = Graph()
    g.parse(str(args.graph), format="turtle")
    print(f"[shacl] Loaded {len(g):,} triples from {args.graph}")

    conforms, n_violations, n_warnings = run_shacl(
        g,
        shapes_path=args.shapes,
        abort_on_violation=False,
        report_path=args.report,
    )

    print(f"[shacl] conforms={conforms}  violations={n_violations}  warnings={n_warnings}")

    if args.strict and n_violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
