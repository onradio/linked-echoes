#!/usr/bin/env python3
"""Stage 4 — the article deliverables (spec §6). Every number comes from data/results/.

    python report.py

Writes into `data/results/`:

    figures/retrieval_and_selection.pdf   the figure (PDF for LaTeX, PNG to eyeball)
    latex/table1_category.tex             Table 1
    latex/table2_type.tex                 Table 2
    latex/table_query_formulations.tex    the figure's companion table
    report.md                             everything above, rendered for reading

Nothing is computed here — `score.py` owns the arithmetic and this module only
formats it. If a number looks wrong, fix it in `score.py` and re-run both; never
edit the output files, which are regenerated on every run.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "data" / "results"
FIGURES = RESULTS / "figures"
LATEX = RESULTS / "latex"

# Categorical slots 1 and 2 of the reference palette, light mode, used unchanged.
# Colour follows the entity: a configuration keeps its hue in both panels.
SERIES = {"term_hint": "#2a78d6", "combine_name": "#eb6834"}
LABELS = {"term_hint": "term + hint", "combine_name": "combine + name"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"


def read(name: str) -> list:
    with open(RESULTS / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def reference() -> str:
    """The model slug the main tables report; cross-model results get their own."""
    from models import load_models, reference_model
    return reference_model(load_models())["slug"]


def is_reference(row: dict, ref: str) -> bool:
    """True for a row of the reference model, or of a system that uses no LLM."""
    return row.get("model", "-") in (ref, "-", "")


def num(row: dict, key: str):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def pct(value, places: int = 1) -> str:
    return "--" if value is None else f"{100 * value:.{places}f}"


def p_value(stats: dict, **match) -> float | None:
    for m in stats["mcnemar"]:
        if all(str(m.get(key)) == str(val) for key, val in match.items()):
            return m["p_value"]
    return None


def fmt_p(p) -> str:
    if p is None:
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def build_figure(retrieval: list, selection: list, config_query: dict) -> Path:
    """
    Two panels on one shared percentage axis: the retrieval ceiling, and what the
    selector achieves against it.

    Two panels rather than two y-scales on one plot — the measures share a unit,
    so a shared axis keeps the heights comparable and avoids inventing a
    correlation between them.
    """
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(7.4, 3.3), sharey=True, constrained_layout=True)

    for config, (strategy, query) in config_query.items():
        colour = SERIES[config]
        points = sorted(
            ((int(r["k"]), num(r, "hit_with_fallback")) for r in retrieval
             if r["index_strategy"] == strategy and r["query"] == query),
            key=lambda pair: pair[0])
        ks = [k for k, _ in points]
        values = [100 * v for _, v in points]
        ax_left.plot(ks, values, color=colour, linewidth=2, marker="o",
                     markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                     solid_capstyle="round", solid_joinstyle="round",
                     label=LABELS[config], zorder=3)
        # Only the endpoint is labelled: a value on every point is unreadable
        # and the axis already carries the rest.
        ax_left.annotate(f"{values[-1]:.0f}%", (ks[-1], values[-1]),
                         textcoords="offset points", xytext=(6, -3),
                         color=INK_MUTED, fontsize=8)

        cells = sorted(
            ((int(r["k"]), num(r, "accuracy")) for r in selection
             if r["configuration"] == config and r["system"] == "T1"),
            key=lambda pair: pair[0])
        ks2 = [k for k, _ in cells]
        values2 = [100 * v for _, v in cells]
        ax_right.plot(ks2, values2, color=colour, linewidth=2, marker="o",
                      markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                      solid_capstyle="round", solid_joinstyle="round",
                      label=LABELS[config], zorder=3)
        ax_right.annotate(f"{values2[-1]:.0f}%", (ks2[-1], values2[-1]),
                          textcoords="offset points", xytext=(6, -3),
                          color=INK_MUTED, fontsize=8)

    # Ticks only where a measurement exists: retrieval swept six depths, selection
    # four, and a tick at k=30 in panel (b) would imply a cell that was not run.
    retrieval_ks = sorted({int(r["k"]) for r in retrieval})
    selection_ks = sorted({int(r["k"]) for r in selection if r["system"] == "T1"})
    for ax, title, ticks in (
            (ax_left, "(a) retrieval ceiling: hit@$k$", retrieval_ks),
            (ax_right, "(b) selection accuracy: T1", selection_ks)):
        ax.set_title(title, fontsize=10, color=INK, pad=8, loc="left")
        ax.set_xlabel("$k$ (candidates retrieved)", fontsize=9, color=INK_MUTED)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 60)          # headroom for the endpoint labels
        ax.set_xticks(ticks)
        ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
        ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.set_facecolor(SURFACE)

    ax_left.set_ylabel("% of objects", fontsize=9, color=INK_MUTED)
    # One legend for the figure: both panels carry the same two series, so a
    # per-panel legend would imply the identity is local to a panel.
    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncols=2,
               loc="outside lower center", labelcolor=INK_MUTED)
    fig.patch.set_facecolor(SURFACE)

    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / "retrieval_and_selection.pdf"
    fig.savefig(pdf, facecolor=SURFACE)
    fig.savefig(FIGURES / "retrieval_and_selection.png", dpi=200,
                facecolor=SURFACE)
    plt.close(fig)
    return pdf


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def table1(category: list, stats: dict) -> tuple:
    """Table 1 — category classification. Three rows; C2-scoped goes in prose."""
    order = ["C0", "C1", "C2"]
    names = {"C0": "C0 majority class", "C1": "C1 nearest definition",
             "C2": "C2 LLM (ours)"}
    ref = reference()
    rows = []
    for system in order:
        row = next((r for r in category
                    if r["system"] == system and is_reference(r, ref)), None)
        if row is None:
            continue
        rows.append([
            names[system], pct(num(row, "accuracy")),
            pct(num(row, "accuracy_strict")), pct(num(row, "macro_precision")),
            pct(num(row, "macro_recall")), pct(num(row, "macro_f1")),
            "--" if system == "C2" else fmt_p(
                p_value(stats, comparison=f"C2 vs {system}")),
        ])
    header = ["System", "Acc.", "Strict", "Macro-P", "Macro-R", "Macro-F1",
              "McNemar $p$"]
    return header, rows


def table2(selection: list, stats: dict) -> tuple:
    """Table 2 — type classification: T0 once per configuration, T1 per k."""
    ref = reference()
    selection = [r for r in selection if is_reference(r, ref)]
    rows = []
    for config in ("term_hint", "combine_name"):
        t0 = next((r for r in selection
                   if r["configuration"] == config and r["system"] == "T0"), None)
        if t0:
            # hit@k is a property of a pool depth; T0 reads only rank 1, so the
            # column does not apply to it and is left blank rather than quoting
            # whichever k happened to be first in the file.
            rows.append([LABELS[config], "T0", "--", pct(num(t0, "accuracy")),
                         pct(num(t0, "term_precision")), pct(num(t0, "term_recall")),
                         pct(num(t0, "term_f1")), "--", "--", "--"])
        cells = sorted((r for r in selection
                        if r["configuration"] == config and r["system"] == "T1"),
                       key=lambda r: int(r["k"]))
        for r in cells:
            rows.append([
                LABELS[config], "T1", r["k"], pct(num(r, "accuracy")),
                pct(num(r, "term_precision")), pct(num(r, "term_recall")),
                pct(num(r, "term_f1")), pct(num(r, "abstention_accuracy")),
                pct(num(r, "hit_at_k")),
                fmt_p(p_value(stats, configuration=config, k=r["k"],
                              comparison="T1 vs T0")),
            ])
    header = ["Retrieval", "Sys.", "$k$", "Acc.", "Term P", "Term R", "F1",
              "Abst.", "hit@$k$", "$p$ vs T0"]
    return header, rows


QUERY_LABELS = {"description": "description",
                "name_description": "name + description",
                "hint_description": "hint + description"}
# Palette slots 1-3, light mode. Three is the documented all-pairs-safe prefix,
# which is why the query dimension gets the hues and the index dimension gets
# the panels rather than the other way round.
QUERY_COLOURS = {"description": "#2a78d6", "name_description": "#eb6834",
                 "hint_description": "#1baf7a"}
CARRIED_FORWARD = {("term", "hint_description"), ("combine", "name_description")}
# Y-range of the appendix grid figure; see the comment where it is applied.
GRID_YLIM = (40, 90)


def table_retrieval_grid(retrieval: list) -> tuple:
    """
    Every retrieval configuration at every depth — the evidence behind carrying
    exactly two of them into Experiment C.
    """
    ks = sorted({int(r["k"]) for r in retrieval})
    rows = []
    for strategy in ("term", "combine"):
        for query in ("description", "name_description", "hint_description"):
            cells = {int(r["k"]): r for r in retrieval
                     if r["index_strategy"] == strategy and r["query"] == query}
            rows.append(
                [strategy, QUERY_LABELS[query]]
                + [pct(num(cells[k], "hit_with_fallback")) if k in cells else "--"
                   for k in ks]
                + [f"{num(cells[ks[-1]], 'mrr'):.3f}" if ks[-1] in cells else "--",
                   "yes" if (strategy, query) in CARRIED_FORWARD else "--"])
    header = (["Index", "Query"] + [f"@{k}" for k in ks]
              + ["MRR@50", "Selection"])
    return header, rows


def build_grid_figure(retrieval: list) -> Path:
    """
    Small multiples over the full retrieval grid: one panel per index strategy,
    one line per query formulation.

    Six series in one panel would exceed what the palette clears on an all-pairs
    basis; splitting the index dimension into panels keeps the hue count at three.
    """
    ks = sorted({int(r["k"]) for r in retrieval})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True,
                             constrained_layout=True)

    for ax, strategy in zip(axes, ("term", "combine")):
        endpoints = {}
        for query, colour in QUERY_COLOURS.items():
            points = sorted(
                ((int(r["k"]), num(r, "hit_with_fallback")) for r in retrieval
                 if r["index_strategy"] == strategy and r["query"] == query),
                key=lambda pair: pair[0])
            xs = [k for k, _ in points]
            ys = [100 * v for _, v in points]
            ax.plot(xs, ys, color=colour, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2,
                    solid_capstyle="round", solid_joinstyle="round",
                    label=QUERY_LABELS[query], zorder=3)
            endpoints[query] = (xs[-1], ys[-1])

        # Label only the highest and lowest final value. Labelling all three
        # collides whenever two configurations converge — on the term index two
        # of them finish on the same number — and a label that overprints another
        # is worse than none. The grid table carries every value exactly.
        extremes = {min(endpoints.values(), key=lambda p: p[1]),
                    max(endpoints.values(), key=lambda p: p[1])}
        for x, y in extremes:
            ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                        xytext=(6, -3), color=INK_MUTED, fontsize=8)
        ax.set_title(f"({'ab'[strategy == 'combine']}) {strategy} index",
                     fontsize=10, color=INK, pad=8, loc="left")
        ax.set_xlabel("$k$ (candidates retrieved)", fontsize=9, color=INK_MUTED)
        # The y-axis starts at 40, not 0. Every configuration lies between 45%
        # and 83%, and the differences that matter here are a few points wide, so
        # a zero baseline spends half the panel on empty space and presses six
        # curves into a band too narrow to separate. Lines encode position rather
        # than length, so the truncation does not distort magnitude the way it
        # would on bars — but it is stated in the caption.
        ax.set_ylim(*GRID_YLIM)
        ax.set_yticks(range(GRID_YLIM[0], GRID_YLIM[1] + 1, 10))
        ax.set_xlim(0, 60)
        ax.set_xticks(ks)
        ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
        ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.set_facecolor(SURFACE)

    axes[0].set_ylabel("hit@$k$ (% of objects)", fontsize=9, color=INK_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncols=3,
               loc="outside lower center", labelcolor=INK_MUTED)
    fig.patch.set_facecolor(SURFACE)

    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / "retrieval_grid.pdf"
    fig.savefig(pdf, facecolor=SURFACE)
    fig.savefig(FIGURES / "retrieval_grid.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return pdf


def table_queries(retrieval: list, k: str = "10") -> tuple:
    """The figure's companion: hit@k for the three query formulations."""
    queries = ["description", "name_description", "hint_description"]
    rows = []
    for strategy in ("term", "combine"):
        row = [strategy]
        for query in queries:
            match = next((r for r in retrieval if r["index_strategy"] == strategy
                          and r["query"] == query and r["k"] == k), None)
            row.append(pct(num(match, "hit_with_fallback")) if match else "--")
        rows.append(row)
    header = ["Index", "description", "name + desc.", "hint + desc."]
    return header, rows


def to_latex(header: list, rows: list, caption: str, label: str) -> str:
    spec = "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def to_markdown(header: list, rows: list) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def main() -> int:
    category = read("category_cells.csv")
    selection = read("selection_cells.csv")
    retrieval = read("retrieval_hits.csv")
    stats = json.loads((RESULTS / "stats.json").read_text())

    # The two configurations Experiment C evaluated end to end.
    config_query = {"term_hint": ("term", "hint_description"),
                    "combine_name": ("combine", "name_description")}
    # The main figure reports the reference model; a multi-model comparison is a
    # separate deliverable rather than more lines on this one.
    ref = reference()
    pdf = build_figure(retrieval, [r for r in selection if is_reference(r, ref)],
                       config_query)
    grid_pdf = build_grid_figure(retrieval)

    h1, r1 = table1(category, stats)
    h2, r2 = table2(selection, stats)
    h3, r3 = table_queries(retrieval)
    h4, r4 = table_retrieval_grid(retrieval)

    n = int(float(category[0]["n_objects"]))
    models_present = sorted({r["model"] for r in category + selection
                             if r.get("model") not in ("-", "", None)})
    n_scored = int(float(retrieval[0]["n_scored"]))
    scoped = next((r for r in category if r["system"] == "C2-scoped"
                   and is_reference(r, ref)), None)
    p_scoped = next((m["p_value"] for m in stats["mcnemar"]
                     if str(m.get("a", "")).startswith("C2-scoped")
                     and m.get("model") == ref), None)

    LATEX.mkdir(parents=True, exist_ok=True)
    (LATEX / "table1_category.tex").write_text(to_latex(
        h1, r1,
        f"Category classification on the {n}-object gold standard. Accuracy credits "
        "a match against either gold category; strict accuracy against the primary "
        "only. Macro averages are over the four attested classes. C2's accuracy is "
        "at ceiling, so its bootstrap interval is degenerate and the sample cannot "
        "separate it from a perfect classifier.",
        "tab:category"))
    (LATEX / "table2_type.tex").write_text(to_latex(
        h2, r2,
        "Type classification. T0 is the top-ranked retrieved term and never "
        "abstains, so it is independent of $k$. Term precision is computed over "
        "everything asserted; recall excludes correct abstentions on the weak tier. "
        "Abstention accuracy is over the weak and NIL rows only ($n=15$) and is "
        "indicative.",
        "tab:type"))
    (LATEX / "table_query_formulations.tex").write_text(to_latex(
        h3, r3,
        "Retrieval hit@10 by query formulation, with the AAT fallback pool "
        "included.", "tab:queries"))
    (LATEX / "table3_retrieval_grid.tex").write_text(to_latex(
        h4, r4,
        "Type retrieval over the full grid: hit@$k$ for each index strategy and "
        "query formulation, with the AAT fallback pool included, on the "
        f"{n_scored} objects whose gold term is reachable. The two configurations "
        "marked in the last column are those carried into the selection "
        "experiment: the deployed pipeline (term + hint) and the configuration "
        "that retrieves best at depth (combine + name). MRR is at $k=50$.",
        "tab:retrieval_grid"))

    # Figure captions live with the figures, so the axis note cannot be lost
    # between generating the PDF and writing it into the article.
    (LATEX / "figure_captions.tex").write_text("\n\n".join([
        r"% Main figure",
        r"\caption{Type retrieval and type selection against the number of "
        r"candidates retrieved, $k$. Panel (a) is the share of the "
        f"{n_scored} objects with a reachable gold term whose term appears in the "
        r"top $k$, with the AAT fallback pool included; panel (b) is the accuracy "
        r"of the LLM selector (T1) over the same pools, averaged over three runs "
        r"at temperature 0. Both panels share one percentage axis. Retrieval "
        r"improves with depth in both configurations, but only \emph{combine + "
        r"name} converts that into selection accuracy: \emph{term + hint} gains "
        r"ceiling and loses accuracy.}",
        r"% Appendix figure",
        r"\caption{Type retrieval over the full grid: one panel per index "
        r"strategy, one line per query formulation. \textbf{The vertical axis "
        r"begins at " + str(GRID_YLIM[0]) + r"\%, not zero}, so that differences "
        r"of a few points between six configurations remain visible; exact values "
        r"for every cell are in Table~\ref{tab:retrieval_grid}. Only the highest "
        r"and lowest final value in each panel are labelled, since two "
        r"configurations on the term index finish on the same number.}",
    ]) + "\n")

    report = [
        "# Classification evaluation — deliverables",
        "",
        "Generated by `report.py` from `data/results/`. Every number is computed "
        "in `score.py`; nothing here is hand-edited.",
        "",
        "## Figure — retrieval ceiling and selection accuracy",
        "",
        f"![retrieval and selection](figures/{pdf.with_suffix('.png').name})",
        "",
        "Panel (a) is the share of objects whose accepted gold term appears in the "
        "top $k$, with the AAT fallback pool included; panel (b) is the accuracy of "
        "the LLM selector over the same pools. Both panels share one percentage "
        "axis.",
        "",
        "## Table 3 — the full retrieval grid",
        "",
        f"All six retrieval configurations at every depth, over the {n_scored} "
        "objects whose gold term is reachable. The last column marks the two "
        "carried into the selection experiment.",
        "",
        to_markdown(h4, r4),
        "",
        f"![retrieval grid](figures/{grid_pdf.with_suffix('.png').name})",
        "",
        "One panel per index strategy, one line per query formulation. The type "
        "hint dominates on the term index at every depth; on the combine index it "
        "leads early and is overtaken by name + description at $k=50$. "
        f"**The vertical axis begins at {GRID_YLIM[0]}%**, not zero — the table "
        "above carries every value exactly.",
        "",
        "### Companion — hit@10 by query formulation",
        "",
        to_markdown(h3, r3),
        "",
        "## Table 1 — category classification",
        "",
        to_markdown(h1, r1),
        "",
    ]
    if scoped is not None:
        report += [
            f"A richer prompt carrying the seven scope notes and eight worked "
            f"examples (`C2-scoped`) was evaluated and did not improve accuracy: "
            f"{pct(num(scoped, 'accuracy'))}% against C2's "
            f"{pct(num(next(r for r in category if r['system'] == 'C2'), 'accuracy'))}%, "
            f"one discordant object, McNemar $p = {p_scoped:.1f}$. The simpler "
            if p_scoped is not None else
            "with no significant difference. The simpler "
            "instruction is deployed.",
            "",
        ]
    report += [
        "## Table 2 — type classification",
        "",
        to_markdown(h2, r2),
        "",
        "## Paired comparisons",
        "",
        "McNemar on the modal outcome across runs. The depth comparisons are the "
        "ones the unpaired intervals cannot settle, since the cells share objects.",
        "",
        to_markdown(["Comparison", "Configuration", "$k$", "Discordant",
                     "Favours first", "Favours second", "$p$"],
                    [[m["comparison"], m["configuration"], m.get("k") or "--",
                      m["n_discordant"], m["a_correct_only"], m["b_correct_only"],
                      fmt_p(m["p_value"]).replace("$<$", "<")]
                     for m in stats["mcnemar"]
                     if "vs T0" not in m["comparison"]
                     and m.get("model", ref) in (ref, "-", None)]),
        "",
    ]
    (RESULTS / "report.md").write_text("\n".join(report))

    print(f"figure  -> {pdf.relative_to(HERE)}")
    for name in sorted(p.name for p in LATEX.glob("*.tex")):
        print(f"latex   -> {(LATEX / name).relative_to(HERE)}")
    print(f"report  -> {(RESULTS / 'report.md').relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
