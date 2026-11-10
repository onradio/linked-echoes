#!/usr/bin/env python3
""" To run the script python3 eval_folder.py \
  --gt inputs/item_descriptions_with_captions_all_but_examples.csv \
  --pred_dir output_pro \
  --out output_pro/individual_eval/per_row_metrics.csv"""

import argparse, json, math, re, sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import numpy as np

# ---- Fields to compare ----
SINGLE_FIELDS = ["base_type", "specific_type", "item_title"]
LIST_FIELDS = [
    "music_genres", "music_works", "music_albums", "music_artists",
    "events", "countries", "places", "venues", "years", "colours", "materials",
]
NUM_FIELDS = ["cardinality"]  # numeric but compared as strings after normalization
ALL_FIELDS = SINGLE_FIELDS + LIST_FIELDS + NUM_FIELDS

# ---- Utils ----
def norm(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        if math.isnan(val):
            return ""
        return str(int(val)) if val.is_integer() else str(val)
    return re.sub(r"\s+", " ", str(val)).strip().lower()

def split_gt(val: Any) -> List[str]:
    s = norm(val)
    if not s:
        return []
    if " | " in s:
        parts = [p.strip() for p in s.split("|")]
    elif "," in s and not re.search(r"\d{4},\s*\d{4}", s):
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [s]
    return [p for p in parts if p]

def atoms_from_gt(row: Dict[str, Any]) -> List[str]:
    atoms: List[str] = []
    for f in SINGLE_FIELDS:
        v = norm(row.get(f))
        if v:
            atoms.append(f"{f}:{v}")
    for f in LIST_FIELDS:
        for v in split_gt(row.get(f)):
            atoms.append(f"{f}:{v}")
    for f in NUM_FIELDS:
        v = norm(row.get(f))
        if v != "":
            atoms.append(f"{f}:{v}")
    return atoms

def _iterable_but_not_str(x: Any) -> bool:
    if isinstance(x, (str, bytes)):
        return False
    try:
        iter(x)
        return True
    except Exception:
        return False

def atoms_from_pred(p: Dict[str, Any]) -> List[str]:
    atoms: List[str] = []
    for f in SINGLE_FIELDS:
        v = norm(p.get(f))
        if v:
            atoms.append(f"{f}:{v}")
    for f in LIST_FIELDS:
        vals = p.get(f)
        if vals is None:
            continue
        if _iterable_but_not_str(vals):
            for v in vals:
                vv = norm(v)
                if vv:
                    atoms.append(f"{f}:{vv}")
        else:
            vv = norm(vals)
            if vv:
                atoms.append(f"{f}:{vv}")
    for f in NUM_FIELDS:
        v = norm(p.get(f))
        if v != "":
            atoms.append(f"{f}:{v}")
    return atoms

def pred_field_coverage_and_atoms(p: Dict[str, Any]) -> Tuple[int, float, int]:
    """
    Returns:
      nonempty_fields: count of fields (from ALL_FIELDS) that are non-empty
      coverage_ratio: nonempty_fields / len(ALL_FIELDS)
      atom_count: total predicted atomic values (each list item counts)
    """
    nonempty = 0
    for f in ALL_FIELDS:
        val = p.get(f)
        if f in LIST_FIELDS:
            if _iterable_but_not_str(val):
                if any(norm(v) for v in val):
                    nonempty += 1
            else:
                if norm(val):
                    nonempty += 1
        else:
            if norm(val):
                nonempty += 1
    coverage = nonempty / len(ALL_FIELDS)
    atoms = len(atoms_from_pred(p))
    return nonempty, coverage, atoms

def prf1(truth_atoms: List[str], pred_atoms: List[str]) -> Tuple[int, int, int, float, float, float]:
    ts, ps = set(truth_atoms), set(pred_atoms)
    tp = len(ts & ps)
    pp = len(ps)
    tt = len(ts)
    precision = tp / pp if pp else 0.0
    recall = tp / tt if tt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return tp, pp, tt, precision, recall, f1

# ---- JSON loaders with auto-detection ----
def load_preds_any(json_path: Path) -> Tuple[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if all(isinstance(x, list) for x in data):
            return "case_by_case", data
        if all(isinstance(x, dict) for x in data):
            return "row_by_row", data
        if data and isinstance(data[0], list):
            return "case_by_case", data
        if data and isinstance(data[0], dict):
            return "row_by_row", data
    raise ValueError(f"{json_path.name}: unsupported top-level JSON format")

# ---- Evaluators ----
def evaluate_case_by_case(gt_df: pd.DataFrame, preds: List[List[Dict[str, Any]]], source_name: str) -> pd.DataFrame:
    grouped = {int(c): df.reset_index(drop=True)
               for c, df in gt_df.sort_values(["case"]).groupby("case")}
    all_cases = set(grouped.keys())
    pred_cases = set(range(1, len(preds) + 1))
    common = sorted(all_cases & pred_cases)

    rows = []
    for case in common:
        gt_items = grouped[case]
        pred_items = preds[case - 1]
        n = min(len(gt_items), len(pred_items))
        for i in range(n):
            g = gt_items.iloc[i].to_dict()
            p = pred_items[i]
            tp, pp, tt, precision, recall, f1 = prf1(atoms_from_gt(g), atoms_from_pred(p))
            nonempty_fields, coverage_ratio, atom_count = pred_field_coverage_and_atoms(p)
            rows.append({
                "file": source_name,
                "json_mode": "case_by_case",
                "set": g.get("set"),
                "case": int(case),
                "row_in_case": i + 1,
                "gt_item_title": g.get("item_title"),
                "pred_item_title": p.get("item_title"),
                "pred_count": pp,
                "truth_count": tt,
                "correct": tp,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "pred_nonempty_fields": nonempty_fields,
                "pred_field_coverage": round(coverage_ratio, 6),
                "pred_atom_count": atom_count,
            })
        if len(gt_items) != len(pred_items):
            print(f"[WARN] {source_name} case {case}: GT rows={len(gt_items)} vs Pred rows={len(pred_items)}; evaluated {n}")
    return pd.DataFrame(rows)

def evaluate_row_by_row(gt_df: pd.DataFrame, preds: List[Dict[str, Any]], source_name: str) -> pd.DataFrame:
    gt_sorted = gt_df.copy().reset_index(drop=True)
    gt_sorted["__orig"] = range(len(gt_sorted))
    if "case" in gt_sorted.columns:
        gt_sorted = gt_sorted.sort_values(["case", "__orig"], ignore_index=True)
    else:
        gt_sorted = gt_sorted.sort_values(["__orig"], ignore_index=True)

    n = min(len(gt_sorted), len(preds))
    if len(gt_sorted) != len(preds):
        print(f"[WARN] {source_name}: GT rows={len(gt_sorted)} vs Pred rows={len(preds)}; evaluating first {n}")

    rows = []
    for i in range(n):
        g = gt_sorted.iloc[i].to_dict()
        p = preds[i]
        tp, pp, tt, precision, recall, f1 = prf1(atoms_from_gt(g), atoms_from_pred(p))
        nonempty_fields, coverage_ratio, atom_count = pred_field_coverage_and_atoms(p)
        case_val = g.get("case")
        try:
            case_val = int(case_val) if pd.notna(case_val) else None
        except Exception:
            pass
        rows.append({
            "file": source_name,
            "json_mode": "row_by_row",
            "set": g.get("set"),
            "case": case_val,
            "row_in_case": None,
            "gt_item_title": g.get("item_title"),
            "pred_item_title": p.get("item_title"),
            "pred_count": pp,
            "truth_count": tt,
            "correct": tp,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "pred_nonempty_fields": nonempty_fields,
            "pred_field_coverage": round(coverage_ratio, 6),
            "pred_atom_count": atom_count,
        })
    return pd.DataFrame(rows)


# ---- Main ----
def main():
    ap = argparse.ArgumentParser(description="Evaluate a folder of JSON predictions (case-by-case OR row-by-row) against a ground truth CSV.")
    ap.add_argument("--gt", required=True, help="Path to ground truth CSV")
    ap.add_argument("--pred_dir", required=True, help="Folder that contains JSON files")
    ap.add_argument("--out", required=True, help="Output CSV for per-row metrics")
    ap.add_argument("--w_f1", type=float, default=0.5, help="Weight for F1 in best-example score (0..1)")
    ap.add_argument("--w_pop", type=float, default=0.5, help="Weight for populatedness in best-example score (0..1)")
    args = ap.parse_args()

    if args.w_f1 < 0 or args.w_pop < 0 or (args.w_f1 + args.w_pop) <= 0:
        sys.exit("Weights must be non-negative and not both zero.")

    gt_path = Path(args.gt)
    pred_dir = Path(args.pred_dir)
    out_path = Path(args.out)

    if not gt_path.exists():
        sys.exit(f"Ground truth CSV not found: {gt_path}")
    if not pred_dir.exists():
        sys.exit(f"Prediction folder not found: {pred_dir}")

    gt_df = pd.read_csv(gt_path)

    all_results = []
    json_files = sorted([p for p in pred_dir.glob("*.json") if p.is_file()])
    if not json_files:
        sys.exit(f"No JSON files found in {pred_dir}")

    for jf in json_files:
        try:
            mode, data = load_preds_any(jf)
            if mode == "case_by_case":
                res = evaluate_case_by_case(gt_df, data, source_name=jf.name)
            else:
                res = evaluate_row_by_row(gt_df, data, source_name=jf.name)
            all_results.append(res)
            print(f"[OK] {jf.name} ({mode}): {len(res)} evaluated rows")
        except Exception as e:
            print(f"[SKIP] {jf.name}: {e}")

    if not all_results:
        sys.exit("No evaluations produced any rows. Check JSON formats or field names.")

    out_df = pd.concat(all_results, ignore_index=True)

    # ---- Build populatedness score and best-example score ----
    # Normalize atom_count by robust scale (95th percentile) to keep in [0..1]
    if len(out_df) == 0:
        sys.exit("Empty evaluation dataframe.")
    atom_q95 = max(1.0, float(np.percentile(out_df["pred_atom_count"], 95)))
    pop_score = 0.6 * out_df["pred_field_coverage"] + 0.4 * (out_df["pred_atom_count"] / atom_q95).clip(0, 1)
    out_df["populatedness_score"] = pop_score.round(6)

    w_sum = args.w_f1 + args.w_pop
    wf1 = args.w_f1 / w_sum
    wpop = args.w_pop / w_sum
    out_df["best_example_score"] = (wf1 * out_df["f1"] + wpop * out_df["populatedness_score"]).round(6)

    # Save per-row metrics
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved per-row metrics to {out_path} with {len(out_df)} rows")

    # Per-file macro averages
    summary = (
        out_df.groupby("file")[["precision", "recall", "f1", "pred_field_coverage", "pred_atom_count", "populatedness_score"]]
        .mean()
        .reset_index()
        .sort_values("f1", ascending=False)
    )
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved per-file averages to {summary_path}")

    # Top-10 leaderboards
    def top10(metric: str) -> pd.DataFrame:
        cols = ["file", "case", "row_in_case", "gt_item_title", "pred_item_title",
                "pred_nonempty_fields", "pred_field_coverage", "pred_atom_count",
                "precision", "recall", "f1", "populatedness_score", "best_example_score", "json_mode"]
        df = out_df.sort_values(
            by=[metric, "file", "case", "row_in_case"],
            ascending=[False, True, True, True],
            na_position="last"
        )
        return df[cols].head(10).reset_index(drop=True)

    top_prec = top10("precision")
    top_rec  = top10("recall")
    top_f1   = top10("f1")
    top_pop  = top10("populatedness_score")
    top_best = top10("best_example_score")

    base = out_path.stem
    parent = out_path.parent
    top_prec.to_csv(parent / f"{base}_top10_precision.csv", index=False)
    top_rec.to_csv(parent / f"{base}_top10_recall.csv", index=False)
    top_f1.to_csv(parent / f"{base}_top10_f1.csv", index=False)
    top_pop.to_csv(parent / f"{base}_top10_populated.csv", index=False)
    top_best.to_csv(parent / f"{base}_top10_best_examples.csv", index=False)

    print(f"Saved Top-10 rows by precision      → {parent / f'{base}_top10_precision.csv'}")
    print(f"Saved Top-10 rows by recall         → {parent / f'{base}_top10_recall.csv'}")
    print(f"Saved Top-10 rows by F1             → {parent / f'{base}_top10_f1.csv'}")
    print(f"Saved Top-10 rows by populatedness  → {parent / f'{base}_top10_populated.csv'}")
    print(f"Saved Top-10 rows by BEST (F1+pop)  → {parent / f'{base}_top10_best_examples.csv'}")

if __name__ == "__main__":
    main()
