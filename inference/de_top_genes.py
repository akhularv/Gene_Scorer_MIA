"""Per-timepoint differential expression: top 2 up- and top 2 downregulated genes.

For each EF timepoint (E15, P0, P13):
  - Computes mean raw counts per gene for polyIC vs saline
  - Classifies each gene's expression level from raw counts (High / Medium / Low)
  - Filters to highly-expressed genes only (mean raw count >= HIGH_EXPR_THRESHOLD)
  - Among those, picks the 2 most upregulated (highest log2FC) and
    2 most downregulated (most negative log2FC)

Expression classification thresholds (raw counts):
    High   : mean raw count >= 500
    Medium : mean raw count >= 100
    Low    : mean raw count <  100

Usage:
    python -m inference.de_top_genes
    python -m inference.de_top_genes --high-expr-threshold 1000
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml


TIMEPOINTS = ["E15", "P0", "P13"]

# Raw-count thresholds for expression classification
HIGH_THRESHOLD   = 500
MEDIUM_THRESHOLD = 100

PSEUDO = 1.0  # pseudocount added before log2FC to avoid division by zero


def classify_expression(mean_raw: float) -> str:
    if mean_raw >= HIGH_THRESHOLD:
        return "High"
    elif mean_raw >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def de_for_timepoint(
    tp: str,
    ef: pd.DataFrame,
    high_expr_threshold: float,
) -> pd.DataFrame:
    """Return top-2 up and top-2 down regulated, highly-expressed genes at one timepoint.

    Args:
        tp:                  Timepoint label, e.g. "E15"
        ef:                  expression_ef DataFrame (animal_id, timepoint, condition, genes...)
        high_expr_threshold: Minimum mean raw count to be considered highly expressed

    Returns:
        DataFrame with 4 rows (2 up, 2 down), sorted by log2FC descending.
    """
    subset = ef[ef["timepoint"] == tp]
    if subset.empty:
        raise ValueError(f"No EF samples found for timepoint {tp}")

    polyic  = subset[subset["condition"] == "polyIC"]
    saline  = subset[subset["condition"] == "saline"]

    if polyic.empty or saline.empty:
        raise ValueError(f"Missing polyIC or saline samples at {tp}")

    gene_cols = [c for c in ef.columns if c not in ("animal_id", "timepoint", "condition", "region")]

    mean_polyic = polyic[gene_cols].mean()
    mean_saline = saline[gene_cols].mean()

    # Overall mean raw expression across both conditions (for classification)
    mean_overall = subset[gene_cols].mean()

    # Log2 fold-change: log2((mean_polyIC + pseudo) / (mean_saline + pseudo))
    log2fc = np.log2((mean_polyic + PSEUDO) / (mean_saline + PSEUDO))

    df = pd.DataFrame({
        "gene_name":        gene_cols,
        "mean_raw_polyIC":  mean_polyic.values,
        "mean_raw_saline":  mean_saline.values,
        "mean_raw_overall": mean_overall.values,
        "log2FC":           log2fc.values,
        "timepoint":        tp,
    })

    df["expression_class"] = df["mean_raw_overall"].apply(classify_expression)

    # Keep only highly-expressed genes
    high_expr = df[df["mean_raw_overall"] >= high_expr_threshold].copy()

    if len(high_expr) < 4:
        raise ValueError(
            f"Only {len(high_expr)} highly-expressed genes at {tp} "
            f"(threshold={high_expr_threshold}). Lower --high-expr-threshold."
        )

    top2_up   = high_expr.nlargest(2, "log2FC")
    top2_down = high_expr.nsmallest(2, "log2FC")

    result = pd.concat([top2_up, top2_down]).sort_values("log2FC", ascending=False)
    result.insert(0, "direction", ["Up", "Up", "Down", "Down"])
    result = result.reset_index(drop=True)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Top 2 up- and 2 downregulated highly-expressed genes per timepoint"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--high-expr-threshold",
        type=float,
        default=HIGH_THRESHOLD,
        help=f"Minimum mean raw count to qualify as highly expressed (default: {HIGH_THRESHOLD})",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_dir   = config["data_dir"]
    output_dir = config["output_dir"]

    ef_path = os.path.join(data_dir, "expression_ef.csv")
    print(f"Loading EF expression data from {ef_path} ...")
    ef = pd.read_csv(ef_path)

    # Drop 'region' column if present (all rows are excitatory_frontal)
    if "region" in ef.columns:
        ef = ef.drop(columns=["region"])

    results = []
    for tp in TIMEPOINTS:
        tp_result = de_for_timepoint(tp, ef, args.high_expr_threshold)
        results.append(tp_result)

    all_results = pd.concat(results, ignore_index=True)

    out_path = os.path.join(output_dir, "de_top_genes.csv")
    all_results.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}\n")

    # Pretty-print
    for tp in TIMEPOINTS:
        tp_rows = all_results[all_results["timepoint"] == tp]
        print(f"{'='*72}")
        print(f"  {tp}  —  Top 2 Up / 2 Down (highly expressed, threshold ≥ {args.high_expr_threshold:.0f} raw counts)")
        print(f"{'='*72}")
        print(f"  {'Dir':<6} {'Gene':<18} {'log2FC':>8}  {'mean_polyIC':>12}  {'mean_saline':>12}  {'ExprClass'}")
        print(f"  {'-'*6} {'-'*18} {'-'*8}  {'-'*12}  {'-'*12}  {'-'*9}")
        for _, row in tp_rows.iterrows():
            print(
                f"  {row['direction']:<6} {row['gene_name']:<18} "
                f"{row['log2FC']:>8.3f}  "
                f"{row['mean_raw_polyIC']:>12.1f}  "
                f"{row['mean_raw_saline']:>12.1f}  "
                f"{row['expression_class']}"
            )
        print()


if __name__ == "__main__":
    main()
