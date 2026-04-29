"""
integration/stream2_kalish_celltype.py

- Loads kalish_pseudobulk_log2FC.csv from outputs/
- For each gene: aggregates across cell types and sexes weighted by cell count
  stream2_score = weighted mean of |log2FC| (weight = n_cells per group)
- Flags sex-specific genes: detected in male only, female only, or both
- Flags cell-type breadth: n_cell_types_perturbed (|log2FC| >= threshold)
- Saves stream2_kalish_top200.csv

Usage:
    python -m integration.stream2_kalish_celltype
    python -m integration.stream2_kalish_celltype --out-dir outputs --log2fc-threshold 0.5
"""

import argparse
import os

import numpy as np
import pandas as pd

from project_paths import resolve_path


LOG2FC_SIG = 0.5   # |log2FC| threshold to count a gene as perturbed in a cell type
TOP_N      = 200   # number of top genes to output


def main():
    parser = argparse.ArgumentParser(
        description="Rank Kalish scRNA-seq genes by weighted log2FC across cell types (stream 2)"
    )
    parser.add_argument("--out-dir",         type=str,   default="outputs")
    parser.add_argument("--log2fc-threshold", type=float, default=LOG2FC_SIG,
                        help=f"Min |log2FC| to count a gene as perturbed in a cell type "
                             f"(default: {LOG2FC_SIG})")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()

    out_dir = resolve_path(args.out_dir)
    log2fc_th = args.log2fc_threshold

    # --- load input ---
    lfc_path = os.path.join(out_dir, "kalish_pseudobulk_log2FC.csv")
    if not os.path.exists(lfc_path):
        raise FileNotFoundError(
            f"Missing required input: {lfc_path}\n"
            f"Run `python -m precompute.process_kalish` first."
        )

    df = pd.read_csv(lfc_path)
    print(f"Loaded {len(df)} rows from kalish_pseudobulk_log2FC.csv")
    print(f"  Columns: {list(df.columns)}")

    # --- validate required columns ---
    required = {"gene_name", "log2FC", "n_cells"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"kalish_pseudobulk_log2FC.csv is missing columns: {missing}\n"
            f"Re-run `python -m precompute.process_kalish`."
        )

    # --- identify sex and cell_type columns (may or may not be present) ---
    has_sex       = "sex"       in df.columns
    has_cell_type = "cell_type" in df.columns

    # drop rows with missing values in the key scoring columns
    df = df.dropna(subset=["log2FC", "n_cells"])
    df["abs_log2FC"] = df["log2FC"].abs()

    # -----------------------------------------------------------------------
    # Per-gene: weighted mean |log2FC| (weight = n_cells)
    # -----------------------------------------------------------------------
    gene_rows = []
    for gene, grp in df.groupby("gene_name", sort=False):
        weights      = grp["n_cells"].values.astype(float)
        abs_lfc      = grp["abs_log2FC"].values
        total_cells  = weights.sum()

        if total_cells == 0:
            continue

        stream2_score = float(np.average(abs_lfc, weights=weights))

        # --- sex specificity ---
        if has_sex:
            male_lfc   = grp.loc[grp["sex"].str.lower() == "male",   "abs_log2FC"]
            female_lfc = grp.loc[grp["sex"].str.lower() == "female", "abs_log2FC"]
            male_sig   = (male_lfc   >= log2fc_th).any() if len(male_lfc)   > 0 else False
            female_sig = (female_lfc >= log2fc_th).any() if len(female_lfc) > 0 else False

            if   male_sig and female_sig: sex_specificity = "both"
            elif male_sig:                sex_specificity = "male_only"
            elif female_sig:              sex_specificity = "female_only"
            else:                         sex_specificity = "neither"
        else:
            sex_specificity = "unknown"

        # --- cell-type breadth ---
        if has_cell_type:
            sig_mask             = grp["abs_log2FC"] >= log2fc_th
            n_types_perturbed    = grp.loc[sig_mask, "cell_type"].nunique()
            total_cell_types     = grp["cell_type"].nunique()

            # strongest cell type: highest mean |log2FC| across that cell type's rows
            ct_means             = grp.groupby("cell_type")["abs_log2FC"].mean()
            strongest_cell_type  = ct_means.idxmax() if len(ct_means) > 0 else "unknown"
        else:
            n_types_perturbed   = 0
            total_cell_types    = 0
            strongest_cell_type = "unknown"

        gene_rows.append({
            "gene_name":            gene,
            "stream2_score":        stream2_score,
            "sex_specificity":      sex_specificity,
            "n_cell_types_perturbed": n_types_perturbed,
            "n_cell_types_total":   total_cell_types,
            "breadth": (
                "widespread" if n_types_perturbed >= 3
                else "moderate" if n_types_perturbed >= 2
                else "narrow"  if n_types_perturbed == 1
                else "none"
            ),
            "strongest_cell_type":  strongest_cell_type,
            "total_cells":          int(total_cells),
        })

    if not gene_rows:
        raise ValueError("No genes survived aggregation. Check input file.")

    results = pd.DataFrame(gene_rows)
    print(f"\nAggregated {len(results)} genes from {len(df)} pseudobulk rows")

    # --- rank and select top N ---
    results = results.sort_values("stream2_score", ascending=False).reset_index(drop=True)
    results.index     = results.index + 1
    results.index.name = "stream2_rank"

    top_n = results.head(args.top_n)

    # --- save ---
    out_path = os.path.join(out_dir, "stream2_kalish_top200.csv")
    top_n.to_csv(out_path)
    print(f"\nSaved top-{args.top_n} stream2 genes → {out_path}")

    # --- summary ---
    print(f"\n{'='*60}")
    print(f"Stream 2 (Kalish) summary:")
    print(f"  Total genes: {len(results)}")
    print(f"  Top {args.top_n} by stream2_score")
    print(f"  Mean stream2_score: {top_n['stream2_score'].mean():.4f}")
    if has_sex:
        sex_counts = top_n["sex_specificity"].value_counts()
        print(f"  Sex specificity in top {args.top_n}:")
        for k, v in sex_counts.items():
            print(f"    {k}: {v}")
    if has_cell_type:
        breadth_counts = top_n["breadth"].value_counts()
        print(f"  Breadth in top {args.top_n}:")
        for k, v in breadth_counts.items():
            print(f"    {k}: {v}")
    print(f"\nTop 10 stream2 genes:")
    for i, row in top_n.head(10).iterrows():
        print(f"  {i:>4}  {row['gene_name']:<18}  "
              f"score={row['stream2_score']:.4f}  "
              f"sex={row['sex_specificity']:<12}  "
              f"breadth={row['breadth']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
