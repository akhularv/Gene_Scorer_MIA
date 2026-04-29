"""
integration/stream1_canales_slopes.py

- Loads canales_slopes.csv and canales_log2FC.csv from outputs/
- Filters genes requiring meaningful fits: R2_MIA > 0.3 AND R2_saline > 0.3
- Computes stream1_score = |slope_divergence| * max_abs_log2FC across timepoints
  (both the trajectory trend AND the peak perturbation magnitude must be high)
- Ranks genes by stream1_score descending
- Saves top 200 genes to stream1_canales_top200.csv

Usage:
    python -m integration.stream1_canales_slopes
    python -m integration.stream1_canales_slopes --out-dir outputs --r2-threshold 0.3
"""

import argparse
import os

import numpy as np
import pandas as pd

from project_paths import resolve_path


R2_THRESHOLD = 0.3    # minimum R2 for both MIA and saline slope fits
TOP_N        = 200    # number of top genes to output


def main():
    parser = argparse.ArgumentParser(
        description="Rank Canales genes by slope divergence × peak log2FC (stream 1)"
    )
    parser.add_argument("--out-dir",      type=str,   default="outputs")
    parser.add_argument("--r2-threshold", type=float, default=R2_THRESHOLD,
                        help=f"Min R2 for slope fits (default: {R2_THRESHOLD})")
    parser.add_argument("--top-n",        type=int,   default=TOP_N)
    args = parser.parse_args()

    out_dir = resolve_path(args.out_dir)

    # --- load inputs ---
    slopes_path = os.path.join(out_dir, "canales_slopes.csv")
    log2fc_path = os.path.join(out_dir, "canales_log2FC.csv")

    for path in [slopes_path, log2fc_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing required input: {path}\n"
                f"Run `python -m precompute.process_canales` first."
            )

    slopes = pd.read_csv(slopes_path)
    log2fc = pd.read_csv(log2fc_path)
    print(f"Loaded {len(slopes)} genes from canales_slopes.csv")
    print(f"Loaded {len(log2fc)} genes from canales_log2FC.csv")

    # --- merge ---
    df = slopes.merge(log2fc, on="gene_name", how="inner")
    print(f"  {len(df)} genes after inner join on gene_name")

    # --- compute max absolute log2FC across all timepoints ---
    fc_cols = [c for c in df.columns if c.startswith("log2FC_")]
    if not fc_cols:
        raise ValueError(
            f"No log2FC columns found after merge. "
            f"Columns: {list(df.columns)}"
        )
    df["max_abs_log2FC"] = df[fc_cols].abs().max(axis=1)

    # --- R2 filter ---
    # R2_MIA and R2_saline may be identical (= R2_divergence) when only DE tables
    # were available in process_canales.py — the filter still provides useful signal
    before = len(df)
    df_filt = df[
        (df["R2_MIA"]    >= args.r2_threshold) &
        (df["R2_saline"] >= args.r2_threshold) &
        df["slope_divergence"].notna() &
        df["max_abs_log2FC"].notna()
    ].copy()
    print(f"\nR2 filter (>= {args.r2_threshold}): {len(df_filt)}/{before} genes pass")

    # --- stream1 score ---
    df_filt["stream1_score"] = df_filt["slope_divergence"].abs() * df_filt["max_abs_log2FC"]

    # --- rank and select top N ---
    df_filt = df_filt.sort_values("stream1_score", ascending=False).reset_index(drop=True)
    df_filt.index = df_filt.index + 1
    df_filt.index.name = "stream1_rank"

    top_n = df_filt.head(args.top_n)

    # --- save ---
    out_path = os.path.join(out_dir, "stream1_canales_top200.csv")
    top_n.to_csv(out_path)
    print(f"\nSaved top-{args.top_n} stream1 genes → {out_path}")

    # --- summary ---
    print(f"\n{'='*60}")
    print(f"Stream 1 (Canales) summary:")
    print(f"  Genes passing R2 filter:  {len(df_filt)}")
    print(f"  Top {args.top_n} by stream1_score")
    print(f"  Mean |slope_divergence|:  {top_n['slope_divergence'].abs().mean():.4f}")
    print(f"  Mean max_abs_log2FC:      {top_n['max_abs_log2FC'].mean():.4f}")
    print(f"  Mean stream1_score:       {top_n['stream1_score'].mean():.4f}")
    print(f"\nTop 10 stream1 genes:")
    for i, row in top_n.head(10).iterrows():
        print(f"  {i:>4}  {row['gene_name']:<18}  "
              f"slope_div={row['slope_divergence']:+.4f}  "
              f"max_log2FC={row['max_abs_log2FC']:.3f}  "
              f"score={row['stream1_score']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
