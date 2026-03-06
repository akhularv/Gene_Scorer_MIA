"""
integration/combine_streams.py

- Loads stream1_canales_top200.csv and stream2_kalish_top200.csv from outputs/
- Combines by reciprocal-rank fusion:
    combined_score = 0.5 * (1 / rank_stream1) + 0.5 * (1 / rank_stream2)
  Genes present in only one stream receive a penalty rank of (TOP_N + 1)
- Classifies each gene by confidence tier:
    TIER_1:  present in both streams                     (both_streams)
    TIER_2:  present in stream1 only (Canales bulk)      (canales_only)
    TIER_3:  present in stream2 only (Kalish scRNA-seq)  (kalish_only)
- Saves generalizable_core_panel.csv

Usage:
    python -m integration.combine_streams
    python -m integration.combine_streams --out-dir outputs --top-n 200
"""

import argparse
import os

import numpy as np
import pandas as pd


TOP_N = 200    # penalty rank for genes absent from one stream


def load_stream(path: str, rank_col: str, score_col: str) -> pd.DataFrame:
    """Load a stream CSV; return df with gene_name, rank, score."""
    df = pd.read_csv(path, index_col=0)   # index = stream rank (stream1_rank / stream2_rank)
    df.index.name = rank_col              # normalise index name before reset
    df = df.reset_index()                 # rank_col becomes a regular column
    # keep only the columns we need
    keep = ["gene_name", rank_col]
    if score_col in df.columns:
        keep.append(score_col)
    return df[keep].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Combine stream1 (Canales) and stream2 (Kalish) via reciprocal-rank fusion"
    )
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--top-n",   type=int, default=TOP_N,
                        help=f"Penalty rank for genes absent from a stream (default: {TOP_N})")
    parser.add_argument("--output-n", type=int, default=100,
                        help="Number of top genes to write to generalizable_core_panel.csv "
                             "(default: 100)")
    args = parser.parse_args()

    out_dir  = args.out_dir
    penalty  = args.top_n + 1

    # --- load streams ---
    s1_path = os.path.join(out_dir, "stream1_canales_top200.csv")
    s2_path = os.path.join(out_dir, "stream2_kalish_top200.csv")

    for p in [s1_path, s2_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing required input: {p}\n"
                f"Run stream1_canales_slopes.py and stream2_kalish_celltype.py first."
            )

    s1 = load_stream(s1_path, rank_col="rank_s1", score_col="stream1_score")
    s2 = load_stream(s2_path, rank_col="rank_s2", score_col="stream2_score")

    print(f"Stream 1 (Canales):  {len(s1)} genes")
    print(f"Stream 2 (Kalish):   {len(s2)} genes")

    # --- merge (outer join) ---
    merged = s1.merge(s2, on="gene_name", how="outer")

    # fill missing ranks with penalty
    merged["rank_s1"] = merged["rank_s1"].fillna(penalty)
    merged["rank_s2"] = merged["rank_s2"].fillna(penalty)

    # --- reciprocal-rank fusion ---
    merged["combined_score"] = (
        0.5 * (1.0 / merged["rank_s1"]) +
        0.5 * (1.0 / merged["rank_s2"])
    )

    # --- confidence tier ---
    in_s1 = merged["rank_s1"] < penalty
    in_s2 = merged["rank_s2"] < penalty
    merged["evidence"] = "both_streams"
    merged.loc[ in_s1 & ~in_s2, "evidence"] = "canales_only"
    merged.loc[~in_s1 &  in_s2, "evidence"] = "kalish_only"

    # --- sort and rank ---
    merged = merged.sort_values("combined_score", ascending=False).reset_index(drop=True)
    merged.index      = merged.index + 1
    merged.index.name = "combined_rank"

    # --- save full list ---
    all_path = os.path.join(out_dir, "combined_streams_all.csv")
    merged.to_csv(all_path)
    print(f"\nSaved all {len(merged)} genes → {all_path}")

    # --- save core panel (top N) ---
    core = merged.head(args.output_n)
    core_path = os.path.join(out_dir, "generalizable_core_panel.csv")
    core.to_csv(core_path)
    print(f"Saved top-{args.output_n} core panel → {core_path}")

    # --- summary ---
    both      = (merged["evidence"] == "both_streams").sum()
    c_only    = (merged["evidence"] == "canales_only").sum()
    k_only    = (merged["evidence"] == "kalish_only").sum()
    core_both = (core["evidence"] == "both_streams").sum()

    print(f"\n{'='*60}")
    print(f"Combined stream summary (all {len(merged)} genes):")
    print(f"  Both streams:  {both}")
    print(f"  Canales only:  {c_only}")
    print(f"  Kalish only:   {k_only}")
    print(f"\nCore panel (top {args.output_n}):")
    print(f"  Both streams:  {core_both} / {args.output_n} "
          f"({100*core_both/args.output_n:.0f}%)")
    print(f"\nTop 10 combined genes:")
    for i, row in core.head(10).iterrows():
        r1 = int(row["rank_s1"]) if row["rank_s1"] < penalty else "—"
        r2 = int(row["rank_s2"]) if row["rank_s2"] < penalty else "—"
        print(f"  {i:>4}  {row['gene_name']:<18}  "
              f"score={row['combined_score']:.5f}  "
              f"s1={str(r1):>4}  s2={str(r2):>4}  "
              f"[{row['evidence']}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
