"""Rank genes at each measured developmental stage."""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from project_paths import resolve_path


TIMEPOINTS = ["E15", "P0", "P13"]


def load_protein_coding_genes(raw_counts_path: str) -> set:
    """Return protein-coding, non-mitochondrial, non-Gm genes."""
    raw = pd.read_csv(raw_counts_path, index_col=0, usecols=range(8))
    genes = raw[raw["gene_type"] == "protein_coding"]["gene_name"]
    genes = genes[~genes.str.lower().str.startswith("mt-")]
    genes = genes[~genes.str.lower().str.startswith("gm")]
    return set(genes)


def score_timepoint(
    tp: str,
    tp_idx: int,
    perturb: np.ndarray,
    transfer: np.ndarray,
    gene_names: np.ndarray,
    priors: np.ndarray,
    panel_size: int,
    output_dir: str,
    protein_coding_genes: set,
) -> pd.DataFrame:
    """Score one timepoint and save the top panel."""
    perturb_tp = perturb[:, tp_idx]
    rank_score = perturb_tp * transfer

    results = pd.DataFrame({
        "gene_name": gene_names,
        f"perturbation_{tp}": perturb_tp,
        "transferability": transfer,
        "rank_score": rank_score,
        **{
            f"perturbation_{other}": perturb[:, i]
            for i, other in enumerate(TIMEPOINTS)
            if other != tp
        },
        "prior_dev_slope_ef": priors[:, 0],
        "prior_dev_slope_wc": priors[:, 1],
        "prior_direction_concordance": priors[:, 2],
        "prior_stability": priors[:, 3],
        "prior_baseline_expr": priors[:, 4],
    })

    results = results[results["gene_name"].isin(protein_coding_genes)].copy()

    results = results.sort_values("rank_score", ascending=False).reset_index(drop=True)
    results.index = results.index + 1
    results.index.name = "rank"

    top = results.head(panel_size)
    out_path = os.path.join(output_dir, f"panel_{tp}.csv")
    top.to_csv(out_path)

    print(f"\n{'='*70}")
    print(f"  {tp}  —  TOP {panel_size} GENES")
    print(f"{'='*70}")
    print(f"{'Rank':<6}{'Gene':<18}{'Perturb@'+tp:<16}{'Transfer':<12}{'RankScore'}")
    print(f"{'-'*6}{'-'*18}{'-'*16}{'-'*12}{'-'*10}")
    for i, row in top.iterrows():
        print(
            f"{i:<6}{row['gene_name']:<18}"
            f"{row[f'perturbation_{tp}']:<16.4f}"
            f"{row['transferability']:<12.4f}"
            f"{row['rank_score']:.4f}"
        )

    print(f"\n  Mean perturbation@{tp}: {top[f'perturbation_{tp}'].mean():.4f}")
    print(f"  Mean transferability:   {top['transferability'].mean():.4f}")
    print(f"  Mean rank score:        {top['rank_score'].mean():.4f}")
    print(f"  Min rank score:         {top['rank_score'].min():.4f}")
    print(f"\n  Saved → {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Score genes per timepoint (E15, P0, P13)"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--panel-size", type=int, default=None,
        help="Number of top genes per timepoint (overrides config panel_size)"
    )
    args = parser.parse_args()

    with open(resolve_path(args.config), "r") as f:
        config = yaml.safe_load(f)

    output_dir = resolve_path(config["output_dir"])
    panel_size = args.panel_size or config["panel_size"]

    perturb = np.load(os.path.join(output_dir, "perturbation_targets.npy"))
    transfer = np.load(os.path.join(output_dir, "transferability_targets.npy"))
    gene_names = np.load(os.path.join(output_dir, "gene_names.npy"), allow_pickle=True)
    priors = np.load(os.path.join(output_dir, "p_g.npy"))
    protein_coding_genes = load_protein_coding_genes(resolve_path("MIA_Data/all_counts.csv"))

    print(f"Loaded {len(gene_names)} genes")
    print(f"Panel size per timepoint: {panel_size}")

    for tp_idx, tp in enumerate(TIMEPOINTS):
        score_timepoint(
            tp,
            tp_idx,
            perturb,
            transfer,
            gene_names,
            priors,
            panel_size,
            output_dir,
            protein_coding_genes,
        )

    print(f"\nDone. CSVs written to {output_dir}/")


if __name__ == "__main__":
    main()
