"""Rank genes from the precomputed perturbation and transfer scores."""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from project_paths import resolve_path


def load_protein_coding_genes(raw_counts_path: str) -> set:
    """Return protein-coding, non-mitochondrial, non-Gm genes."""
    raw = pd.read_csv(raw_counts_path, index_col=0, usecols=range(8))
    genes = raw[raw["gene_type"] == "protein_coding"]["gene_name"]
    genes = genes[~genes.str.lower().str.startswith("mt-")]
    genes = genes[~genes.str.lower().str.startswith("gm")]
    return set(genes)


def main():
    parser = argparse.ArgumentParser(description="Score genes directly from precomputed targets")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--panel-size", type=int, default=None,
                        help="Override panel size from config")
    args = parser.parse_args()

    with open(resolve_path(args.config), "r") as f:
        config = yaml.safe_load(f)

    output_dir = resolve_path(config["output_dir"])
    panel_size = args.panel_size or config["panel_size"]

    perturb = np.load(os.path.join(output_dir, "perturbation_targets.npy"))
    transfer = np.load(os.path.join(output_dir, "transferability_targets.npy"))
    gene_names = np.load(os.path.join(output_dir, "gene_names.npy"), allow_pickle=True)
    priors = np.load(os.path.join(output_dir, "p_g.npy"))

    n_genes = len(gene_names)
    ef_timepoints = ["E15", "P0", "P13"]

    print(f"Loaded {n_genes} genes")
    print(f"Perturbation targets shape: {perturb.shape}")
    print(f"Transferability targets shape: {transfer.shape}")

    mean_perturb = perturb.mean(axis=1)
    rank_score = mean_perturb * transfer
    results = pd.DataFrame({
        "gene_name": gene_names,
        "perturbation_E15": perturb[:, 0],
        "perturbation_P0": perturb[:, 1],
        "perturbation_P13": perturb[:, 2],
        "mean_perturbation": mean_perturb,
        "transferability": transfer,
        "rank_score": rank_score,
        "prior_dev_slope_ef": priors[:, 0],
        "prior_dev_slope_wc": priors[:, 1],
        "prior_direction_concordance": priors[:, 2],
        "prior_stability": priors[:, 3],
        "prior_baseline_expr": priors[:, 4],
    })

    raw_counts_path = resolve_path("MIA_Data/all_counts.csv")
    if os.path.exists(raw_counts_path):
        protein_coding = load_protein_coding_genes(raw_counts_path)
        before = len(results)
        results = results[results["gene_name"].isin(protein_coding)].copy()
        print(f"Kept {len(results)} protein-coding genes (removed {before - len(results)})")

    results = results.sort_values("rank_score", ascending=False).reset_index(drop=True)
    results.index.name = "rank"
    results.index = results.index + 1
    full_path = os.path.join(output_dir, "all_genes_ranked.csv")
    results.to_csv(full_path)
    print(f"\nSaved full ranking: {full_path}")

    top = results.head(panel_size)
    panel_path = os.path.join(output_dir, "top_panel.csv")
    top.to_csv(panel_path)
    print(f"Saved top-{panel_size} panel: {panel_path}")

    print(f"\n{'='*80}")
    print(f"TOP {panel_size} GENES FOR qPCR PANEL")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Gene':<15}{'Perturb':<10}{'Transfer':<10}{'RankScore':<10}")
    print(f"{'-'*6}{'-'*15}{'-'*10}{'-'*10}{'-'*10}")
    for i, row in top.iterrows():
        print(f"{i:<6}{row['gene_name']:<15}{row['mean_perturbation']:<10.4f}"
              f"{row['transferability']:<10.4f}{row['rank_score']:<10.4f}")

    print(f"\n--- Panel summary ---")
    print(f"Mean perturbation:    {top['mean_perturbation'].mean():.4f}")
    print(f"Mean transferability: {top['transferability'].mean():.4f}")
    print(f"Mean rank score:      {top['rank_score'].mean():.4f}")
    print(f"Min rank score:       {top['rank_score'].min():.4f}")

    print(f"\n--- Per-timepoint perturbation (panel genes) ---")
    for j, tp in enumerate(ef_timepoints):
        col = f"perturbation_{tp}"
        print(f"  {tp}: mean={top[col].mean():.4f}, "
              f"min={top[col].min():.4f}, max={top[col].max():.4f}")


if __name__ == "__main__":
    main()
