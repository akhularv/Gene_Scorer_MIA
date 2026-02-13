# - Computes supervised targets for perturbation and transferability losses
# - perturbation_targets: |log2FC| between polyIC and saline per gene per timepoint
# - transferability_targets: Spearman correlation of EF vs WC trajectory per gene
# - Both normalized to [0,1], saved as .npy files
# - Run once after compute_priors.py

import argparse
import os

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from precompute.compute_priors import (
    cpm_log1p,
    load_config,
    normalize_01,
)


def load_expression(
    data_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load and normalize both expression files. Returns dataframes + gene column names."""
    meta_cols = ["animal_id", "timepoint", "condition"]

    ef_raw = pd.read_csv(os.path.join(data_dir, "expression_ef.csv"))
    wc_raw = pd.read_csv(os.path.join(data_dir, "expression_wc.csv"))

    gene_cols = [c for c in ef_raw.columns if c not in meta_cols]

    # normalize counts to CPM then log1p
    ef_expr = cpm_log1p(ef_raw[gene_cols].values.astype(np.float64))
    wc_expr = cpm_log1p(wc_raw[gene_cols].values.astype(np.float64))

    ef_df = ef_raw[meta_cols].copy()
    ef_df[gene_cols] = ef_expr
    wc_df = wc_raw[meta_cols].copy()
    wc_df[gene_cols] = wc_expr

    return ef_df, wc_df, gene_cols


def compute_log2fc(
    df: pd.DataFrame, gene_cols: list[str], timepoint: str
) -> np.ndarray:
    """Compute |log2FC| between polyIC and saline means at one timepoint."""
    saline = df[(df["timepoint"] == timepoint) & (df["condition"] == "saline")]
    polyic = df[(df["timepoint"] == timepoint) & (df["condition"] == "polyIC")]

    # mean expression per gene — already in log1p(CPM) space
    mean_sal = saline[gene_cols].values.mean(axis=0)
    mean_poly = polyic[gene_cols].values.mean(axis=0)

    # log2FC from log1p space: convert back, compute ratio, take log2
    # use pseudocount to avoid log(0)
    pseudo = 1e-6
    fc = np.abs(
        np.log2(np.expm1(mean_poly) + pseudo)
        - np.log2(np.expm1(mean_sal) + pseudo)
    )
    return fc


def compute_targets(config: dict) -> None:
    """Compute perturbation and transferability targets, save as .npy."""
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    ef_df, wc_df, gene_cols = load_expression(data_dir)
    n_genes = len(gene_cols)

    # === Perturbation targets: |log2FC| per gene per EF timepoint ===
    # these are the timepoints where we have EF data and thus perturbation signal
    ef_timepoints = ["E15", "P0", "P13"]
    perturb_cols = []
    for tp in ef_timepoints:
        fc = compute_log2fc(ef_df, gene_cols, tp)
        perturb_cols.append(normalize_01(fc))  # normalize per timepoint

    # shape: [n_genes x 3]
    perturbation_targets = np.stack(perturb_cols, axis=1).astype(np.float32)

    # === Transferability targets: Spearman of EF vs WC mean trajectory ===
    # shared timepoints where both EF and WC have saline data
    shared_tps = ["E15", "P0"]
    ef_means = []
    wc_means = []
    for tp in shared_tps:
        ef_sal = ef_df[(ef_df["timepoint"] == tp) & (ef_df["condition"] == "saline")]
        wc_sal = wc_df[(wc_df["timepoint"] == tp) & (wc_df["condition"] == "saline")]
        ef_means.append(ef_sal[gene_cols].values.mean(axis=0))
        wc_means.append(wc_sal[gene_cols].values.mean(axis=0))

    ef_traj = np.stack(ef_means)  # [2 x n_genes]
    wc_traj = np.stack(wc_means)  # [2 x n_genes]

    transfer = np.zeros(n_genes)
    for g in range(n_genes):
        rho, _ = spearmanr(ef_traj[:, g], wc_traj[:, g])
        transfer[g] = rho if not np.isnan(rho) else 0.0

    transferability_targets = normalize_01(transfer).astype(np.float32)

    # save both
    np.save(
        os.path.join(output_dir, "perturbation_targets.npy"), perturbation_targets
    )
    np.save(
        os.path.join(output_dir, "transferability_targets.npy"),
        transferability_targets,
    )

    print(f"Saved perturbation_targets.npy: shape {perturbation_targets.shape}")
    print(f"Saved transferability_targets.npy: shape {transferability_targets.shape}")
    print(
        f"Perturbation range: [{perturbation_targets.min():.3f}, "
        f"{perturbation_targets.max():.3f}]"
    )
    print(
        f"Transferability range: [{transferability_targets.min():.3f}, "
        f"{transferability_targets.max():.3f}]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute training targets")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    compute_targets(cfg)
