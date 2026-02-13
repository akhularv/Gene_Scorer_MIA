# - Computes 5 biological prior features per gene from raw expression data
# - Features capture developmental trajectory, cross-tissue concordance, stability, and baseline
# - All features normalized to [0,1], saved as p_g.npy [n_genes x 5]
# - Run once before training — nothing here is learned
# - Expects expression_ef.csv, expression_wc.csv, metadata.csv in data_dir

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def normalize_01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0, 1]. Handles constant arrays gracefully."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def cpm_log1p(counts: np.ndarray) -> np.ndarray:
    """Library-size normalize to CPM then log1p. Rows are samples."""
    lib_sizes = counts.sum(axis=1, keepdims=True)
    # avoid division by zero for empty libraries
    lib_sizes = np.maximum(lib_sizes, 1.0)
    cpm = counts / lib_sizes * 1e6
    return np.log1p(cpm)


def load_expression_and_meta(
    data_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load both expression matrices and metadata. Returns normalized expression."""
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    ef_raw = pd.read_csv(os.path.join(data_dir, "expression_ef.csv"))
    wc_raw = pd.read_csv(os.path.join(data_dir, "expression_wc.csv"))

    # separate metadata columns from gene columns
    meta_cols = ["animal_id", "timepoint", "condition"]
    gene_cols = [c for c in ef_raw.columns if c not in meta_cols]

    # normalize expression values
    ef_expr = cpm_log1p(ef_raw[gene_cols].values.astype(np.float64))
    wc_expr = cpm_log1p(wc_raw[gene_cols].values.astype(np.float64))

    ef_df = ef_raw[meta_cols].copy()
    ef_df[gene_cols] = ef_expr
    wc_df = wc_raw[meta_cols].copy()
    wc_df[gene_cols] = wc_expr

    return ef_df, wc_df, meta, gene_cols


def compute_mean_by_timepoint(
    df: pd.DataFrame, gene_cols: list[str], timepoints: list[str], condition: str
) -> dict[str, np.ndarray]:
    """Compute mean expression per gene at each timepoint for a given condition."""
    means = {}
    subset = df[df["condition"] == condition]
    for tp in timepoints:
        tp_data = subset[subset["timepoint"] == tp][gene_cols].values
        if len(tp_data) > 0:
            means[tp] = tp_data.mean(axis=0)
    return means


def linear_slope(y_vals: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
    """Least-squares slope for each gene across timepoints. Vectorized."""
    # x_vals: [n_timepoints], y_vals: [n_timepoints x n_genes]
    x_mean = x_vals.mean()
    y_mean = y_vals.mean(axis=0)
    numerator = ((x_vals[:, None] - x_mean) * (y_vals - y_mean)).sum(axis=0)
    denominator = ((x_vals - x_mean) ** 2).sum()
    return numerator / (denominator + 1e-12)


def compute_priors(config: dict) -> None:
    """Main function: compute p_g matrix [n_genes x 5] and save it."""
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    ef_df, wc_df, meta, gene_cols = load_expression_and_meta(data_dir)
    n_genes = len(gene_cols)

    # save n_genes back so downstream scripts know dimensionality
    config["n_genes"] = n_genes

    # === Feature 0: developmental slope in EF (saline only) ===
    ef_timepoints = ["E15", "P0", "P13"]
    ef_ages = np.array([-5.0, 0.0, 13.0])  # numeric ages for regression
    ef_means = compute_mean_by_timepoint(ef_df, gene_cols, ef_timepoints, "saline")
    ef_y = np.stack([ef_means[tp] for tp in ef_timepoints])  # [3 x n_genes]
    feat0 = linear_slope(ef_y, ef_ages)

    # === Feature 1: developmental slope in WC (saline only) ===
    wc_timepoints = ["E15", "P0", "P70", "P189"]
    wc_ages = np.array([-5.0, 0.0, 70.0, 189.0])
    wc_means = compute_mean_by_timepoint(wc_df, gene_cols, wc_timepoints, "saline")
    wc_y = np.stack([wc_means[tp] for tp in wc_timepoints])  # [4 x n_genes]
    feat1 = linear_slope(wc_y, wc_ages)

    # === Feature 2: direction concordance (Spearman EF vs WC at shared timepoints) ===
    shared_timepoints = ["E15", "P0"]  # only timepoints present in both tissues
    ef_shared = np.stack([ef_means[tp] for tp in shared_timepoints])  # [2 x n_genes]
    wc_shared = np.stack([wc_means[tp] for tp in shared_timepoints])
    feat2 = np.zeros(n_genes)
    for g in range(n_genes):
        # spearman over shared timepoints for this gene
        if ef_shared.shape[0] >= 2:
            rho, _ = spearmanr(ef_shared[:, g], wc_shared[:, g])
            feat2[g] = rho if not np.isnan(rho) else 0.0
        else:
            feat2[g] = 0.0

    # === Feature 3: expression stability — how similar EF and WC variances are ===
    # use all available saline timepoints per tissue
    ef_all = np.stack([ef_means[tp] for tp in ef_timepoints])
    wc_all = np.stack([wc_means[tp] for tp in wc_timepoints])
    var_ef = ef_all.var(axis=0)   # variance across timepoints per gene
    var_wc = wc_all.var(axis=0)
    feat3 = 1.0 / (1.0 + np.abs(var_ef - var_wc))

    # === Feature 4: baseline expression in WC (mean across saline timepoints) ===
    feat4 = wc_all.mean(axis=0)

    # === Normalize all to [0, 1] ===
    p_g = np.stack([
        normalize_01(feat0),
        normalize_01(feat1),
        normalize_01(feat2),
        normalize_01(feat3),
        normalize_01(feat4),
    ], axis=1)  # [n_genes x 5]

    # save
    np.save(os.path.join(output_dir, "p_g.npy"), p_g.astype(np.float32))

    # also save gene names for downstream use
    np.save(os.path.join(output_dir, "gene_names.npy"), np.array(gene_cols))

    # update config with n_genes
    config_path = os.path.join("configs", "config.yaml")
    config["n_genes"] = n_genes
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Saved p_g.npy: shape {p_g.shape}")
    print(f"n_genes = {n_genes}")
    print(f"Feature ranges: {p_g.min(axis=0)} to {p_g.max(axis=0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute gene prior features")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    compute_priors(cfg)
