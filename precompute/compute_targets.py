# - Computes supervised targets for perturbation and transferability losses
# - perturbation_targets: log-space difference between polyIC and saline per gene per timepoint
# - transferability_targets: Pearson correlation of EF vs WC across shared condition-timepoint groups
# - Both normalized to [0,1], saved as .npy files
# - Run once after compute_priors.py

import argparse
import os

import numpy as np
import pandas as pd
import yaml

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

    ef_df = pd.concat(
        [ef_raw[meta_cols].reset_index(drop=True),
         pd.DataFrame(ef_expr, columns=gene_cols)], axis=1
    )
    wc_df = pd.concat(
        [wc_raw[meta_cols].reset_index(drop=True),
         pd.DataFrame(wc_expr, columns=gene_cols)], axis=1
    )

    return ef_df, wc_df, gene_cols


def compute_perturbation(
    df: pd.DataFrame, gene_cols: list[str], timepoint: str
) -> np.ndarray:
    """Compute perturbation strength between polyIC and saline at one timepoint.

    Uses absolute difference in log1p(CPM) space rather than back-transformed
    log2FC. This is critical because the old log2FC method inflates fold-changes
    for near-zero genes: expm1(tiny) + 1e-6 pseudocount makes a count of 0 vs 1
    look like a 13+ fold-change, drowning out real biological signal from
    well-expressed genes with modest but real perturbation.

    In log1p(CPM) space, the same 0-vs-1 noise difference compresses to ~0.01,
    while a real 1.5x change in a well-expressed gene gives ~0.4. This naturally
    combines fold-change with magnitude — the same fold-change produces a larger
    log-space difference for higher-expressed genes.
    """
    saline = df[(df["timepoint"] == timepoint) & (df["condition"] == "saline")]
    polyic = df[(df["timepoint"] == timepoint) & (df["condition"] == "polyIC")]

    # mean expression per gene — already in log1p(CPM) space
    mean_sal = saline[gene_cols].values.mean(axis=0)
    mean_poly = polyic[gene_cols].values.mean(axis=0)

    # absolute difference in log1p(CPM) space
    return np.abs(mean_poly - mean_sal)


def compute_targets(config: dict) -> None:
    """Compute perturbation and transferability targets, save as .npy."""
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    ef_df, wc_df, gene_cols = load_expression(data_dir)
    n_genes = len(gene_cols)

    # === Perturbation targets: log-space difference per gene per EF timepoint ===
    # these are the timepoints where we have EF data and thus perturbation signal
    ef_timepoints = ["E15", "P0", "P13"]
    perturb_cols = []
    for tp in ef_timepoints:
        fc = compute_perturbation(ef_df, gene_cols, tp)
        perturb_cols.append(normalize_01(fc))  # normalize per timepoint

    # shape: [n_genes x 3]
    perturbation_targets = np.stack(perturb_cols, axis=1).astype(np.float32)

    # === Transferability targets: Pearson of EF vs WC across shared groups ===
    # The old approach used 2-point Spearman (saline only at E15, P0), which
    # always gives exactly +1 or -1 — meaningless. Noise genes got perfect 1.0
    # by chance because both tissues had near-zero values tracking together.
    #
    # New approach: use ALL shared (timepoint, condition) groups to get 4+ data
    # points per gene, then compute Pearson correlation. This measures whether
    # the gene's expression co-varies across conditions/timepoints in both
    # tissues — i.e., will perturbation in EF neurons be visible in WC bulk?
    shared_tps = ["E15", "P0"]
    shared_groups = []
    ef_group_means = []
    wc_group_means = []
    for tp in shared_tps:
        for cond in ["saline", "polyIC"]:
            ef_sub = ef_df[(ef_df["timepoint"] == tp) & (ef_df["condition"] == cond)]
            wc_sub = wc_df[(wc_df["timepoint"] == tp) & (wc_df["condition"] == cond)]
            if len(ef_sub) > 0 and len(wc_sub) > 0:
                shared_groups.append((tp, cond))
                ef_group_means.append(ef_sub[gene_cols].values.mean(axis=0))
                wc_group_means.append(wc_sub[gene_cols].values.mean(axis=0))

    ef_traj = np.stack(ef_group_means)  # [n_groups x n_genes]
    wc_traj = np.stack(wc_group_means)  # [n_groups x n_genes]

    print(f"Transferability: using {len(shared_groups)} shared groups: {shared_groups}")

    transfer = np.zeros(n_genes)
    for g in range(n_genes):
        ef_vals = ef_traj[:, g]
        wc_vals = wc_traj[:, g]
        ef_std = np.std(ef_vals)
        wc_std = np.std(wc_vals)

        if ef_std < 1e-8 and wc_std < 1e-8:
            # both constant across groups — no variation to correlate;
            # gene isn't dynamic, so transferability is indeterminate
            transfer[g] = 0.0
        elif ef_std < 1e-8 or wc_std < 1e-8:
            # one tissue varies, the other doesn't — poor transferability
            transfer[g] = 0.0
        else:
            rho = np.corrcoef(ef_vals, wc_vals)[0, 1]
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
