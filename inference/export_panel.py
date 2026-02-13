# - Takes scored genes from score_genes.py and exports top-25 as a clean CSV
# - One row per gene with perturbation, transferability, and rank scores per timepoint
# - Sorted by global_rank_score descending — top gene is the best qPCR candidate
# - This is the final deliverable for wet-lab primer selection

import argparse
import os

import numpy as np
import pandas as pd
import yaml


def export_panel(config: dict, scores_path: str) -> None:
    """Export top-N genes as a formatted CSV table.

    Args:
        config: full config dict
        scores_path: path to gene_scores.npz from score_genes.py
    """
    output_dir = config["output_dir"]
    panel_size = config["panel_size"]

    # load gene names
    gene_names = np.load(os.path.join(output_dir, "gene_names.npy"), allow_pickle=True)

    # load scores
    scores = np.load(scores_path)
    global_rank = scores["global_rank"]

    # get top-N gene indices by global rank
    top_indices = np.argsort(global_rank)[-panel_size:][::-1]

    # build output table
    ef_timepoints = ["E15", "P0", "P13"]
    rows = []

    for rank, idx in enumerate(top_indices, 1):
        row = {"rank": rank, "gene": gene_names[idx]}

        for tp in ef_timepoints:
            p_key = f"perturbation_{tp}"
            t_key = f"transferability_{tp}"
            r_key = f"rank_{tp}"

            # some timepoints may be missing if no data existed
            row[f"perturbation_{tp}"] = (
                float(scores[p_key][idx]) if p_key in scores else np.nan
            )
            row[f"transferability_{tp}"] = (
                float(scores[t_key][idx]) if t_key in scores else np.nan
            )
            row[f"rank_score_{tp}"] = (
                float(scores[r_key][idx]) if r_key in scores else np.nan
            )

        row["global_rank_score"] = float(global_rank[idx])
        rows.append(row)

    df = pd.DataFrame(rows)

    # save
    csv_path = os.path.join(output_dir, "top25_panel.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")

    # also print to console for quick inspection
    print(f"\nTop {panel_size} genes for qPCR panel:\n")
    print(df.to_string(index=False))
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export top-N gene panel")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--scores", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    export_panel(cfg, args.scores)
