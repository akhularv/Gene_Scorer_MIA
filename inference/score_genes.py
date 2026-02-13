# - Runs trained model on all animals to produce per-gene scores
# - Aggregates by cohort (timepoint + condition) using median (robust to outliers)
# - Outputs perturbation, transferability, and rank scores per gene per timepoint
# - Saves as gene_scores.npz for export_panel.py to consume

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from model.embedder import MetadataEmbedder
from model.encoder import SampleContextEncoder
from model.gene_scorer import GeneScorer
from training.dataset import MIADataset, collate_batch
from training.train import MIAModel


def score_all_animals(
    model: MIAModel,
    dataset: MIADataset,
    priors: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Run every animal through the model and collect scores + metadata.

    Returns dict with arrays for perturbation, transferability, rank scores,
    plus timepoint and condition labels for cohort aggregation.
    """
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)
    model.eval()

    all_perturb = []
    all_transfer = []
    all_rank = []
    all_tp_names = []
    all_cond_names = []

    with torch.no_grad():
        for batch in loader:
            expr = batch["expression"].to(device)
            tp_ids = batch["timepoint_id"].to(device)
            cond_ids = batch["condition_id"].to(device)
            reg_ids = batch["region_id"].to(device)

            ps, ts, rs = model(expr, tp_ids, cond_ids, reg_ids, priors)

            all_perturb.append(ps.cpu().numpy())
            all_transfer.append(ts.cpu().numpy())
            all_rank.append(rs.cpu().numpy())
            all_tp_names.extend(batch["timepoint_name"])
            all_cond_names.extend(batch["condition_name"])

    return {
        "perturb": np.concatenate(all_perturb, axis=0),
        "transfer": np.concatenate(all_transfer, axis=0),
        "rank": np.concatenate(all_rank, axis=0),
        "timepoints": np.array(all_tp_names),
        "conditions": np.array(all_cond_names),
    }


def aggregate_cohorts(
    scores: dict[str, np.ndarray],
    ef_timepoints: list[str],
) -> dict[str, np.ndarray]:
    """Aggregate scores by cohort using median. polyIC animals only for rank.

    Args:
        scores: output from score_all_animals
        ef_timepoints: timepoints to report (E15, P0, P13)

    Returns:
        dict with cohort-level medians per gene per timepoint
    """
    result = {}

    for tp in ef_timepoints:
        # polyIC animals at this timepoint
        mask = (scores["timepoints"] == tp) & (scores["conditions"] == "polyIC")
        if mask.sum() == 0:
            continue

        # median across animals within this cohort
        result[f"perturbation_{tp}"] = np.median(
            scores["perturb"][mask], axis=0
        )
        result[f"transferability_{tp}"] = np.median(
            scores["transfer"][mask], axis=0
        )
        result[f"rank_{tp}"] = np.median(scores["rank"][mask], axis=0)

    return result


def compute_global_rank(
    cohort_scores: dict[str, np.ndarray],
    ef_timepoints: list[str],
) -> np.ndarray:
    """Mean rank score across timepoints. This is the final ranking metric."""
    rank_arrays = []
    for tp in ef_timepoints:
        key = f"rank_{tp}"
        if key in cohort_scores:
            rank_arrays.append(cohort_scores[key])

    if not rank_arrays:
        raise ValueError("No rank scores found for any timepoint")

    return np.mean(rank_arrays, axis=0)


def score_genes(config: dict, checkpoint_path: str) -> None:
    """Main scoring pipeline: load model, score all animals, aggregate, save."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]

    # load full dataset (all animals, no split)
    dataset = MIADataset(data_dir, output_dir, config)
    n_genes = dataset.n_genes

    # load priors
    priors = torch.from_numpy(dataset.priors).to(device)

    # load model
    model = MIAModel(n_genes, config).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded model from {checkpoint_path}")

    # score all animals
    scores = score_all_animals(model, dataset, priors, device)

    # aggregate by cohort
    ef_timepoints = ["E15", "P0", "P13"]
    cohort_scores = aggregate_cohorts(scores, ef_timepoints)

    # global rank
    global_rank = compute_global_rank(cohort_scores, ef_timepoints)
    cohort_scores["global_rank"] = global_rank

    # save everything
    save_path = os.path.join(output_dir, "gene_scores.npz")
    np.savez(save_path, **cohort_scores)
    print(f"Saved gene scores to {save_path}")
    print(f"Top 5 global rank scores: {np.sort(global_rank)[-5:][::-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score genes with trained model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    score_genes(cfg, args.checkpoint)
