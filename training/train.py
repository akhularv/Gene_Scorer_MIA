# - Full training loop with three-phase schedule for MIA gene scoring
# - Phase 1: encoder warm-up (transferability only, 20% of epochs)
# - Phase 2: perturbation introduction (low weight, 40% of epochs)
# - Phase 3: full training (all losses, final 40% of epochs)
# - Cosine annealing with warm restart at phase transitions
# - Stops when top-25 gene ranking stabilizes (Spearman > 0.95 for 5 epochs)

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import yaml

from model.embedder import MetadataEmbedder
from model.encoder import SampleContextEncoder
from model.gene_scorer import GeneScorer
from training.dataset import (
    MIADataset,
    StratifiedBatchSampler,
    collate_batch,
    train_val_split,
)
from training.losses import total_loss


class MIAModel(nn.Module):
    """Full MIA gene scoring model: embedder + encoder + scorer."""

    def __init__(self, n_genes: int, config: dict) -> None:
        super().__init__()
        self.embedder = MetadataEmbedder(config["timepoint_ages"])
        self.encoder = SampleContextEncoder(
            n_genes=n_genes,
            metadata_dim=config["metadata_dim"],
            context_dim=config["context_dim"],
            h1_dim=config["hidden_dims"]["encoder_h1"],
            h2_dim=config["hidden_dims"]["encoder_h2"],
        )
        self.scorer = GeneScorer(
            context_dim=config["context_dim"],
            prior_dim=5,
        )

    def forward(
        self,
        expression: torch.Tensor,
        timepoint_ids: torch.Tensor,
        condition_ids: torch.Tensor,
        region_ids: torch.Tensor,
        priors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: embed metadata, encode context, score genes.

        Returns:
            perturb_scores, transfer_scores, rank_scores — all [batch x n_genes]
        """
        m = self.embedder(timepoint_ids, condition_ids, region_ids)
        z = self.encoder(expression, m)
        return self.scorer(expression, z, priors)


def get_phase_lambdas(epoch: int, config: dict) -> tuple[float, float, float]:
    """Return loss weights for the current training phase."""
    total = config["total_epochs"]
    phase1_end = int(total * config["phase1_frac"])
    phase2_end = phase1_end + int(total * config["phase2_frac"])

    if epoch < phase1_end:
        p = config["phase1"]
    elif epoch < phase2_end:
        p = config["phase2"]
    else:
        p = config["phase3"]

    return p["lambda_transfer"], p["lambda_perturb"], p["lambda_temporal"]


def get_phase_name(epoch: int, config: dict) -> str:
    """Return human-readable phase name for logging."""
    total = config["total_epochs"]
    phase1_end = int(total * config["phase1_frac"])
    phase2_end = phase1_end + int(total * config["phase2_frac"])

    if epoch < phase1_end:
        return "Phase 1 (encoder warm-up)"
    elif epoch < phase2_end:
        return "Phase 2 (perturbation intro)"
    else:
        return "Phase 3 (full training)"


def validate(
    model: MIAModel,
    val_loader: DataLoader,
    priors: torch.Tensor,
    perturb_target_matrix: torch.Tensor,
    transfer_targets: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    """Run validation and compute correlation metrics.

    Returns:
        transfer_corr: Spearman between predicted and target transferability
        perturb_corr: Spearman between predicted and target perturbation
        mean_rank_scores: [n_genes] average rank scores across val set
    """
    model.eval()
    all_transfer = []
    all_perturb = []
    all_rank = []

    with torch.no_grad():
        for batch in val_loader:
            expr = batch["expression"].to(device)
            tp_ids = batch["timepoint_id"].to(device)
            cond_ids = batch["condition_id"].to(device)
            reg_ids = batch["region_id"].to(device)

            ps, ts, rs = model(expr, tp_ids, cond_ids, reg_ids, priors)
            all_transfer.append(ts.cpu().numpy())
            all_perturb.append(ps.cpu().numpy())
            all_rank.append(rs.cpu().numpy())

    # concatenate across batches
    all_transfer = np.concatenate(all_transfer, axis=0)  # [n_val x n_genes]
    all_perturb = np.concatenate(all_perturb, axis=0)
    all_rank = np.concatenate(all_rank, axis=0)

    # mean scores across val animals
    mean_transfer = all_transfer.mean(axis=0)
    mean_perturb = all_perturb.mean(axis=0)
    mean_rank = all_rank.mean(axis=0)

    # spearman correlation with targets
    transfer_corr, _ = spearmanr(mean_transfer, transfer_targets.cpu().numpy())
    perturb_corr, _ = spearmanr(
        mean_perturb, perturb_target_matrix.cpu().numpy().mean(axis=1)
    )

    model.train()
    return (
        transfer_corr if not np.isnan(transfer_corr) else 0.0,
        perturb_corr if not np.isnan(perturb_corr) else 0.0,
        mean_rank,
    )


def train(config: dict) -> None:
    """Main training loop with three-phase schedule."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # split animals into train/val
    train_ids, val_ids = train_val_split(data_dir, config)
    print(f"Train animals: {len(train_ids)}, Val animals: {len(val_ids)}")

    # create datasets
    train_ds = MIADataset(data_dir, output_dir, config, animal_ids=train_ids)
    val_ds = MIADataset(data_dir, output_dir, config, animal_ids=val_ids)
    n_genes = train_ds.n_genes

    # load precomputed priors and targets onto device
    priors = torch.from_numpy(train_ds.priors).to(device)
    perturb_targets = torch.from_numpy(train_ds.perturb_targets).to(device)
    transfer_targets = torch.from_numpy(train_ds.transfer_targets).to(device)

    # data loaders with stratified batching
    train_sampler = StratifiedBatchSampler(
        train_ds, animals_per_group=config["animals_per_group"]
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], collate_fn=collate_batch
    )

    # model
    model = MIAModel(n_genes, config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # optimizer
    optimizer = AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # cosine annealing scheduler — T_0 is phase 1 length, restarts at transitions
    phase1_epochs = int(config["total_epochs"] * config["phase1_frac"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, phase1_epochs))

    # tracking for early stopping
    prev_top25 = None
    stability_count = 0
    best_val_corr = -1.0

    for epoch in range(config["total_epochs"]):
        model.train()
        lam_t, lam_p, lam_temp = get_phase_lambdas(epoch, config)
        phase = get_phase_name(epoch, config)

        epoch_losses = {"loss_total": 0, "loss_transfer": 0,
                        "loss_perturb": 0, "loss_temporal": 0}
        n_batches = 0

        for batch in train_loader:
            expr = batch["expression"].to(device)
            tp_ids = batch["timepoint_id"].to(device)
            cond_ids = batch["condition_id"].to(device)
            reg_ids = batch["region_id"].to(device)

            ps, ts, rs = model(expr, tp_ids, cond_ids, reg_ids, priors)

            loss, loss_dict = total_loss(
                ps, ts, rs, cond_ids, tp_ids,
                perturb_targets, transfer_targets,
                lam_t, lam_p, lam_temp,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k] += v
            n_batches += 1

        scheduler.step(epoch)

        # average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # validation
        val_transfer_corr, val_perturb_corr, mean_rank = validate(
            model, val_loader, priors, perturb_targets, transfer_targets, device
        )

        # check ranking stability
        top25_idx = np.argsort(mean_rank)[-config["panel_size"]:]
        if prev_top25 is not None:
            # spearman of rankings between consecutive epochs
            rho, _ = spearmanr(
                mean_rank[top25_idx],
                np.zeros_like(top25_idx),  # dummy, we just want the rank correlation
            )
            # simpler: check set overlap and rank correlation
            rank_corr, _ = spearmanr(
                np.argsort(np.argsort(-mean_rank)),
                np.argsort(np.argsort(-prev_mean_rank)),
            )
            if rank_corr > config["ranking_stability_threshold"]:
                stability_count += 1
            else:
                stability_count = 0
        else:
            rank_corr = 0.0

        prev_top25 = top25_idx
        prev_mean_rank = mean_rank.copy()

        # save best model
        combined_corr = val_transfer_corr + val_perturb_corr
        if combined_corr > best_val_corr:
            best_val_corr = combined_corr
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

        # logging
        print(
            f"Epoch {epoch:3d} [{phase}] "
            f"loss={epoch_losses['loss_total']:.4f} "
            f"(T={epoch_losses['loss_transfer']:.4f} "
            f"P={epoch_losses['loss_perturb']:.4f} "
            f"C={epoch_losses['loss_temporal']:.4f}) "
            f"val_transfer_r={val_transfer_corr:.3f} "
            f"val_perturb_r={val_perturb_corr:.3f} "
            f"rank_stability={rank_corr:.3f} "
            f"({stability_count}/{config['ranking_stability_patience']})"
        )

        # early stopping when ranking is stable
        if stability_count >= config["ranking_stability_patience"]:
            print(f"Ranking stable for {stability_count} epochs. Stopping.")
            break

        # warm restart scheduler at phase transitions
        phase1_end = int(config["total_epochs"] * config["phase1_frac"])
        phase2_end = phase1_end + int(config["total_epochs"] * config["phase2_frac"])
        if epoch + 1 in (phase1_end, phase2_end):
            print(f"Phase transition at epoch {epoch + 1} — resetting scheduler")
            remaining = config["total_epochs"] - (epoch + 1)
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=max(1, remaining // 2)
            )

    # save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    print(f"Training complete. Models saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIA gene scoring model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
