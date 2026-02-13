# - Three loss terms for MIA gene scoring: perturbation, transferability, temporal
# - L_perturb: MSE on polyIC samples only (saline contributes zero)
# - L_transfer: MSE on all samples (every animal helps learn cross-tissue correlation)
# - L_temporal: variance regularizer — penalizes genes with unstable rank across timepoints
# - Total loss is weighted sum with phase-dependent lambdas

import torch
import torch.nn.functional as F


def perturbation_loss(
    perturb_scores: torch.Tensor,
    perturb_targets: torch.Tensor,
    condition_ids: torch.Tensor,
    timepoint_ids: torch.Tensor,
    target_matrix: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted perturbation scores and |log2FC| targets.

    Only active for polyIC samples. Saline animals have no perturbation target
    because they ARE the baseline.

    Args:
        perturb_scores: [batch x n_genes] predicted perturbation scores
        perturb_targets: not used (kept for API compat), targets come from target_matrix
        condition_ids: [batch] 0=saline, 1=polyIC
        timepoint_ids: [batch] index into target_matrix columns
        target_matrix: [n_genes x n_timepoints] precomputed perturbation targets

    Returns:
        scalar loss (zero if no polyIC samples in batch)
    """
    # mask: only polyIC samples contribute
    poly_mask = condition_ids == 1
    if poly_mask.sum() == 0:
        return torch.tensor(0.0, device=perturb_scores.device)

    poly_scores = perturb_scores[poly_mask]       # [n_poly x n_genes]
    poly_tp_ids = timepoint_ids[poly_mask]         # [n_poly]

    # gather the correct timepoint column for each polyIC sample
    # target_matrix columns: E15=0, P0=1, P13=2 (only 3 columns for EF timepoints)
    # but timepoint_ids use the global map (E15=0, P0=1, P13=2, P70=3, P189=4)
    # we need to map global tp id to perturbation target column index
    tp_to_col = {0: 0, 1: 1, 2: 2}  # E15->0, P0->1, P13->2
    valid_mask = torch.zeros(len(poly_tp_ids), dtype=torch.bool,
                             device=perturb_scores.device)
    col_indices = torch.zeros(len(poly_tp_ids), dtype=torch.long,
                              device=perturb_scores.device)

    for i, tp_id in enumerate(poly_tp_ids.tolist()):
        if tp_id in tp_to_col:
            valid_mask[i] = True
            col_indices[i] = tp_to_col[tp_id]

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=perturb_scores.device)

    # gather targets for valid samples
    valid_scores = poly_scores[valid_mask]           # [n_valid x n_genes]
    valid_cols = col_indices[valid_mask]              # [n_valid]
    targets = target_matrix[:, valid_cols].T          # [n_valid x n_genes]

    return F.mse_loss(valid_scores, targets)


def transferability_loss(
    transfer_scores: torch.Tensor,
    transfer_targets: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted transferability and Spearman-based target.

    Active for ALL samples — every animal helps learn the EF-WC correlation
    pattern regardless of condition.

    Args:
        transfer_scores: [batch x n_genes] predicted transferability
        transfer_targets: [n_genes] precomputed target (broadcast across batch)

    Returns:
        scalar loss
    """
    # broadcast target to batch dimension
    targets = transfer_targets.unsqueeze(0).expand_as(transfer_scores)
    return F.mse_loss(transfer_scores, targets)


def temporal_consistency_loss(
    rank_scores: torch.Tensor,
    timepoint_ids: torch.Tensor,
    condition_ids: torch.Tensor,
) -> torch.Tensor:
    """Variance of rank scores across timepoints, averaged over genes.

    This is a regularizer with no external target. It penalizes genes whose
    MIA disruption score bounces around across development. We want genes
    that are stably disrupted, not timepoint-specific noise.

    Only uses polyIC samples (saline has no meaningful perturbation signal).

    Args:
        rank_scores: [batch x n_genes]
        timepoint_ids: [batch]
        condition_ids: [batch]

    Returns:
        scalar loss (zero if batch has <2 timepoints)
    """
    poly_mask = condition_ids == 1
    if poly_mask.sum() == 0:
        return torch.tensor(0.0, device=rank_scores.device)

    poly_scores = rank_scores[poly_mask]       # [n_poly x n_genes]
    poly_tp_ids = timepoint_ids[poly_mask]     # [n_poly]

    # group by timepoint, compute mean rank per timepoint per gene
    unique_tps = poly_tp_ids.unique()
    if len(unique_tps) < 2:
        # can't compute variance with one timepoint
        return torch.tensor(0.0, device=rank_scores.device)

    tp_means = []
    for tp in unique_tps:
        tp_mask = poly_tp_ids == tp
        # mean rank score for this timepoint across animals
        tp_means.append(poly_scores[tp_mask].mean(dim=0))

    # stack to [n_timepoints_in_batch x n_genes]
    tp_stack = torch.stack(tp_means, dim=0)

    # variance across timepoints per gene, then mean over genes
    var_per_gene = tp_stack.var(dim=0)  # [n_genes]
    return var_per_gene.mean()


def total_loss(
    perturb_scores: torch.Tensor,
    transfer_scores: torch.Tensor,
    rank_scores: torch.Tensor,
    condition_ids: torch.Tensor,
    timepoint_ids: torch.Tensor,
    perturb_target_matrix: torch.Tensor,
    transfer_targets: torch.Tensor,
    lambda_transfer: float,
    lambda_perturb: float,
    lambda_temporal: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted total loss from all three terms.

    Returns total loss tensor and a dict of individual loss values for logging.
    """
    l_transfer = transferability_loss(transfer_scores, transfer_targets)
    l_perturb = perturbation_loss(
        perturb_scores, None, condition_ids, timepoint_ids, perturb_target_matrix
    )
    l_temporal = temporal_consistency_loss(rank_scores, timepoint_ids, condition_ids)

    loss = (
        lambda_transfer * l_transfer
        + lambda_perturb * l_perturb
        + lambda_temporal * l_temporal
    )

    loss_dict = {
        "loss_total": loss.item(),
        "loss_transfer": l_transfer.item(),
        "loss_perturb": l_perturb.item(),
        "loss_temporal": l_temporal.item(),
    }

    return loss, loss_dict
