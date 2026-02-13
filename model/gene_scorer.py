# - Scores every gene for perturbation strength and transferability
# - Same MLP weights shared across all genes (weight sharing is the point)
# - What differentiates genes: their raw expression scalar + precomputed prior
# - Outputs two [0,1] scores per gene; rank_score = product of both
# - Runs all genes in parallel via batched matrix ops

import torch
import torch.nn as nn


class GeneScorer(nn.Module):
    """Shared-weight MLP that scores each gene independently.

    For each gene g, the input is:
      - x_g [1]: that gene's expression value in this animal
      - z [128]: animal-level context from encoder (same for all genes)
      - p_g [5]: precomputed biological prior for this gene

    Concatenated to 134-dim, pushed through a small MLP, split into
    perturbation and transferability scores. Both sigmoid-bounded to [0,1].
    Rank score is their product — both must be high.
    """

    def __init__(self, context_dim: int = 128, prior_dim: int = 5) -> None:
        super().__init__()
        input_dim = 1 + context_dim + prior_dim  # 134

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # two outputs: perturbation, transferability
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        expression: torch.Tensor,
        z: torch.Tensor,
        priors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score all genes for one batch of animals.

        Args:
            expression: [batch x n_genes] full expression vector
            z: [batch x 128] context embedding from encoder
            priors: [n_genes x 5] precomputed prior matrix (same for all animals)

        Returns:
            perturb_scores: [batch x n_genes] perturbation score in [0,1]
            transfer_scores: [batch x n_genes] transferability score in [0,1]
            rank_scores: [batch x n_genes] product of both scores
        """
        batch_size, n_genes = expression.shape

        # expand z to match every gene: [batch x n_genes x 128]
        z_expanded = z.unsqueeze(1).expand(-1, n_genes, -1)

        # expand expression to [batch x n_genes x 1] — per-gene scalar
        x_g = expression.unsqueeze(2)

        # expand priors to [batch x n_genes x 5] — same priors for all animals
        p_g = priors.unsqueeze(0).expand(batch_size, -1, -1)

        # concatenate per-gene inputs: [batch x n_genes x 134]
        gene_input = torch.cat([x_g, z_expanded, p_g], dim=2)

        # reshape for shared MLP: [batch*n_genes x 134]
        flat_input = gene_input.reshape(-1, gene_input.shape[-1])
        flat_output = self.mlp(flat_input)  # [batch*n_genes x 2]

        # reshape back: [batch x n_genes x 2]
        output = flat_output.reshape(batch_size, n_genes, 2)

        # split and sigmoid
        perturb_scores = self.sigmoid(output[:, :, 0])   # [batch x n_genes]
        transfer_scores = self.sigmoid(output[:, :, 1])   # [batch x n_genes]

        # multiplicative rank — both must be high
        rank_scores = perturb_scores * transfer_scores

        return perturb_scores, transfer_scores, rank_scores
