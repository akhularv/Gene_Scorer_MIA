# - Encodes a raw expression vector [n_genes] into context embedding z [128]
# - Two FiLM-conditioned layers let metadata modulate gene signal processing
# - Final clean projection (no FiLM) produces context vector for gene scorer
# - One forward pass per animal; z is shared across all genes for that animal

import torch
import torch.nn as nn

from model.film import FiLMLayer


class SampleContextEncoder(nn.Module):
    """Compresses a full-transcriptome expression vector into a 128-dim context.

    Uses FiLM conditioning so the same encoder handles all timepoints and regions.
    The metadata doesn't get concatenated — it modulates the hidden state via
    learned scale/shift. This preserves the full n_genes input bandwidth for
    gene expression signal.
    """

    def __init__(
        self,
        n_genes: int,
        metadata_dim: int = 32,
        context_dim: int = 128,
        h1_dim: int = 2048,
        h2_dim: int = 512,
    ) -> None:
        super().__init__()
        # two FiLM layers progressively compress the transcriptome
        self.film1 = FiLMLayer(n_genes, h1_dim, metadata_dim)
        self.film2 = FiLMLayer(h1_dim, h2_dim, metadata_dim)
        # final projection has no FiLM — produces raw context embedding
        self.proj = nn.Linear(h2_dim, context_dim)

    def forward(
        self, expression: torch.Tensor, m: torch.Tensor
    ) -> torch.Tensor:
        """Encode expression vector conditioned on metadata.

        Args:
            expression: [batch x n_genes] raw (normalized) expression values
            m: [batch x 32] metadata vector from embedder

        Returns:
            z: [batch x 128] context embedding
        """
        h = self.film1(expression, m)  # [batch x 2048]
        h = self.film2(h, m)           # [batch x 512]
        z = self.proj(h)               # [batch x 128]
        return z
