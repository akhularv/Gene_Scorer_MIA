# - Implements Feature-wise Linear Modulation (FiLM) layer
# - Metadata vector generates per-feature scale (gamma) and shift (beta)
# - Modulates normalized hidden states without concatenating metadata to input
# - This lets biological context condition encoding without drowning gene signal

import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """Generates FiLM parameters (gamma, beta) from a conditioning vector.

    Given metadata vector m, produces scale and shift vectors that modulate
    a hidden representation element-wise. This is better than concatenation
    because it conditions every feature independently rather than adding
    metadata as extra dimensions that compete with gene expression signal.
    """

    def __init__(self, cond_dim: int, hidden_dim: int) -> None:
        """
        Args:
            cond_dim: dimension of conditioning vector (metadata, 32)
            hidden_dim: dimension of hidden state to modulate
        """
        super().__init__()
        # single linear produces both gamma and beta, split after
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)

        # init gamma near 1 and beta near 0 so FiLM starts as identity
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)

    def forward(self, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate scale and shift from metadata.

        Args:
            m: [batch x cond_dim] conditioning vector

        Returns:
            gamma: [batch x hidden_dim] multiplicative scale
            beta: [batch x hidden_dim] additive shift
        """
        params = self.proj(m)  # [batch x hidden_dim*2]
        gamma, beta = params.chunk(2, dim=-1)
        # offset gamma by 1 so default modulation is identity (1*h + 0)
        gamma = gamma + 1.0
        return gamma, beta


class FiLMLayer(nn.Module):
    """One FiLM-conditioned layer: Linear -> LayerNorm -> FiLM -> GELU.

    The key insight: LayerNorm removes magnitude info from the hidden state,
    then FiLM re-injects context-appropriate magnitude via gamma/beta. This
    lets the same encoder process E15 and P189 samples differently without
    needing separate networks.
    """

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.film = FiLMGenerator(cond_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Forward pass with FiLM conditioning.

        Args:
            h: [batch x in_dim] hidden state
            m: [batch x cond_dim] metadata vector

        Returns:
            [batch x out_dim] modulated hidden state
        """
        h = self.linear(h)
        h = self.norm(h)
        gamma, beta = self.film(m)
        h = self.act(gamma * h + beta)  # FiLM modulation then activation
        return h
