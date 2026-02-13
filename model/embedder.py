# - Produces 32-dim metadata vector from timepoint, condition, and region IDs
# - Timepoint embedding initialized from log-scaled postnatal ages (biological prior)
# - Condition and region embeddings randomly initialized
# - Output feeds exclusively into FiLM generators in the encoder

import torch
import torch.nn as nn
import numpy as np


class MetadataEmbedder(nn.Module):
    """Embeds biological metadata (timepoint, condition, region) into a 32-dim vector.

    Timepoint embedding uses biologically-informed initialization: postnatal ages
    projected to 16-dim so the embedding space starts with correct developmental
    spacing. Backprop fine-tunes from there.
    """

    def __init__(self, timepoint_ages: list[float]) -> None:
        super().__init__()

        # timepoint gets 16 dims — most biological variance lives here
        self.timepoint_emb = nn.Embedding(5, 16)
        # condition (saline/polyIC) and region (EF/WC) get 8 dims each
        self.condition_emb = nn.Embedding(2, 8)
        self.region_emb = nn.Embedding(2, 8)

        # biologically-informed init for timepoint embedding
        self._init_timepoint_embedding(timepoint_ages)

    def _init_timepoint_embedding(self, ages: list[float]) -> None:
        """Initialize timepoint embeddings from log-scaled postnatal ages.

        Maps raw ages through log-scale (handles the huge range from E15 to P189),
        then projects to 16-dim via a fixed random projection. This gives the
        optimizer a head start: nearby developmental stages start nearby in
        embedding space.
        """
        ages_arr = np.array(ages, dtype=np.float32)
        # log-scale: shift so all positive, then log
        shifted = ages_arr - ages_arr.min() + 1.0  # ensures all > 0
        log_ages = np.log(shifted)
        # normalize to unit range
        log_ages = (log_ages - log_ages.min()) / (log_ages.max() - log_ages.min() + 1e-8)

        # project scalar to 16-dim via fixed random basis
        rng = np.random.RandomState(42)  # reproducible init
        proj = rng.randn(1, 16).astype(np.float32) * 0.1
        # each timepoint gets its log-age scaled along the projection
        init_weights = log_ages[:, None] * proj  # [5 x 16]

        with torch.no_grad():
            self.timepoint_emb.weight.copy_(torch.from_numpy(init_weights))

    def forward(
        self,
        timepoint_ids: torch.Tensor,
        condition_ids: torch.Tensor,
        region_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embed metadata and concatenate to 32-dim vector.

        Args:
            timepoint_ids: [batch] int tensor, values 0-4
            condition_ids: [batch] int tensor, values 0-1
            region_ids: [batch] int tensor, values 0-1

        Returns:
            m: [batch x 32] metadata vector
        """
        t = self.timepoint_emb(timepoint_ids)   # [batch x 16]
        c = self.condition_emb(condition_ids)    # [batch x 8]
        r = self.region_emb(region_ids)          # [batch x 8]
        m = torch.cat([t, c, r], dim=-1)         # [batch x 32]
        return m
