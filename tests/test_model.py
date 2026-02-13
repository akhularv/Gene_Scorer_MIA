# - Tests for all model components using synthetic data
# - Verifies shapes, value ranges, and architectural invariants
# - No real data needed — everything is generated in fixtures
# - Run with: python -m pytest tests/test_model.py -v

import numpy as np
import pytest
import torch

from model.embedder import MetadataEmbedder
from model.encoder import SampleContextEncoder
from model.gene_scorer import GeneScorer
from training.losses import (
    perturbation_loss,
    temporal_consistency_loss,
    total_loss,
    transferability_loss,
)
from training.train import MIAModel


# --- Fixtures ---

N_GENES = 500  # small for fast tests
BATCH_SIZE = 8
CONTEXT_DIM = 128
METADATA_DIM = 32
TIMEPOINT_AGES = [-5.0, 0.0, 13.0, 70.0, 189.0]


@pytest.fixture
def synthetic_priors() -> np.ndarray:
    """Fake p_g matrix with correct shape and range."""
    rng = np.random.RandomState(0)
    p_g = rng.rand(N_GENES, 5).astype(np.float32)
    return p_g


@pytest.fixture
def synthetic_config() -> dict:
    """Minimal config for model construction."""
    return {
        "timepoint_ages": TIMEPOINT_AGES,
        "metadata_dim": METADATA_DIM,
        "context_dim": CONTEXT_DIM,
        "hidden_dims": {
            "encoder_h1": 2048,
            "encoder_h2": 512,
            "scorer_h1": 64,
            "scorer_h2": 32,
        },
    }


@pytest.fixture
def synthetic_batch() -> dict[str, torch.Tensor]:
    """Synthetic batch matching expected data format."""
    rng = np.random.RandomState(42)
    return {
        "expression": torch.from_numpy(
            rng.rand(BATCH_SIZE, N_GENES).astype(np.float32)
        ),
        "timepoint_id": torch.randint(0, 5, (BATCH_SIZE,)),
        "condition_id": torch.randint(0, 2, (BATCH_SIZE,)),
        "region_id": torch.randint(0, 2, (BATCH_SIZE,)),
    }


# --- Tests ---


class TestPriors:
    """Verify precomputed prior matrix properties."""

    def test_shape(self, synthetic_priors: np.ndarray) -> None:
        """p_g must be [n_genes x 5]."""
        assert synthetic_priors.shape == (N_GENES, 5)

    def test_range(self, synthetic_priors: np.ndarray) -> None:
        """All values must be in [0, 1]."""
        assert synthetic_priors.min() >= 0.0
        assert synthetic_priors.max() <= 1.0


class TestMetadataEmbedder:
    """Verify embedder produces correct output shape."""

    def test_output_dim(self) -> None:
        """Embedder must produce 32-dim vector."""
        emb = MetadataEmbedder(TIMEPOINT_AGES)
        tp = torch.tensor([0, 1, 2])
        cond = torch.tensor([0, 1, 0])
        reg = torch.tensor([1, 0, 1])
        m = emb(tp, cond, reg)
        assert m.shape == (3, METADATA_DIM)

    def test_single_sample(self) -> None:
        """Works for single-sample batch."""
        emb = MetadataEmbedder(TIMEPOINT_AGES)
        m = emb(torch.tensor([3]), torch.tensor([1]), torch.tensor([0]))
        assert m.shape == (1, METADATA_DIM)


class TestEncoder:
    """Verify encoder compresses expression to z [128]."""

    def test_output_shape(self) -> None:
        """Encoder must output [batch x 128]."""
        enc = SampleContextEncoder(N_GENES)
        expr = torch.randn(BATCH_SIZE, N_GENES)
        m = torch.randn(BATCH_SIZE, METADATA_DIM)
        z = enc(expr, m)
        assert z.shape == (BATCH_SIZE, CONTEXT_DIM)

    def test_single_sample(self) -> None:
        """Works for a single expression vector."""
        enc = SampleContextEncoder(N_GENES)
        expr = torch.randn(1, N_GENES)
        m = torch.randn(1, METADATA_DIM)
        z = enc(expr, m)
        assert z.shape == (1, CONTEXT_DIM)


class TestGeneScorer:
    """Verify scorer outputs correct shapes and ranges."""

    def test_output_shapes(self, synthetic_priors: np.ndarray) -> None:
        """All three outputs must be [batch x n_genes]."""
        scorer = GeneScorer(context_dim=CONTEXT_DIM)
        expr = torch.randn(BATCH_SIZE, N_GENES)
        z = torch.randn(BATCH_SIZE, CONTEXT_DIM)
        priors = torch.from_numpy(synthetic_priors)

        ps, ts, rs = scorer(expr, z, priors)
        assert ps.shape == (BATCH_SIZE, N_GENES)
        assert ts.shape == (BATCH_SIZE, N_GENES)
        assert rs.shape == (BATCH_SIZE, N_GENES)

    def test_scores_in_unit_range(self, synthetic_priors: np.ndarray) -> None:
        """Perturbation and transferability scores must be in [0, 1]."""
        scorer = GeneScorer(context_dim=CONTEXT_DIM)
        expr = torch.randn(BATCH_SIZE, N_GENES)
        z = torch.randn(BATCH_SIZE, CONTEXT_DIM)
        priors = torch.from_numpy(synthetic_priors)

        ps, ts, rs = scorer(expr, z, priors)
        assert ps.min() >= 0.0 and ps.max() <= 1.0
        assert ts.min() >= 0.0 and ts.max() <= 1.0

    def test_rank_is_product(self, synthetic_priors: np.ndarray) -> None:
        """rank_score must equal perturbation * transferability."""
        scorer = GeneScorer(context_dim=CONTEXT_DIM)
        expr = torch.randn(BATCH_SIZE, N_GENES)
        z = torch.randn(BATCH_SIZE, CONTEXT_DIM)
        priors = torch.from_numpy(synthetic_priors)

        ps, ts, rs = scorer(expr, z, priors)
        expected = ps * ts
        assert torch.allclose(rs, expected, atol=1e-6)


class TestLosses:
    """Verify loss term behavior."""

    def test_temporal_zero_single_timepoint(self) -> None:
        """L_temporal must be zero when batch has only one timepoint."""
        rank = torch.rand(BATCH_SIZE, N_GENES)
        tp_ids = torch.zeros(BATCH_SIZE, dtype=torch.long)  # all same timepoint
        cond_ids = torch.ones(BATCH_SIZE, dtype=torch.long)  # all polyIC
        loss = temporal_consistency_loss(rank, tp_ids, cond_ids)
        assert loss.item() == 0.0

    def test_temporal_nonzero_multiple_timepoints(self) -> None:
        """L_temporal should be > 0 when rank varies across timepoints."""
        torch.manual_seed(42)
        n = 10
        rank = torch.rand(n, N_GENES)
        # assign different timepoints
        tp_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        cond_ids = torch.ones(n, dtype=torch.long)
        loss = temporal_consistency_loss(rank, tp_ids, cond_ids)
        # with random scores, variance across timepoints should be nonzero
        assert loss.item() > 0.0

    def test_perturb_zero_for_saline(self) -> None:
        """Perturbation loss must be zero when all samples are saline."""
        ps = torch.rand(BATCH_SIZE, N_GENES)
        cond_ids = torch.zeros(BATCH_SIZE, dtype=torch.long)  # all saline
        tp_ids = torch.zeros(BATCH_SIZE, dtype=torch.long)
        target_matrix = torch.rand(N_GENES, 3)
        loss = perturbation_loss(ps, None, cond_ids, tp_ids, target_matrix)
        assert loss.item() == 0.0

    def test_transfer_loss_positive(self) -> None:
        """Transferability loss should be > 0 with mismatched predictions."""
        ts = torch.rand(BATCH_SIZE, N_GENES)
        targets = torch.rand(N_GENES)
        loss = transferability_loss(ts, targets)
        assert loss.item() > 0.0


class TestFullForward:
    """End-to-end forward pass with synthetic data."""

    def test_forward_no_error(
        self, synthetic_config: dict, synthetic_priors: np.ndarray,
        synthetic_batch: dict[str, torch.Tensor]
    ) -> None:
        """Full model forward pass should run without error."""
        model = MIAModel(N_GENES, synthetic_config)
        priors = torch.from_numpy(synthetic_priors)

        ps, ts, rs = model(
            synthetic_batch["expression"],
            synthetic_batch["timepoint_id"],
            synthetic_batch["condition_id"],
            synthetic_batch["region_id"],
            priors,
        )

        # check shapes
        assert ps.shape == (BATCH_SIZE, N_GENES)
        assert ts.shape == (BATCH_SIZE, N_GENES)
        assert rs.shape == (BATCH_SIZE, N_GENES)

        # check ranges
        assert ps.min() >= 0.0 and ps.max() <= 1.0
        assert ts.min() >= 0.0 and ts.max() <= 1.0
        assert rs.min() >= 0.0 and rs.max() <= 1.0

    def test_forward_gradient_flows(
        self, synthetic_config: dict, synthetic_priors: np.ndarray,
        synthetic_batch: dict[str, torch.Tensor]
    ) -> None:
        """Gradients should flow through the full model."""
        model = MIAModel(N_GENES, synthetic_config)
        priors = torch.from_numpy(synthetic_priors)

        ps, ts, rs = model(
            synthetic_batch["expression"],
            synthetic_batch["timepoint_id"],
            synthetic_batch["condition_id"],
            synthetic_batch["region_id"],
            priors,
        )

        # backward on a simple loss
        loss = ps.mean() + ts.mean()
        loss.backward()

        # check that encoder weights got gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients flowed through the model"
