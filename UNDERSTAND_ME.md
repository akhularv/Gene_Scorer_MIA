# MIA Gene Scoring Neural Network

## What This Project Does

This system finds the 25 best genes for detecting maternal immune activation (MIA)
effects in mouse cortex using qPCR. The problem: MIA disrupts gene expression in
specific neuron types (excitatory frontal neurons), but wet-lab screening uses bulk
whole-cortex tissue where the signal is diluted by all other cell types. We need genes
where MIA disruption is both **strong** (high perturbation) and **visible in bulk
tissue** (high transferability). A neural network scores every gene on both criteria
simultaneously, then ranks by the product so both must be high.

## Data You Need

Place three CSV files in `data/`:

### `expression_ef.csv`
- RNA-seq counts from **excitatory frontal neurons** (cell-type sorted)
- Rows = animals, columns = genes (plus metadata columns)
- Timepoints available: E15, P0, P13
- Conditions: saline, polyIC
- ~20+ animals per timepoint-condition group
- Values: raw counts (unnormalized)

### `expression_wc.csv`
- RNA-seq counts from **whole cortex bulk tissue** (all cell types)
- Rows = animals, columns = genes (plus metadata columns)
- Timepoints available: E15, P0, P70, P189
- Conditions: saline, polyIC
- Values: raw counts (unnormalized)
- Gene columns must be identical to `expression_ef.csv` in name and order

### `metadata.csv`
- One row per animal-region combination
- Columns: `animal_id`, `timepoint`, `condition`, `region`
- `timepoint`: one of E15, P0, P13, P70, P189
- `condition`: one of saline, polyIC
- `region`: one of excitatory_frontal, whole_cortex
- Same animal appears twice if it has both EF and WC data

## Architecture Diagram

```
                         STAGE 0: PRECOMPUTE (offline, no learning)
                         ==========================================
  expression_ef.csv ──┐
  expression_wc.csv ──┼──> compute_priors.py  ──> p_g.npy [n_genes x 5]
  metadata.csv ───────┘         │
                                ├──> compute_targets.py ──> perturbation_targets.npy [n_genes x n_tp]
                                └──────────────────────────> transferability_targets.npy [n_genes]


                         STAGE 1: METADATA EMBEDDER
                         ==========================
  timepoint_id ──> Embedding(5, 16) ──┐
  condition_id ──> Embedding(2,  8) ──┼──> concat ──> m [32-dim]
  region_id    ──> Embedding(2,  8) ──┘
                   (bio-informed init for timepoint)


                         STAGE 2: SAMPLE CONTEXT ENCODER
                         ================================
                                            m [32]
                                              │
  expression [n_genes] ──> Linear(n_genes, 2048) ──> LayerNorm ──┐
                                                                  ├─ FiLM(gamma1, beta1) ──> GELU
                                                                  │
                                                   ──> Linear(2048, 512) ──> LayerNorm ──┐
                                                                                          ├─ FiLM(gamma2, beta2) ──> GELU
                                                                                          │
                                                                          ──> Linear(512, 128) ──> z [128-dim]


                         STAGE 3: GENE SCORER (shared weights, runs per gene)
                         ====================================================
  For each gene g:
    x_g [1]   ── raw expression scalar ──┐
    z   [128] ── context from encoder  ──┼──> concat [134] ──> MLP(134→64→32→2) ──> sigmoid
    p_g [5]   ── precomputed prior     ──┘                          │
                                                          ┌─────────┴──────────┐
                                                   perturbation_g      transferability_g
                                                          │                    │
                                                          └────── multiply ────┘
                                                                    │
                                                              rank_score_g


                         STAGE 4: COHORT AGGREGATION (inference only)
                         =============================================
  For each gene g, for each timepoint t:
    cohort_rank_score_g_t = median(rank_score_g across polyIC animals at t)

  global_rank_score_g = mean(cohort_rank_score_g_t across timepoints)

  Output: top-25 genes by global_rank_score
```

## Training Phases

| Phase | Steps     | L_transfer | L_perturb | L_temporal | Goal                                    |
|-------|-----------|------------|-----------|------------|-----------------------------------------|
| 1     | 0-20%     | 1.0        | 0.0       | 0.0        | Encoder warm-up: z clusters by timepoint|
| 2     | 20-60%    | 1.0        | 0.3       | 0.05       | Introduce perturbation signal gently    |
| 3     | 60-100%   | 1.0        | 1.0       | 0.1        | Full training, all losses active        |

**Stopping criterion**: Top-25 gene ranking Spearman > 0.95 for 5 consecutive epochs.

## How to Run the Full Pipeline

```bash
# 1. Install dependencies
pip install torch numpy scipy pandas pyyaml

# 2. Place data files
cp expression_ef.csv expression_wc.csv metadata.csv project/data/

# 3. Precompute priors and targets (run once)
cd project
python -m precompute.compute_priors --config configs/config.yaml
python -m precompute.compute_targets --config configs/config.yaml

# 4. Train the model
python -m training.train --config configs/config.yaml

# 5. Score genes and export panel
python -m inference.score_genes --config configs/config.yaml --checkpoint outputs/best_model.pt
python -m inference.export_panel --config configs/config.yaml --scores outputs/gene_scores.npz

# 6. Run tests (optional, uses synthetic data)
python -m pytest tests/test_model.py -v
```

## What the Output Means Biologically

The final table (`top25_panel.csv`) contains 25 genes ranked by `global_rank_score`.

- **perturbation_score** (per timepoint): How strongly MIA treatment (Poly I:C)
  disrupts this gene's expression compared to saline controls, at each developmental
  stage. High score = large fold-change between conditions. Computed at E15, P0, P13.

- **transferability_score** (per timepoint): How well this gene's expression pattern
  in sorted excitatory neurons tracks its pattern in bulk whole-cortex tissue. High
  score = the neuron-specific signal survives dilution by other cell types. If this is
  low, the gene might be disrupted in neurons but invisible in bulk qPCR.

- **global_rank_score**: Product of perturbation and transferability, averaged across
  timepoints. Both must be high for a gene to rank well. This is the number you use to
  pick qPCR primers.

A gene with perturbation=0.9 and transferability=0.1 gets rank_score=0.09 (correctly
suppressed — disrupted but undetectable in bulk). A gene with both at 0.8 gets 0.64
(good candidate — disrupted AND detectable).
