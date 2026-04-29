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

  rank_score_g = mean(cohort_rank_score_g_t across timepoints)

  Output: top panel genes by rank_score
```

## How to Run the Full Pipeline

```bash
# 1. Install dependencies
pip install torch numpy scipy pandas pyyaml

# 2. Place data files
cp expression_ef.csv expression_wc.csv metadata.csv project/data/

# 3. Precompute priors and targets (run once)
python -m precompute.compute_priors --config configs/config.yaml
python -m precompute.compute_targets --config configs/config.yaml

# 4. Build the main ranked panel
python -m inference.direct_score --config configs/config.yaml

# 5. Build one panel per measured timepoint
python -m inference.score_by_timepoint --config configs/config.yaml

# 6. Predict P3 and P7 panels
python -m inference.predict_unseen_timepoints --config configs/config.yaml
```

## What the Output Means Biologically

The final table (`top_panel.csv`) contains the top `panel_size` protein-coding
genes ranked by `rank_score`. The default config exports 50 genes.

- **perturbation_score** (per timepoint): How strongly MIA treatment (Poly I:C)
  disrupts this gene's expression compared to saline controls, at each developmental
  stage. High score = large fold-change between conditions. Computed at E15, P0, P13.

- **transferability_score** (per timepoint): How well this gene's expression pattern
  in sorted excitatory neurons tracks its pattern in bulk whole-cortex tissue. High
  score = the neuron-specific signal survives dilution by other cell types. If this is
  low, the gene might be disrupted in neurons but invisible in bulk qPCR.

- **rank_score**: Product of perturbation and transferability, averaged across
  timepoints. Both must be high for a gene to rank well. This is the number you use to
  pick qPCR primers.

A gene with perturbation=0.9 and transferability=0.1 gets rank_score=0.09 (correctly
suppressed — disrupted but undetectable in bulk). A gene with both at 0.8 gets 0.64
(good candidate — disrupted AND detectable).

---

## Predicting Unseen Timepoints (P3, P7)

### Why P3 and P7 matter biologically

The early postnatal window (P0–P14) is a critical period for cortical circuit
refinement — synaptogenesis, pruning, and layer-specific wiring are all active.
In MIA mouse models, ASD-associated behavioral phenotypes (social deficits,
repetitive behaviors) emerge around P21 and later, but the molecular disruptions
that drive them are likely seeded earlier. P3 and P7 may be the window where
MIA-induced transcriptional dysregulation transitions from cell-level perturbation
to circuit-level dysfunction. No RNA-seq data exists at these timepoints in this
dataset — these predictions are hypotheses for future wet-lab validation.

### How trajectory fitting works

Each gene has a perturbation score (MIA vs saline divergence, normalized [0,1]) at
three postnatal-equivalent days: E15 (day −15), P0 (day 0), and P13 (day 13). We
fit both a linear and a quadratic polynomial to these three points, then extrapolate
to day 3 (P3) and day 7 (P7). The quadratic is only preferred if it meaningfully
outperforms the linear fit (R² improvement > 0.05) AND its extrapolated values stay
within biologically plausible bounds (within 2× the maximum observed divergence) —
this prevents runaway extrapolations from noisy trajectories. Predictions are weighted
by fit quality (R²), and genes with R² < 0.5 are excluded from the ranked outputs.
Bootstrap confidence intervals (1000 residual resamples) are computed for the reported
P3/P7 panel genes.

### Confidence caveat

Extrapolating from 3 data points produces wide confidence intervals and is sensitive
to noise in the known timepoints. Quadratic fits interpolate the 3 points exactly
(R²=1.0), which means their bootstrap CIs collapse to a point — this reflects a lack
of residual fit uncertainty, not a lack of extrapolation uncertainty. Treat all P3/P7
predictions as ranked hypotheses, not measurements. Low-R² genes are filtered before
ranking, but even high-R² linear predictions should be validated experimentally.

### How to validate

1. Run RNA-seq (or targeted qPCR) at P3 and P7 in MIA vs saline animals
2. Compute actual log2FC for each gene at each timepoint
3. Rank genes by observed |log2FC| and compare to the predicted rank
4. Primary validation metric: **Spearman correlation between predicted rank and
   actual log2FC rank** — a value > 0.4 would support the trajectory model

### Scripts

| Script | Description |
|--------|-------------|
| `inference/predict_unseen_timepoints.py` | Fits trajectories, predicts P3/P7, generates CSVs and plots |
| `inference/compare_known_vs_predicted.py` | Compares the known panel to predicted lists, flags novel/resolving genes |

```bash
# Run predictions first
python -m inference.predict_unseen_timepoints

# Then compare to known panel
python -m inference.compare_known_vs_predicted
```

### Output files

| File | Contents |
|------|----------|
| `outputs/predicted_P3_P7.csv` | All genes with predicted divergence and fit statistics; CI columns are filled for panel genes |
| `outputs/trajectory_fits.csv` | Fit coefficients per gene — inspect fits without re-running |
| `outputs/predicted_P3_panel.csv` | Top panel genes ranked by predicted divergence × R² at P3 |
| `outputs/predicted_P7_panel.csv` | Top panel genes ranked by predicted divergence × R² at P7 |
| `outputs/trajectory_plots.pdf` | Trajectory plots for the top-10 P3 genes |
| `outputs/comparison_table.csv` | Gene-level boolean table: in_known, in_P3, in_P7, novel, resolves |

---

## Cross-Dataset Integration (Canales 2021 + Kalish 2021)

### Why integrate external datasets

The internal dataset (expression_ef.csv) comes from a single lab, a single MIA protocol, and a single RNA-seq run. Any gene that appears in the internal panel purely due to technical artifacts, batch effects, or lab-specific biology will be silently propagated downstream into qPCR panels. Cross-referencing against independent datasets with different platforms, different MIA protocols, and different tissue contexts (bulk vs scRNA-seq) provides a principled confidence filter. A gene with strong signal in all three independent datasets is much less likely to be a false positive.

### Data sources

| Dataset | GEO Accession | Type | Conditions | Notes |
|---------|--------------|------|------------|-------|
| Internal (this lab) | — | Bulk RNA-seq, sorted EF neurons | Poly I:C vs saline | E15, P0, P13 |
| Canales et al. 2021, eLife | GSE166376 | Bulk RNA-seq, whole cortex | MIA vs saline | E12.5, E14.5, E17.5, P0 |
| Kalish et al. 2021, Nature Neuroscience | GSE161529 | scRNA-seq, fetal cortex | MIA vs PBS | E14.5, E18.5, male + female |

### Integration strategy

Two independent evidence streams are computed and then combined:

**Stream 1 — Canales temporal slopes (bulk RNA-seq)**
- Genes are ranked by `stream1_score = |slope_divergence| × max_abs_log2FC`
- `slope_divergence = slope_MIA − slope_saline` across the Canales developmental timepoints
- Only genes with R² ≥ 0.3 on both MIA and saline slope fits are kept
- This captures genes whose trajectories *diverge* between conditions and stay diverged

**Stream 2 — Kalish cell-type log2FC (scRNA-seq)**
- Counts are pseudobulked by (cell_type, condition, sex, timepoint), then log1p(CPM)-normalized
- `stream2_score = weighted mean |log2FC|` across all pseudobulk groups (weight = n_cells)
- Each gene is also annotated with sex specificity and cell-type breadth
- This captures genes with consistent perturbation across multiple cell types at single-cell resolution

**Combination — reciprocal-rank fusion**
```
combined_score = 0.5 × (1 / rank_stream1) + 0.5 × (1 / rank_stream2)
```
Genes absent from one stream receive a penalty rank of (TOP_N + 1 = 201).
The combined list is confidence-tiered:
- **TIER 1 — both_streams**: present in both stream1 and stream2 top-200 (strongest evidence)
- **TIER 2 — canales_only**: present in stream1 only (replicated in independent bulk RNA-seq)
- **TIER 3 — kalish_only**: present in stream2 only (replicated in scRNA-seq)

### How to run

```bash
# Step 1: Download raw data (run once; Kalish files can be 1-5 GB)
python -m precompute.download_canales
python -m precompute.download_kalish        # use --skip-large to skip files > 500 MB

# Step 2: Process both datasets → outputs/
python -m precompute.process_canales        # → canales_slopes.csv, canales_log2FC.csv
python -m precompute.process_kalish         # → kalish_pseudobulk_log2FC.csv, kalish_gene_summary.csv

# Step 3: Score each stream
python -m integration.stream1_canales_slopes    # → stream1_canales_top200.csv
python -m integration.stream2_kalish_celltype   # → stream2_kalish_top200.csv

# Step 4: Combine and cross-validate
python -m integration.combine_streams           # → generalizable_core_panel.csv
python -m integration.compare_to_existing       # → cross_validation_report.csv
```

### Output files

| File | Contents |
|------|----------|
| `outputs/canales_slopes.csv` | Per-gene linear slope fits (MIA, saline, divergence, R²) from Canales |
| `outputs/canales_log2FC.csv` | Per-gene log2FC at each Canales timepoint |
| `outputs/kalish_pseudobulk_log2FC.csv` | Pseudobulk log2FC per (gene, cell_type, sex, timepoint) from Kalish |
| `outputs/kalish_gene_summary.csv` | Per-gene summary: max log2FC, n_cell_types_significant, sex_specificity |
| `outputs/stream1_canales_top200.csv` | Top-200 genes ranked by stream1_score |
| `outputs/stream2_kalish_top200.csv` | Top-200 genes ranked by stream2_score |
| `outputs/combined_streams_all.csv` | All genes with combined scores and evidence tier |
| `outputs/generalizable_core_panel.csv` | Top-100 genes from combined ranking (confidence-tiered) |
| `outputs/cross_validation_report.csv` | Comparison of core panel vs the existing internal panel |

### Known limitations

- **Normalization mismatch**: Canales and Kalish data are both normalized to log1p(CPM) to match the internal pipeline, but library preparation protocols, sequencing depth, and tissue handling differ across labs.
- **Timepoint mapping**: Canales uses E12.5–P0 while the internal dataset uses E15–P13. The day-mapping (E15 → day −15) is approximate; the slope regression is robust to small mapping errors.
- **Pseudobulk limitations**: Kalish pseudobulk groups with < 10 cells are skipped. For rare cell types, stream2 signal may be absent even for truly perturbed genes.
- **Reciprocal-rank fusion weights**: Equal weights (0.5/0.5) are assumed. If one dataset is higher-quality (more replicates, larger N), consider adjusting.
- **No raw Canales counts**: If the Canales GEO deposit provides only DE tables (not raw counts), slope fitting uses the DE log2FC values directly rather than separate MIA/saline slopes. In this case R2_MIA = R2_saline = R2_divergence (see process_canales.py comments).
