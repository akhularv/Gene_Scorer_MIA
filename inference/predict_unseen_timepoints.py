"""
predict_unseen_timepoints.py

- Loads perturbation divergence scores at E15, P0, P13 from all_genes_ranked.csv
  (these are normalized [0,1] absolute differences in log1p(CPM) between polyIC
  and saline — the best per-gene proxy for slope divergence from the precompute step)
- For each gene: fits both a linear and quadratic trajectory to the 3 known timepoints
  mapped to postnatal-equivalent days: E15→-15, P0→0, P13→13
- Selection rule: use quadratic only if it meaningfully outperforms linear (R²_quad >
  R²_lin + 0.05) AND the extrapolated values stay within biologically plausible bounds.
  Note: with exactly 3 data points, quadratic always fits perfectly (R²=1.0), so the
  real gate is the bounds check preventing runaway extrapolations.
- Extrapolates to P3 (day 3) and P7 (day 7)
- Bootstraps 95% CIs at each predicted timepoint via residual resampling (1000 iters)
- Ranks genes by predicted_divergence * R²  (penalizes low-confidence fits)
- Genes with R² < 0.5 are excluded from the top-25 lists
- Outputs: predicted_P3_P7.csv, top25_predicted_P3.csv, top25_predicted_P7.csv,
           trajectory_fits.csv, trajectory_plots.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for PDF on any machine
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import yaml


# postnatal-equivalent days for each timepoint
KNOWN_DAYS   = [-15, 0, 13]   # E15, P0, P13 — actual data exists here
PREDICT_DAYS = [3, 7]          # P3, P7 — extrapolation targets

N_BOOTSTRAP     = 1000
RANDOM_SEED     = 42
LOW_R2_CUTOFF      = 0.5   # genes below this are excluded from ranked top lists
HIGH_EXPR_THRESHOLD = 500  # minimum mean raw count across all EF samples to be included


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard coefficient of determination. Clamped to 0 (can't be negative)."""
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 1.0  # constant gene — any horizontal line fits perfectly
    ss_res = np.sum((y_true - y_pred) ** 2)
    return max(0.0, 1.0 - ss_res / ss_tot)


def fit_gene(y: np.ndarray, x: list = None):
    """Fit linear + quadratic trajectory. Return the better-fitting model.

    Selection rule:
        Use quadratic if R²_quad > R²_lin + 0.05  AND
        predictions at P3/P7 stay within [-2, 2] * max(|y|).
        Otherwise, use linear (safer extrapolation with 3 points).

    Returns:
        fit_type   (str)         "linear" or "quadratic"
        coeffs     (np.ndarray)  polynomial coefficients, highest-degree first
        r2_linear  (float)       R² of the linear fit (used as confidence metric)
        r2_selected(float)       R² of whichever model was chosen
        preds      (dict)        {postnatal_day: predicted_value}
    """
    if x is None:
        x = KNOWN_DAYS
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # --- linear (degree 1) ---
    lin_coeffs = np.polyfit(x, y, 1)
    r2_lin = r_squared(y, np.polyval(lin_coeffs, x))

    # --- quadratic (degree 2) — always R²=1.0 with exactly 3 points ---
    quad_coeffs = np.polyfit(x, y, 2)
    r2_quad = r_squared(y, np.polyval(quad_coeffs, x))  # will be ~1.0

    preds_lin  = {d: float(np.polyval(lin_coeffs,  d)) for d in PREDICT_DAYS}
    preds_quad = {d: float(np.polyval(quad_coeffs, d)) for d in PREDICT_DAYS}

    # bounds check: reject quadratic if it predicts biologically absurd values
    bound = 2.0 * max(float(np.abs(y).max()), 1e-6)
    quad_in_bounds = all(abs(preds_quad[d]) <= bound for d in PREDICT_DAYS)

    use_quad = (r2_quad > r2_lin + 0.05) and quad_in_bounds

    if use_quad:
        return "quadratic", quad_coeffs, r2_lin, r2_quad, preds_quad
    else:
        return "linear", lin_coeffs, r2_lin, r2_lin, preds_lin


def residual_bootstrap_ci(
    y: np.ndarray,
    fit_type: str,
    coeffs: np.ndarray,
    x: list = None,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict:
    """95% CIs at P3 and P7 via residual bootstrap.

    Procedure:
      1. Compute residuals from the fitted model
      2. For each of n_boot iterations: resample residuals with replacement,
         add to fitted values, refit same model type, predict at P3/P7
      3. Return [2.5th, 97.5th] percentiles

    Note: quadratic residuals are ~0 (perfect 3-point fit), so quadratic CIs
    will collapse to a point. This correctly reflects no residual fit uncertainty
    (the uncertainty is in extrapolation trend, not in interpolation accuracy).

    Returns:
        {postnatal_day: (ci_lower, ci_upper)}
    """
    if x is None:
        x = KNOWN_DAYS
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    degree = 2 if fit_type == "quadratic" else 1
    rng = np.random.default_rng(seed)

    y_hat = np.polyval(coeffs, x)
    residuals = y - y_hat  # will be ~0 for quadratic

    boot_preds = {d: [] for d in PREDICT_DAYS}
    for _ in range(n_boot):
        resampled_resid = rng.choice(residuals, size=len(residuals), replace=True)
        y_boot = y_hat + resampled_resid
        boot_coeffs = np.polyfit(x, y_boot, degree)
        for d in PREDICT_DAYS:
            boot_preds[d].append(float(np.polyval(boot_coeffs, d)))

    return {
        d: (float(np.percentile(boot_preds[d], 2.5)),
            float(np.percentile(boot_preds[d], 97.5)))
        for d in PREDICT_DAYS
    }


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_gene_trajectory(
    ax,
    gene_name: str,
    y_known: np.ndarray,
    pred_p3: float,
    pred_p7: float,
    ci_p3: tuple,
    ci_p7: tuple,
    fit_type: str,
    coeffs: np.ndarray,
    r2: float,
):
    """Draw one gene's trajectory subplot."""
    x_known = np.array(KNOWN_DAYS, dtype=float)

    # dashed fitted line spanning the full known range
    x_line = np.linspace(-15, 13, 300)
    ax.plot(x_line, np.polyval(coeffs, x_line),
            "k--", linewidth=1.2, alpha=0.65, zorder=1, label="Fit")

    # shaded CI band between P3 and P7
    ax.fill_between(
        [3, 7],
        [ci_p3[0], ci_p7[0]],
        [ci_p3[1], ci_p7[1]],
        alpha=0.18, color="steelblue", zorder=2, label="95% CI"
    )

    # solid dots — known timepoints
    ax.scatter(x_known, y_known,
               color="black", s=55, zorder=5, label="Known")

    # open circles with error bars — predicted timepoints
    for day, pred, ci, lbl in [(3, pred_p3, ci_p3, "P3"), (7, pred_p7, ci_p7, "P7")]:
        ax.errorbar(
            day, pred,
            yerr=[[pred - ci[0]], [ci[1] - pred]],
            fmt="o", mfc="white", mec="steelblue", ecolor="steelblue",
            capsize=4, markersize=7, zorder=6, label=f"{lbl} pred"
        )

    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_title(f"{gene_name}  |  {fit_type}, R²={r2:.2f}", fontsize=8.5)
    ax.set_xlabel("Postnatal day", fontsize=8)
    ax.set_ylabel("Perturbation score", fontsize=8)
    ax.set_xticks([-15, 0, 3, 7, 13])
    ax.set_xticklabels(["E15\n(−15)", "P0\n(0)", "P3\n(3)", "P7\n(7)", "P13\n(13)"], fontsize=7)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Gene filter
# ---------------------------------------------------------------------------

def load_high_expr_genes(ef_path: str, threshold: float = HIGH_EXPR_THRESHOLD) -> set:
    """Return genes whose mean raw count across all EF samples meets the threshold.

    expression_ef.csv columns: animal_id, timepoint, condition, [gene columns...]
    We average over every sample row regardless of timepoint/condition.
    """
    ef = pd.read_csv(ef_path)
    meta_cols = {"animal_id", "timepoint", "condition", "region"}
    gene_cols = [c for c in ef.columns if c not in meta_cols]
    mean_counts = ef[gene_cols].mean()
    return set(mean_counts[mean_counts >= threshold].index)


def load_protein_coding_genes(raw_counts_path: str) -> set:
    """Return protein-coding, non-mitochondrial gene names from all_counts.csv.

    all_counts.csv layout: col 0 = R row index, cols 1-7 = gene metadata.
    After index_col=0, the remaining columns are named by the header row:
    gene_id, chr, start, end, strand, gene_type, gene_name.
    """
    raw = pd.read_csv(raw_counts_path, index_col=0, usecols=range(8))
    protein_coding = raw[raw["gene_type"] == "protein_coding"]["gene_name"]
    # drop mitochondrial genes (mt- prefix, case-insensitive)
    protein_coding = protein_coding[~protein_coding.str.lower().str.startswith("mt-")]
    # drop predicted gene models (Gm prefix, case-insensitive)
    protein_coding = protein_coding[~protein_coding.str.lower().str.startswith("gm")]
    return set(protein_coding)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict gene perturbation at P3 and P7 via trajectory extrapolation"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--panel-size", type=int, default=None,
                        help="Number of top genes to output (default: panel_size from config)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = config["output_dir"]
    panel_size = args.panel_size or config.get("panel_size", 25)

    # --- load known perturbation scores ---
    ranked_path = os.path.join(output_dir, "all_genes_ranked.csv")
    if not os.path.exists(ranked_path):
        raise FileNotFoundError(
            f"Missing required input: {ranked_path}\n"
            f"Run `python -m inference.direct_score` to generate it."
        )

    df = pd.read_csv(ranked_path, index_col=0)
    print(f"Loaded {len(df)} genes from {ranked_path}")

    # --- filter to protein-coding, non-mitochondrial genes ---
    raw_counts_path = os.path.join("MIA_Data", "all_counts.csv")
    if not os.path.exists(raw_counts_path):
        raise FileNotFoundError(
            f"Missing required input: {raw_counts_path}\n"
            f"Needed to identify protein-coding genes."
        )
    protein_coding = load_protein_coding_genes(raw_counts_path)
    n_before = len(df)
    df = df[df["gene_name"].isin(protein_coding)].copy()
    print(f"Kept {len(df)} protein-coding, non-mitochondrial, non-Gm genes "
          f"(removed {n_before - len(df)})")

    # --- filter to highly-expressed genes (mean raw count >= HIGH_EXPR_THRESHOLD) ---
    ef_path = os.path.join(config["data_dir"], "expression_ef.csv")
    if not os.path.exists(ef_path):
        raise FileNotFoundError(
            f"Missing required input: {ef_path}\n"
            f"Needed to compute mean expression per gene."
        )
    high_expr = load_high_expr_genes(ef_path, HIGH_EXPR_THRESHOLD)
    n_before = len(df)
    df = df[df["gene_name"].isin(high_expr)].copy()
    print(f"Kept {len(df)} highly-expressed genes (mean raw count ≥ {HIGH_EXPR_THRESHOLD}, "
          f"removed {n_before - len(df)})")

    # confirm the columns we need are present
    needed = ["gene_name", "perturbation_E15", "perturbation_P0", "perturbation_P13"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns not found in {ranked_path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # --- fit trajectories and bootstrap CIs for every gene ---
    print(f"Fitting trajectories + bootstrapping CIs ({N_BOOTSTRAP} iters) ...")

    pred_records = []   # → predicted_P3_P7.csv
    fit_records  = []   # → trajectory_fits.csv

    for _, row in df.iterrows():
        gene = row["gene_name"]
        y = np.array([
            row["perturbation_E15"],
            row["perturbation_P0"],
            row["perturbation_P13"],
        ], dtype=float)

        fit_type, coeffs, r2_lin, r2_sel, preds = fit_gene(y)
        ci = residual_bootstrap_ci(y, fit_type, coeffs)

        pred_records.append({
            "gene_name":               gene,
            "slope_divergence_E15":    float(y[0]),
            "slope_divergence_P0":     float(y[1]),
            "slope_divergence_P13":    float(y[2]),
            "predicted_divergence_P3": preds[3],
            "predicted_divergence_P7": preds[7],
            "fit_type":                fit_type,
            "R2":                      r2_sel,
            "R2_linear":               r2_lin,   # always <1, useful confidence metric
            "CI_lower_P3":             ci[3][0],
            "CI_upper_P3":             ci[3][1],
            "CI_lower_P7":             ci[7][0],
            "CI_upper_P7":             ci[7][1],
        })

        # save fit coefficients so fits are inspectable without re-running
        fit_records.append({
            "gene_name": gene,
            "fit_type":  fit_type,
            "R2":        r2_sel,
            "R2_linear": r2_lin,
            # coeff_0 = highest-degree term (e.g., 'a' in ax²+bx+c)
            "coeff_0":   float(coeffs[0]),
            "coeff_1":   float(coeffs[1]),
            "coeff_2":   float(coeffs[2]) if len(coeffs) > 2 else np.nan,
        })

    results = pd.DataFrame(pred_records)
    fits    = pd.DataFrame(fit_records)

    # --- ranking ---
    # negative predicted_divergence = gene converges toward baseline → ranks last naturally
    results["rank_score_P3"] = results["predicted_divergence_P3"] * results["R2"]
    results["rank_score_P7"] = results["predicted_divergence_P7"] * results["R2"]

    # filter out low-confidence genes before building top lists
    high_conf = results[results["R2"] >= LOW_R2_CUTOFF].copy()
    n_low_conf = len(results) - len(high_conf)

    top_p3 = (high_conf
              .sort_values(["rank_score_P3", "R2"], ascending=False)
              .head(panel_size))
    top_p7 = (high_conf
              .sort_values(["rank_score_P7", "R2"], ascending=False)
              .head(panel_size))

    # --- save CSVs ---
    pred_path  = os.path.join(output_dir, "predicted_P3_P7.csv")
    fits_path  = os.path.join(output_dir, "trajectory_fits.csv")
    top_p3_path = os.path.join(output_dir, "top25_predicted_P3.csv")
    top_p7_path = os.path.join(output_dir, "top25_predicted_P7.csv")
    pdf_path   = os.path.join(output_dir, "trajectory_plots.pdf")

    results.to_csv(pred_path,   index=False)
    fits.to_csv(fits_path,      index=False)
    top_p3.to_csv(top_p3_path,  index=False)
    top_p7.to_csv(top_p7_path,  index=False)

    print(f"Saved predictions   → {pred_path}")
    print(f"Saved fit params    → {fits_path}")
    print(f"Saved top-{panel_size} at P3 → {top_p3_path}")
    print(f"Saved top-{panel_size} at P7 → {top_p7_path}")

    # --- trajectory plots for the top-10 P3 genes ---
    top10 = top_p3.head(10)
    # pull their full row data from results for plotting
    res_idx  = results.set_index("gene_name")
    fits_idx = fits.set_index("gene_name")

    fig, axes = plt.subplots(5, 2, figsize=(11, 16))
    axes_flat = axes.flatten()

    for i, (_, trow) in enumerate(top10.iterrows()):
        gene = trow["gene_name"]
        rr = res_idx.loc[gene]
        fr = fits_idx.loc[gene]

        y_known = np.array([
            rr["slope_divergence_E15"],
            rr["slope_divergence_P0"],
            rr["slope_divergence_P13"],
        ])
        # reconstruct coefficient array
        if fr["fit_type"] == "quadratic":
            coeffs = np.array([fr["coeff_0"], fr["coeff_1"], fr["coeff_2"]])
        else:
            coeffs = np.array([fr["coeff_0"], fr["coeff_1"]])

        plot_gene_trajectory(
            ax=axes_flat[i],
            gene_name=gene,
            y_known=y_known,
            pred_p3=rr["predicted_divergence_P3"],
            pred_p7=rr["predicted_divergence_P7"],
            ci_p3=(rr["CI_lower_P3"], rr["CI_upper_P3"]),
            ci_p7=(rr["CI_lower_P7"], rr["CI_upper_P7"]),
            fit_type=fr["fit_type"],
            coeffs=coeffs,
            r2=rr["R2"],
        )
        if i == 0:
            axes_flat[0].legend(fontsize=7, loc="upper right")

    plt.suptitle(
        "Top-10 Predicted Genes at P3 — Developmental Trajectories\n"
        "(solid = known, open = predicted, dashed = fit, shaded = 95% CI)",
        fontsize=10, y=1.01
    )
    plt.tight_layout()

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plots         → {pdf_path}")

    # --- stdout summary ---
    lin_r2_vals = results.loc[results["fit_type"] == "linear", "R2"]

    print("\n" + "=" * 65)
    print(f"Top 5 predicted genes at P3: "
          f"{', '.join(top_p3['gene_name'].head(5).tolist())}")
    print(f"Top 5 predicted genes at P7: "
          f"{', '.join(top_p7['gene_name'].head(5).tolist())}")
    print(f"Mean R2 of linear fits:      {lin_r2_vals.mean():.3f}")
    print(f"Genes with low confidence (R2 < {LOW_R2_CUTOFF}): "
          f"{n_low_conf} genes, excluded from top lists")
    fit_counts = results["fit_type"].value_counts().to_dict()
    print(f"Fit types used:              {fit_counts}")
    print("=" * 65)


if __name__ == "__main__":
    main()
