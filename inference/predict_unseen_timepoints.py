"""Predict P3 and P7 perturbation scores from E15, P0, and P13."""

import argparse
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "gene_scorer_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import yaml

from project_paths import resolve_path


KNOWN_DAYS = [-15, 0, 13]
PREDICT_DAYS = [3, 7]

N_BOOTSTRAP = 1000
RANDOM_SEED = 42
LOW_R2_CUTOFF = 0.5
HIGH_EXPR_THRESHOLD = 500

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return R^2, clamped at 0."""
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 1.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    return max(0.0, 1.0 - ss_res / ss_tot)


def fit_gene(y: np.ndarray, x: list = None):
    """Fit one gene with a linear or quadratic curve."""
    if x is None:
        x = KNOWN_DAYS
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    lin_coeffs = np.polyfit(x, y, 1)
    r2_lin = r_squared(y, np.polyval(lin_coeffs, x))

    quad_coeffs = np.polyfit(x, y, 2)
    r2_quad = r_squared(y, np.polyval(quad_coeffs, x))

    preds_lin  = {d: float(np.polyval(lin_coeffs,  d)) for d in PREDICT_DAYS}
    preds_quad = {d: float(np.polyval(quad_coeffs, d)) for d in PREDICT_DAYS}

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
    """Bootstrap confidence intervals at P3 and P7."""
    if x is None:
        x = KNOWN_DAYS
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    degree = 2 if fit_type == "quadratic" else 1
    rng = np.random.default_rng(seed)

    y_hat = np.polyval(coeffs, x)
    residuals = y - y_hat

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
    """Draw one gene panel."""
    x_known = np.array(KNOWN_DAYS, dtype=float)

    x_line = np.linspace(-15, 13, 300)
    ax.plot(x_line, np.polyval(coeffs, x_line),
            "k--", linewidth=1.2, alpha=0.65, zorder=1, label="Fit")

    ax.fill_between(
        [3, 7],
        [ci_p3[0], ci_p7[0]],
        [ci_p3[1], ci_p7[1]],
        alpha=0.18, color="steelblue", zorder=2, label="95% CI"
    )

    ax.scatter(x_known, y_known,
               color="black", s=55, zorder=5, label="Known")

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

def load_high_expr_genes(ef_path: str, threshold: float = HIGH_EXPR_THRESHOLD) -> set:
    """Return genes with mean EF counts above the threshold."""
    ef = pd.read_csv(ef_path)
    meta_cols = {"animal_id", "timepoint", "condition", "region"}
    gene_cols = [c for c in ef.columns if c not in meta_cols]
    mean_counts = ef[gene_cols].mean()
    return set(mean_counts[mean_counts >= threshold].index)


def load_protein_coding_genes(raw_counts_path: str) -> set:
    """Return protein-coding, non-mitochondrial, non-Gm genes."""
    raw = pd.read_csv(raw_counts_path, index_col=0, usecols=range(8))
    protein_coding = raw[raw["gene_type"] == "protein_coding"]["gene_name"]
    protein_coding = protein_coding[~protein_coding.str.lower().str.startswith("mt-")]
    protein_coding = protein_coding[~protein_coding.str.lower().str.startswith("gm")]
    return set(protein_coding)

def main():
    parser = argparse.ArgumentParser(
        description="Predict gene perturbation at P3 and P7 via trajectory extrapolation"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--panel-size", type=int, default=None,
                        help="Number of top genes to output (default: panel_size from config)")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                        help=f"Bootstrap iterations for reported panel genes (default: {N_BOOTSTRAP})")
    args = parser.parse_args()

    with open(resolve_path(args.config)) as f:
        config = yaml.safe_load(f)

    output_dir = resolve_path(config["output_dir"])
    panel_size = args.panel_size or config.get("panel_size", 25)

    ranked_path = os.path.join(output_dir, "all_genes_ranked.csv")
    if not os.path.exists(ranked_path):
        raise FileNotFoundError(
            f"Missing required input: {ranked_path}\n"
            f"Run `python -m inference.direct_score` to generate it."
        )

    df = pd.read_csv(ranked_path, index_col=0)
    print(f"Loaded {len(df)} genes from {ranked_path}")

    raw_counts_path = resolve_path("MIA_Data/all_counts.csv")
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

    ef_path = os.path.join(resolve_path(config["data_dir"]), "expression_ef.csv")
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

    needed = ["gene_name", "perturbation_E15", "perturbation_P0", "perturbation_P13"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns not found in {ranked_path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    print("Fitting trajectories ...")

    pred_records = []
    fit_records  = []

    for _, row in df.iterrows():
        gene = row["gene_name"]
        y = np.array([
            row["perturbation_E15"],
            row["perturbation_P0"],
            row["perturbation_P13"],
        ], dtype=float)

        fit_type, coeffs, r2_lin, r2_sel, preds = fit_gene(y)
        pred_records.append({
            "gene_name":               gene,
            "slope_divergence_E15":    float(y[0]),
            "slope_divergence_P0":     float(y[1]),
            "slope_divergence_P13":    float(y[2]),
            "predicted_divergence_P3": preds[3],
            "predicted_divergence_P7": preds[7],
            "fit_type":                fit_type,
            "R2":                      r2_sel,
            "R2_linear":               r2_lin,
            "CI_lower_P3":             np.nan,
            "CI_upper_P3":             np.nan,
            "CI_lower_P7":             np.nan,
            "CI_upper_P7":             np.nan,
        })

        fit_records.append({
            "gene_name": gene,
            "fit_type":  fit_type,
            "R2":        r2_sel,
            "R2_linear": r2_lin,
            "coeff_0":   float(coeffs[0]),
            "coeff_1":   float(coeffs[1]),
            "coeff_2":   float(coeffs[2]) if len(coeffs) > 2 else np.nan,
        })

    results = pd.DataFrame(pred_records)
    fits    = pd.DataFrame(fit_records)

    results["rank_score_P3"] = results["predicted_divergence_P3"] * results["R2"]
    results["rank_score_P7"] = results["predicted_divergence_P7"] * results["R2"]

    high_conf = results[results["R2"] >= LOW_R2_CUTOFF].copy()
    n_low_conf = len(results) - len(high_conf)

    top_p3 = (high_conf
              .sort_values(["rank_score_P3", "R2"], ascending=False)
              .head(panel_size))
    top_p7 = (high_conf
              .sort_values(["rank_score_P7", "R2"], ascending=False)
              .head(panel_size))

    ci_genes = sorted(set(top_p3["gene_name"]) | set(top_p7["gene_name"]))
    fit_map = fits.set_index("gene_name")
    result_map = results.set_index("gene_name")
    print(f"Bootstrapping CIs for {len(ci_genes)} panel genes ({args.n_bootstrap} iters) ...")
    for gene in ci_genes:
        rr = result_map.loc[gene]
        fr = fit_map.loc[gene]
        y = np.array([
            rr["slope_divergence_E15"],
            rr["slope_divergence_P0"],
            rr["slope_divergence_P13"],
        ], dtype=float)
        if fr["fit_type"] == "quadratic":
            coeffs = np.array([fr["coeff_0"], fr["coeff_1"], fr["coeff_2"]], dtype=float)
        else:
            coeffs = np.array([fr["coeff_0"], fr["coeff_1"]], dtype=float)
        ci = residual_bootstrap_ci(y, fr["fit_type"], coeffs, n_boot=args.n_bootstrap)
        results.loc[results["gene_name"] == gene, ["CI_lower_P3", "CI_upper_P3", "CI_lower_P7", "CI_upper_P7"]] = [
            ci[3][0], ci[3][1], ci[7][0], ci[7][1]
        ]

    pred_path = os.path.join(output_dir, "predicted_P3_P7.csv")
    fits_path = os.path.join(output_dir, "trajectory_fits.csv")
    top_p3_path = os.path.join(output_dir, "predicted_P3_panel.csv")
    top_p7_path = os.path.join(output_dir, "predicted_P7_panel.csv")
    pdf_path = os.path.join(output_dir, "trajectory_plots.pdf")

    results.to_csv(pred_path,   index=False)
    fits.to_csv(fits_path,      index=False)
    top_p3.to_csv(top_p3_path,  index=False)
    top_p7.to_csv(top_p7_path,  index=False)

    print(f"Saved predictions   → {pred_path}")
    print(f"Saved fit params    → {fits_path}")
    print(f"Saved top-{panel_size} at P3 → {top_p3_path}")
    print(f"Saved top-{panel_size} at P7 → {top_p7_path}")

    top10 = top_p3.head(10)
    res_idx = results.set_index("gene_name")
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
