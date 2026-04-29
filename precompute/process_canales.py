"""
precompute/process_canales.py

- Loads Canales et al. 2021 (GSE166376) supplementary DE tables (E12.5, E14.5, E17.5, P0)
  and raw count matrix if available from data/canales_GSE166376/
- Normalization: log1p(CPM) — matches the existing pipeline's normalization exactly
- Maps timepoints to postnatal-equivalent days: E12.5→-15, E14.5→-13.5, E17.5→-10.5, P0→0
- For each gene: fits linear regression to log2FC trajectory across timepoints
  (slope of log2FC over time ≈ slope_MIA - slope_saline)
- If raw counts + metadata available: computes slope_MIA and slope_saline separately
  and reports R2_MIA / R2_saline for use as confidence filters in stream1
- Outputs:
    canales_slopes.csv     — gene_name, slope_MIA, slope_saline, slope_divergence, R2_MIA, R2_saline
    canales_log2FC.csv     — gene_name, log2FC_E12.5, log2FC_E14.5, log2FC_E17.5, log2FC_P0
    canales_timecourse.csv — gene_name, mean_MIA_E12.5, mean_saline_E12.5, ... (all timepoints)

Usage:
    python -m precompute.process_canales
    python -m precompute.process_canales --data-dir data/canales_GSE166376 --out-dir outputs
"""

import argparse
import glob
import gzip
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from project_paths import resolve_path


# Postnatal-equivalent days for Canales timepoints
TIMEPOINT_DAYS = {
    "E12.5": -15.0,
    "E14.5": -13.5,
    "E17.5": -10.5,
    "P0":     0.0,
}
TIMEPOINTS = list(TIMEPOINT_DAYS.keys())   # ordered
DAYS       = [TIMEPOINT_DAYS[t] for t in TIMEPOINTS]

# Patterns used to identify which file belongs to which timepoint
# GEO authors name files inconsistently; we try multiple patterns
TP_PATTERNS = {
    "E12.5": [r"E12", r"12\.5", r"table.?1\b", r"supp.?1\b", r"file.?1\b"],
    "E14.5": [r"E14", r"14\.5", r"table.?2\b", r"supp.?2\b", r"file.?2\b"],
    "E17.5": [r"E17", r"17\.5", r"table.?3\b", r"supp.?3\b", r"file.?3\b"],
    "P0":    [r"[_\-]P0[_\-\.]", r"postnatal", r"table.?4\b", r"supp.?4\b", r"file.?4\b"],
}

# edgeR LRT column names expected in each DE table
LOGFC_COL  = "logFC"    # case-insensitive search used below
GENE_COL   = "gene_name"


# ---------------------------------------------------------------------------
# File detection helpers
# ---------------------------------------------------------------------------

def find_de_tables(data_dir: str) -> dict:
    """Scan data_dir for files matching each timepoint pattern.

    Returns {timepoint: filepath} for whichever files are found.
    Raises ValueError if no files match any timepoint.
    """
    all_files = []
    for ext in ["*.xlsx", "*.xls", "*.txt.gz", "*.txt", "*.csv.gz", "*.csv", "*.tsv.gz", "*.tsv"]:
        all_files.extend(glob.glob(os.path.join(data_dir, ext)))

    print(f"  Found {len(all_files)} candidate file(s) in {data_dir}")

    matched = {}
    for tp, patterns in TP_PATTERNS.items():
        for fpath in all_files:
            fname = os.path.basename(fpath).lower()
            for pat in patterns:
                if re.search(pat, fname, re.IGNORECASE):
                    matched[tp] = fpath
                    break
            if tp in matched:
                break

    return matched


def load_de_table(fpath: str) -> pd.DataFrame:
    """Load a DE table from xlsx, txt.gz, or csv. Returns a DataFrame.

    Expected columns (from edgeR LRT output):
        gene_name (or gene), logFC, logCPM, LR, PValue, FDR, [WGCNA_module]
    """
    fname = fpath.lower()
    try:
        if fname.endswith(".xlsx") or fname.endswith(".xls"):
            df = pd.read_excel(fpath)
        elif fname.endswith(".txt.gz") or fname.endswith(".csv.gz") or fname.endswith(".tsv.gz"):
            with gzip.open(fpath, "rt") as f:
                sep = "\t" if (fname.endswith(".txt.gz") or fname.endswith(".tsv.gz")) else ","
                df  = pd.read_csv(f, sep=sep)
        else:
            sep = "\t" if (fname.endswith(".txt") or fname.endswith(".tsv")) else ","
            df  = pd.read_csv(fpath, sep=sep)
    except Exception as e:
        raise RuntimeError(f"Could not load {fpath}: {e}") from e

    # normalize column names to lowercase for matching
    df.columns = [c.strip() for c in df.columns]
    col_map    = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)

    # gene column: look for 'gene_name', 'gene', 'genename', 'id', first column
    gene_candidates = ["gene_name", "gene", "genename", "gene.name", "id", "symbol"]
    gene_col = next((c for c in gene_candidates if c in df.columns), None)
    if gene_col is None:
        gene_col = df.columns[0]  # fall back to first column
        print(f"  Warning: no standard gene column found; using '{gene_col}'")
    df = df.rename(columns={gene_col: "gene_name"})

    # log2FC column: look for 'logfc', 'log2fc', 'log_fc', 'lfc'
    fc_candidates = ["logfc", "log2fc", "log_fc", "lfc", "log2foldchange", "foldchange"]
    fc_col = next((c for c in fc_candidates if c in df.columns), None)
    if fc_col is None:
        raise ValueError(
            f"No log2FC column found in {fpath}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df.rename(columns={fc_col: "log2FC"})

    # logCPM column (optional, for timecourse means)
    cpm_candidates = ["logcpm", "log2cpm", "avexpr", "basemean", "log2_mean", "avg_logfc"]
    cpm_col = next((c for c in cpm_candidates if c in df.columns), None)
    if cpm_col:
        df = df.rename(columns={cpm_col: "logCPM"})

    return df[["gene_name", "log2FC"] + (["logCPM"] if cpm_col else []) +
              [c for c in ["fdr", "pvalue", "lr", "wgcna_module"] if c in df.columns]]


def find_raw_counts(data_dir: str) -> Optional[str]:
    """Find the raw count matrix in data_dir. Returns path or None."""
    patterns = ["*count*", "*raw*", "*matrix*", "*expression*"]
    for pat in patterns:
        for ext in [".txt.gz", ".txt", ".csv.gz", ".csv"]:
            matches = glob.glob(os.path.join(data_dir, pat + ext))
            if matches:
                return matches[0]
    return None


# ---------------------------------------------------------------------------
# Normalization — matches existing pipeline exactly
# ---------------------------------------------------------------------------

def log1p_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    """Library-size normalize to CPM, then log1p. Same as compute_priors.py."""
    lib_sizes = counts.sum(axis=1).values[:, None]  # per-sample total counts
    cpm       = counts.values / lib_sizes * 1e6
    return pd.DataFrame(np.log1p(cpm), index=counts.index, columns=counts.columns)


# ---------------------------------------------------------------------------
# Slope computation
# ---------------------------------------------------------------------------

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 1.0
    return max(0.0, 1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)


def linear_slope_r2(x: np.ndarray, y: np.ndarray) -> tuple:
    """Return (slope, r2) for a simple linear regression y ~ x.

    Uses the same least-squares formula as compute_priors.py.
    Handles NaN by dropping those timepoints.
    """
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan, np.nan
    xm = x[mask] - x[mask].mean()
    ym = y[mask] - y[mask].mean()
    denom = (xm ** 2).sum()
    if denom < 1e-12:
        return 0.0, 1.0
    slope  = (xm * ym).sum() / denom
    y_pred = slope * (x[mask] - x[mask].mean()) + y[mask].mean()
    return float(slope), float(r_squared(y[mask], y_pred))


# ---------------------------------------------------------------------------
# Main processing functions
# ---------------------------------------------------------------------------

def build_log2fc_table(de_tables: dict) -> pd.DataFrame:
    """Merge DE tables from all timepoints into a single log2FC matrix.

    Returns DataFrame: gene_name × {log2FC_E12.5, log2FC_E14.5, log2FC_E17.5, log2FC_P0}
    """
    frames = []
    for tp, df in de_tables.items():
        tmp = df[["gene_name", "log2FC"]].copy()
        tmp = tmp.rename(columns={"log2FC": f"log2FC_{tp}"})
        frames.append(tmp.set_index("gene_name"))

    merged = pd.concat(frames, axis=1, join="outer")
    merged.index.name = "gene_name"
    return merged.reset_index()


def compute_slopes_from_log2fc(log2fc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute slope_divergence per gene from the log2FC trajectory.

    slope_divergence ≈ slope_MIA - slope_saline when groups are balanced.
    R2 measures how linearly the perturbation changes over development.

    Without raw counts we can't separate R2_MIA and R2_saline, so we
    report R2_divergence and set R2_MIA = R2_saline = R2_divergence
    (they're used identically in the downstream filter).
    """
    x = np.array(DAYS, dtype=float)

    rows = []
    for _, row in log2fc_df.iterrows():
        gene = row["gene_name"]
        y    = np.array([row.get(f"log2FC_{tp}", np.nan) for tp in TIMEPOINTS], dtype=float)

        slope_div, r2_div = linear_slope_r2(x, y)

        rows.append({
            "gene_name":       gene,
            "slope_divergence": slope_div,
            # without raw counts we can't separate MIA/saline slopes;
            # set slope_MIA/saline to NaN and R2 values to the divergence R2
            "slope_MIA":       np.nan,
            "slope_saline":    np.nan,
            "R2_MIA":          r2_div,   # proxy: stream1 filter uses R2_MIA > 0.3
            "R2_saline":       r2_div,   # same proxy
            "R2_divergence":   r2_div,
        })

    return pd.DataFrame(rows)


def compute_slopes_from_raw(counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Compute slope_MIA, slope_saline, slope_divergence from raw counts.

    Used when the raw count matrix is available (preferred over DE tables).

    counts:   samples × genes DataFrame (index = sample names)
    metadata: one row per sample; must have columns that we normalize below:
              - timepoint column (age / timepoint / Age)
              - condition column (treatment / condition / group)
              - sample_id column (sample_title / sample_id / index)
    """
    # --- normalize column names in metadata ---
    meta = metadata.copy()
    meta.columns = [c.strip() for c in meta.columns]

    # find timepoint column
    tp_candidates = ["timepoint", "age", "Age", "Timepoint"]
    tp_col = next((c for c in tp_candidates if c in meta.columns), None)
    if tp_col is None:
        raise ValueError(f"No timepoint column in metadata. Found: {list(meta.columns)}")
    meta = meta.rename(columns={tp_col: "timepoint"})

    # find condition column (treatment / condition / group)
    cond_candidates = ["treatment", "condition", "group", "Treatment", "Condition"]
    cond_col = next((c for c in cond_candidates if c in meta.columns), None)
    if cond_col is None:
        raise ValueError(f"No condition column in metadata. Found: {list(meta.columns)}")
    meta = meta.rename(columns={cond_col: "condition"})

    # normalise condition values to 'MIA' / 'saline'
    meta["condition"] = meta["condition"].str.strip()
    meta["condition"] = meta["condition"].replace({
        "PolyIC": "MIA", "Poly I:C": "MIA", "polyIC": "MIA", "poly_ic": "MIA",
        "Saline": "saline", "PBS": "saline", "ctrl": "saline", "control": "saline",
    })

    # find sample_id column and align to counts index
    id_candidates = ["sample_title", "sample_id", "sample", "geo_accession"]
    id_col = next((c for c in id_candidates if c in meta.columns), None)
    if id_col:
        meta = meta.set_index(id_col)
    # keep only samples present in counts
    shared = meta.index.intersection(counts.index)
    if len(shared) == 0:
        raise ValueError(
            f"No matching sample IDs between metadata and counts index.\n"
            f"Counts index sample (first 3): {counts.index[:3].tolist()}\n"
            f"Metadata index sample (first 3): {meta.index[:3].tolist()}"
        )
    meta   = meta.loc[shared]
    counts = counts.loc[shared]

    # map timepoint labels to days (normalise spacing/capitalisation)
    tp_map = {tp.replace(" ", "").upper(): tp for tp in TIMEPOINTS}
    meta["timepoint"] = meta["timepoint"].apply(
        lambda v: tp_map.get(str(v).replace(" ", "").upper(), v)
    )
    valid_tps = [tp for tp in TIMEPOINTS if tp in meta["timepoint"].values]
    if not valid_tps:
        raise ValueError(
            f"No matching timepoints found in metadata. "
            f"Expected one of: {TIMEPOINTS}. "
            f"Found: {meta['timepoint'].unique().tolist()}"
        )

    # normalize counts (samples × genes)
    norm = log1p_cpm(counts)

    x = np.array([TIMEPOINT_DAYS[tp] for tp in valid_tps], dtype=float)

    rows = []
    for gene in norm.columns:
        y_mia    = []
        y_saline = []
        for tp in valid_tps:
            tp_meta     = meta[meta["timepoint"] == tp]
            mia_idx     = tp_meta[tp_meta["condition"] == "MIA"].index
            saline_idx  = tp_meta[tp_meta["condition"] == "saline"].index
            mia_vals    = norm.loc[norm.index.isin(mia_idx),    gene].values
            saline_vals = norm.loc[norm.index.isin(saline_idx), gene].values
            y_mia.append(   float(np.mean(mia_vals))    if len(mia_vals)    > 0 else np.nan)
            y_saline.append(float(np.mean(saline_vals)) if len(saline_vals) > 0 else np.nan)

        y_mia    = np.array(y_mia,    dtype=float)
        y_saline = np.array(y_saline, dtype=float)

        s_mia,    r2_mia    = linear_slope_r2(x, y_mia)
        s_saline, r2_saline = linear_slope_r2(x, y_saline)
        s_div = (s_mia - s_saline) if not (np.isnan(s_mia) or np.isnan(s_saline)) else np.nan

        rows.append({
            "gene_name":        gene,
            "slope_MIA":        s_mia,
            "slope_saline":     s_saline,
            "slope_divergence": s_div,
            "R2_MIA":           r2_mia,
            "R2_saline":        r2_saline,
            "R2_divergence":    np.nan,
        })

    return pd.DataFrame(rows)


def build_timecourse_table(de_tables: dict) -> pd.DataFrame:
    """Build mean_MIA / mean_saline timecourse from DE table logCPM and log2FC.

    logCPM (from edgeR) ≈ log2(average CPM across all samples).
    Approximation: mean_MIA ≈ logCPM + log2FC/2, mean_saline ≈ logCPM - log2FC/2
    (exact only when groups are equal-sized, but close enough for ranking purposes).
    """
    frames = []
    for tp, df in de_tables.items():
        if "logCPM" not in df.columns:
            continue
        tmp = df[["gene_name", "log2FC", "logCPM"]].copy()
        tmp[f"mean_MIA_{tp}"]    = tmp["logCPM"] + tmp["log2FC"] / 2
        tmp[f"mean_saline_{tp}"] = tmp["logCPM"] - tmp["log2FC"] / 2
        frames.append(tmp[["gene_name", f"mean_MIA_{tp}", f"mean_saline_{tp}"]].set_index("gene_name"))

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, axis=1, join="outer")
    return merged.reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process Canales GSE166376 data: slopes + log2FC tables"
    )
    parser.add_argument("--data-dir",  type=str, default="data/canales_GSE166376")
    parser.add_argument("--out-dir",   type=str, default="outputs")
    args = parser.parse_args()

    data_dir = resolve_path(args.data_dir)
    out_dir = resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Run `python -m precompute.download_canales` first."
        )

    # -----------------------------------------------------------------------
    # Step 1: try raw counts + metadata (preferred path)
    # -----------------------------------------------------------------------
    raw_counts_path = find_raw_counts(data_dir)
    meta_path       = os.path.join(data_dir, "metadata.csv")
    use_raw         = False
    raw             = None
    meta            = None
    de_tables       = {}

    if raw_counts_path and os.path.exists(meta_path):
        print(f"\nStep 1: Raw counts found → {os.path.basename(raw_counts_path)}")
        try:
            raw_genes_rows = pd.read_csv(raw_counts_path, index_col=0)
            # GEO convention: genes as rows, samples as columns → transpose
            raw   = raw_genes_rows.T
            meta  = pd.read_csv(meta_path)
            use_raw = True
            print(f"  Shape after transpose: {raw.shape[0]} samples × {raw.shape[1]} genes")
        except Exception as e:
            print(f"  Warning: could not load raw counts: {e}")
            print("  Falling back to DE table approach.")
    else:
        print("\nStep 1: No raw counts found — looking for DE tables ...")

    # -----------------------------------------------------------------------
    # Step 2: fall back to DE tables if raw counts unavailable
    # -----------------------------------------------------------------------
    if not use_raw:
        print("\nStep 2: Locating DE tables ...")
        matched = find_de_tables(data_dir)

        if not matched:
            raise FileNotFoundError(
                f"No DE table files and no raw count matrix found in {data_dir}.\n"
                f"Files present: {os.listdir(data_dir)}\n"
                f"Run `python -m precompute.download_canales` first."
            )

        print(f"  Matched {len(matched)}/{len(TIMEPOINTS)} timepoints:")
        for tp, fp in matched.items():
            print(f"    {tp} → {os.path.basename(fp)}")

        missing = [tp for tp in TIMEPOINTS if tp not in matched]
        if missing:
            print(f"  Warning: no file for {missing}. Those timepoints will be NaN.")

        for tp, fpath in matched.items():
            print(f"  Loading {tp} ...")
            de_tables[tp] = load_de_table(fpath)
            print(f"    {len(de_tables[tp])} genes, columns: {list(de_tables[tp].columns)}")

    # -----------------------------------------------------------------------
    # Step 3: build outputs
    # -----------------------------------------------------------------------
    if use_raw:
        print("\nStep 2: Computing log2FC table from raw counts ...")
        slopes_df = compute_slopes_from_raw(raw, meta)
        print(f"  {len(slopes_df)} genes with slope estimates")

        # build log2FC table from slopes (MIA - saline mean expression difference → log2 scale)
        # use the slope difference as a proxy: for each gene, compute per-timepoint log2FC
        print("Step 3: Computing per-timepoint log2FC from raw counts ...")
        norm = log1p_cpm(raw)
        meta_work = meta.copy()
        meta_work.columns = [c.strip() for c in meta_work.columns]
        tp_col = next((c for c in ["timepoint","age","Age"] if c in meta_work.columns), None)
        cond_col = next((c for c in ["treatment","condition","group"] if c in meta_work.columns), None)
        if tp_col:   meta_work = meta_work.rename(columns={tp_col: "timepoint"})
        if cond_col: meta_work = meta_work.rename(columns={cond_col: "condition"})
        meta_work["condition"] = meta_work["condition"].str.strip().replace({
            "PolyIC":"MIA","Poly I:C":"MIA","Saline":"saline","PBS":"saline",
        })
        id_col = next((c for c in ["sample_title","sample_id","sample"] if c in meta_work.columns), None)
        if id_col:
            meta_work = meta_work.set_index(id_col)

        tp_map = {tp.replace(" ","").upper(): tp for tp in TIMEPOINTS}
        meta_work["timepoint"] = meta_work["timepoint"].apply(
            lambda v: tp_map.get(str(v).replace(" ","").upper(), v)
        )
        valid_tps = [tp for tp in TIMEPOINTS if tp in meta_work["timepoint"].values]

        shared = meta_work.index.intersection(norm.index)
        meta_work = meta_work.loc[shared]
        norm_aligned = norm.loc[shared]

        fc_rows = []
        for gene in norm_aligned.columns:
            row = {"gene_name": gene}
            for tp in valid_tps:
                tp_meta  = meta_work[meta_work["timepoint"] == tp]
                mia_idx  = tp_meta[tp_meta["condition"] == "MIA"].index
                sal_idx  = tp_meta[tp_meta["condition"] == "saline"].index
                mia_mean = float(norm_aligned.loc[norm_aligned.index.isin(mia_idx), gene].mean()) if len(mia_idx) > 0 else np.nan
                sal_mean = float(norm_aligned.loc[norm_aligned.index.isin(sal_idx), gene].mean()) if len(sal_idx) > 0 else np.nan
                # log2FC in log1p(CPM) space ≈ log2 ratio
                row[f"log2FC_{tp}"] = (mia_mean - sal_mean) / np.log(2) if not (np.isnan(mia_mean) or np.isnan(sal_mean)) else np.nan
            fc_rows.append(row)

        log2fc_df = pd.DataFrame(fc_rows)
        timecourse_df = pd.DataFrame()   # full timecourse from raw counts is too large to save usefully

    else:
        print("\nStep 2: Building log2FC table ...")
        log2fc_df = build_log2fc_table(de_tables)
        print(f"  {len(log2fc_df)} genes × {len(log2fc_df.columns)-1} timepoints")

        print("Step 3: Computing slopes ...")
        slopes_df = compute_slopes_from_log2fc(log2fc_df)
        print(f"  {len(slopes_df)} genes with slope estimates")

        print("Step 4: Building timecourse table ...")
        timecourse_df = build_timecourse_table(de_tables)

    print(f"\n  log2FC table: {len(log2fc_df)} genes")

    # -----------------------------------------------------------------------
    # Step 4: save outputs
    # -----------------------------------------------------------------------
    slopes_path    = os.path.join(out_dir, "canales_slopes.csv")
    log2fc_path    = os.path.join(out_dir, "canales_log2FC.csv")
    timecourse_path = os.path.join(out_dir, "canales_timecourse.csv")

    slopes_df.to_csv(slopes_path,    index=False)
    log2fc_df.to_csv(log2fc_path,    index=False)
    if not timecourse_df.empty:
        timecourse_df.to_csv(timecourse_path, index=False)

    print(f"\n{'='*60}")
    print(f"Saved canales_slopes.csv    → {slopes_path}")
    print(f"Saved canales_log2FC.csv    → {log2fc_path}")
    if not timecourse_df.empty:
        print(f"Saved canales_timecourse.csv → {timecourse_path}")

    # quick summary stats
    n_valid = slopes_df["slope_divergence"].notna().sum()
    print(f"\nSlope divergence: {n_valid}/{len(slopes_df)} genes with valid estimates")
    print(f"  Mean |slope_divergence|: {slopes_df['slope_divergence'].abs().mean():.4f}")
    print(f"  Mean R2_MIA:  {slopes_df['R2_MIA'].mean():.3f}")
    print(f"\nNext step: python -m integration.stream1_canales_slopes")
    print("=" * 60)


if __name__ == "__main__":
    main()
