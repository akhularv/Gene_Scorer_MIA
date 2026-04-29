"""
precompute/process_kalish.py

- Loads Kalish et al. 2021 (GSE148237) inDrops scRNA-seq data
  from data/kalish_GSE148237/
- Each *.counts.tsv.gz file is one sample (barcodes × genes)
- Parses condition (MIA/PBS), sex (M/F), timepoint (E14/E18) from filename
- Pseudobulk: sums raw counts across all cells within each sample
- Normalization: log1p(CPM) — matches existing pipeline exactly
- Computes log2FC = (mean_log1p_CPM_MIA - mean_log1p_CPM_PBS) per (sex, timepoint)
- n_cells = total cells (all MIA + PBS samples) for that (sex, timepoint) group
- Outputs:
    kalish_pseudobulk_log2FC.csv  — gene_name, log2FC, sex, timepoint, n_cells, mean_MIA, mean_PBS
    kalish_gene_summary.csv       — gene_name, max_abs_log2FC, n_groups_significant, sex_specificity

Usage:
    python -m precompute.process_kalish
    python -m precompute.process_kalish --data-dir data/kalish_GSE148237 --out-dir outputs
"""

import argparse
import gzip
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from project_paths import resolve_path


LOG2FC_SIG_THRESHOLD = 0.5   # |log2FC| threshold for "significant" in a group


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Pattern: (MIA|PBS)(litter/rep)(sex: M|F)(timepoint: E14|E18)
# Examples: PBSBME14, MIAFE18, PBS2ME18, MIACFE14
_SAMPLE_RE = re.compile(
    r"(MIA|PBS)([A-Z0-9]?)(M|F)(E\d+)",
    re.IGNORECASE,
)


def parse_sample_name(filename: str) -> Optional[dict]:
    """Extract condition, sex, timepoint from a counts filename.

    Returns None if the filename does not match the expected pattern.
    """
    # strip path and prefix (e.g. 'GSM4456522_')
    base = os.path.basename(filename)
    base = re.sub(r"^GSM\d+_", "", base)
    base = base.replace(".counts.tsv.gz", "").replace(".counts.tsv", "")

    m = _SAMPLE_RE.search(base)
    if not m:
        return None

    condition = "MIA" if m.group(1).upper() == "MIA" else "PBS"
    sex       = "male"   if m.group(3).upper() == "M" else "female"
    timepoint = m.group(4).upper()   # E14 or E18

    return {"condition": condition, "sex": sex, "timepoint": timepoint, "base": base}


# ---------------------------------------------------------------------------
# Per-sample loading
# ---------------------------------------------------------------------------

def pseudobulk_streaming(path: str, chunksize: int = 2000) -> tuple:
    """Stream-sum a counts.tsv.gz without loading all cells into memory.

    Reads in chunks, accumulates sum. Each chunk is (chunksize × n_genes).
    Returns (gene_names: list, pseudobulk_counts: np.ndarray, n_cells: int).
    """
    total_sum = None
    gene_names = None
    n_cells = 0

    for chunk in pd.read_csv(path, sep="\t", index_col=0,
                              compression="gzip", chunksize=chunksize):
        if gene_names is None:
            gene_names = list(chunk.columns)
            total_sum  = chunk.values.sum(axis=0).astype(np.float64)
        else:
            total_sum += chunk.values.sum(axis=0)
        n_cells += chunk.shape[0]

    return gene_names, total_sum, n_cells


def log1p_cpm(counts: np.ndarray) -> np.ndarray:
    """log1p(counts / library_size * 1e6) — matches existing pipeline."""
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return np.log1p(counts / total * 1e6)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process Kalish GSE148237 scRNA-seq data into pseudobulk log2FC tables"
    )
    parser.add_argument("--data-dir", type=str, default="data/kalish_GSE148237")
    parser.add_argument("--out-dir",  type=str, default="outputs")
    args = parser.parse_args()

    data_dir = resolve_path(args.data_dir)
    out_dir = resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Run `python -m precompute.download_kalish` first."
        )

    # -----------------------------------------------------------------------
    # Step 1: find all counts.tsv.gz files
    # -----------------------------------------------------------------------
    all_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".counts.tsv.gz")
    ]
    if not all_files:
        raise FileNotFoundError(
            f"No *.counts.tsv.gz files found in {data_dir}.\n"
            f"Run `python -m precompute.download_kalish` first (without --skip-large).\n"
            f"Files present: {os.listdir(data_dir)}"
        )
    print(f"\nFound {len(all_files)} counts.tsv.gz files in {data_dir}")

    # -----------------------------------------------------------------------
    # Step 2: parse sample metadata from filenames and pseudobulk each sample
    # -----------------------------------------------------------------------
    print("\nStep 1: Loading and pseudobulking samples ...")
    samples = []
    genes_ref = None

    for path in sorted(all_files):
        meta = parse_sample_name(path)
        if meta is None:
            print(f"  [skip] could not parse sample name from: {os.path.basename(path)}")
            continue

        print(f"  {meta['base']:<22} → {meta['condition']}, {meta['sex']}, {meta['timepoint']} ...", end=" ", flush=True)
        genes, pb, n_cells = pseudobulk_streaming(path)
        print(f"({n_cells:,} cells)")

        if genes_ref is None:
            genes_ref = genes
        elif genes != genes_ref:
            # reorder to match reference gene list
            gene_idx = {g: i for i, g in enumerate(genes)}
            pb = np.array([pb[gene_idx[g]] if g in gene_idx else 0.0 for g in genes_ref])

        norm = log1p_cpm(pb)

        samples.append({
            "condition":  meta["condition"],
            "sex":        meta["sex"],
            "timepoint":  meta["timepoint"],
            "n_cells":    n_cells,
            "counts":     pb,
            "norm":       norm,
        })

    if not samples:
        raise RuntimeError("No samples could be parsed. Check filename patterns.")

    n_genes = len(genes_ref)
    print(f"\n  {len(samples)} samples loaded, {n_genes} genes")

    # -----------------------------------------------------------------------
    # Step 3: compute log2FC per (sex, timepoint)
    # -----------------------------------------------------------------------
    print("\nStep 2: Computing log2FC per (sex, timepoint) group ...")

    sexes      = sorted({s["sex"]       for s in samples})
    timepoints = sorted({s["timepoint"] for s in samples})

    rows = []
    for sex in sexes:
        for tp in timepoints:
            mia_s = [s for s in samples if s["condition"] == "MIA" and s["sex"] == sex and s["timepoint"] == tp]
            pbs_s = [s for s in samples if s["condition"] == "PBS" and s["sex"] == sex and s["timepoint"] == tp]

            if not mia_s or not pbs_s:
                print(f"  Skipping ({sex}, {tp}): MIA={len(mia_s)} PBS={len(pbs_s)} samples")
                continue

            n_cells_mia = sum(s["n_cells"] for s in mia_s)
            n_cells_pbs = sum(s["n_cells"] for s in pbs_s)
            n_cells     = n_cells_mia + n_cells_pbs

            # mean of log1p(CPM) across samples in each group
            mean_mia = np.mean(np.stack([s["norm"] for s in mia_s], axis=0), axis=0)
            mean_pbs = np.mean(np.stack([s["norm"] for s in pbs_s], axis=0), axis=0)

            log2fc = (mean_mia - mean_pbs) / np.log(2)

            print(f"  ({sex}, {tp}): MIA_samples={len(mia_s)} ({n_cells_mia} cells), "
                  f"PBS_samples={len(pbs_s)} ({n_cells_pbs} cells)")

            for g_idx, gene in enumerate(genes_ref):
                rows.append({
                    "gene_name":  gene,
                    "sex":        sex,
                    "timepoint":  tp,
                    "log2FC":     float(log2fc[g_idx]),
                    "mean_MIA":   float(mean_mia[g_idx]),
                    "mean_PBS":   float(mean_pbs[g_idx]),
                    "n_cells":    n_cells,
                    "n_mia":      n_cells_mia,
                    "n_pbs":      n_cells_pbs,
                })

    if not rows:
        raise RuntimeError("No matched (sex, timepoint) groups found. Check sample filenames.")

    log2fc_df = pd.DataFrame(rows)
    print(f"\n  {len(log2fc_df)} rows ({log2fc_df['gene_name'].nunique()} genes × "
          f"{log2fc_df.groupby(['sex','timepoint']).ngroups} groups)")

    # -----------------------------------------------------------------------
    # Step 4: gene summary
    # -----------------------------------------------------------------------
    print("\nStep 3: Building per-gene summary ...")
    summary_rows = []
    for gene, g in log2fc_df.groupby("gene_name"):
        abs_fc  = g["log2FC"].abs()
        weights = g["n_cells"].values.astype(float)
        w_norm  = weights / (weights.sum() + 1e-9)
        wmean   = float((abs_fc.values * w_norm).sum())
        maxfc   = float(abs_fc.max())
        n_sig   = int((abs_fc > LOG2FC_SIG_THRESHOLD).sum())

        male_sig   = bool((g[g["sex"]=="male"]["log2FC"].abs()   > LOG2FC_SIG_THRESHOLD).any())
        female_sig = bool((g[g["sex"]=="female"]["log2FC"].abs() > LOG2FC_SIG_THRESHOLD).any())
        if   male_sig and female_sig: sex_spec = "both"
        elif male_sig:                sex_spec = "male_only"
        elif female_sig:              sex_spec = "female_only"
        else:                         sex_spec = "neither"

        summary_rows.append({
            "gene_name":                 gene,
            "max_abs_log2FC":            maxfc,
            "weighted_mean_abs_log2FC":  wmean,
            "n_groups_significant":      n_sig,
            "detected_in_male":          male_sig,
            "detected_in_female":        female_sig,
            "sex_specificity":           sex_spec,
        })

    summary_df = pd.DataFrame(summary_rows)

    # -----------------------------------------------------------------------
    # Step 5: save
    # -----------------------------------------------------------------------
    log2fc_path  = os.path.join(out_dir, "kalish_pseudobulk_log2FC.csv")
    summary_path = os.path.join(out_dir, "kalish_gene_summary.csv")

    log2fc_df.to_csv(log2fc_path,  index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print(f"Saved pseudobulk log2FC → {log2fc_path}")
    print(f"Saved gene summary      → {summary_path}")

    sig = summary_df[summary_df["n_groups_significant"] > 0]
    top = summary_df.nlargest(10, "max_abs_log2FC")
    print(f"\nGenes significant in ≥1 group: {len(sig)}/{len(summary_df)}")
    print(f"\nTop 10 genes by max |log2FC|:")
    for _, row in top.iterrows():
        print(f"  {row['gene_name']:<18}  max_log2FC={row['max_abs_log2FC']:.3f}  "
              f"sex={row['sex_specificity']}")
    print(f"\nNext step: python -m integration.stream2_kalish_celltype")
    print("=" * 60)


if __name__ == "__main__":
    main()
