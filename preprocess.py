"""Preprocess raw MIA data into pipeline-ready format.

Reads from MIA_Data/, writes formatted files to data/.
Handles: header shift, transpose, ID normalization, field renaming,
and minimum-expression gene filtering.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml


def normalize_sample_id(sid: str) -> str:
    """Normalize sample ID: strip trailing _, remove trailing _S/_P suffix,
    strip leading zeros from the final numeric segment."""
    sid = sid.strip().rstrip("_")
    # Remove trailing _S or _P condition-indicator suffix
    sid = re.sub(r"_[SP]$", "", sid)
    # Strip leading zeros from the last purely-numeric segment
    parts = sid.split("_")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            parts[i] = str(int(parts[i]))
            break
    return "_".join(parts)


def parse_ef_column(col: str) -> dict:
    """Derive metadata from an EF (all_counts) column name.
    Pattern: {age}_{cond}_{litter}_{pup}[_suffix]
    """
    age_map = {"E14": "E15", "P0": "P0", "P13": "P13"}
    parts = col.split("_")
    age_code = parts[0]
    cond_letter = parts[1]
    return {
        "timepoint": age_map[age_code],
        "condition": "saline" if cond_letter == "S" else "polyIC",
    }


def parse_wc_column(col: str) -> dict:
    """Derive metadata from a WC (wholetissue) column name.
    Pattern: {cond}_{age}_{litter}_{pup}[_suffix]
    """
    age_map = {"E14": "E15", "P0": "P0", "W10": "P70"}
    parts = col.split("_")
    cond_letter = parts[0]
    age_code = parts[1]
    return {
        "timepoint": age_map[age_code],
        "condition": "saline" if cond_letter == "S" else "polyIC",
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw MIA data")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(base_dir, args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    min_mean_count = config.get("min_mean_count", 10)

    raw_dir = os.path.join(base_dir, "MIA_Data")
    out_dir = os.path.join(base_dir, "data")
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Load raw count matrices (index_col=0 handles the R row-index shift)
    # ----------------------------------------------------------------
    print("Loading raw count files...")
    ef_raw = pd.read_csv(os.path.join(raw_dir, "all_counts.csv"), index_col=0)
    wc_raw = pd.read_csv(os.path.join(raw_dir, "wholetissue_counts.csv"), index_col=0)

    # ----------------------------------------------------------------
    # 2. Identify gene names and sample columns
    # ----------------------------------------------------------------
    ef_gene_meta = {"gene_id", "chr", "start", "end", "strand", "gene_type", "gene_name"}
    wc_gene_meta = {"EnsID", "chr", "gene_cat", "Start", "End", "Strand", "gene_type", "gene_name"}

    ef_gene_names = ef_raw["gene_name"].values
    wc_gene_names = wc_raw["gene_name"].values

    ef_sample_cols = [c for c in ef_raw.columns if c not in ef_gene_meta]
    wc_sample_cols = [c for c in wc_raw.columns if c not in wc_gene_meta]

    print(f"EF: {len(ef_gene_names)} genes, {len(ef_sample_cols)} samples")
    print(f"WC: {len(wc_gene_names)} genes, {len(wc_sample_cols)} samples")

    # ----------------------------------------------------------------
    # 3. Verify gene sets match
    # ----------------------------------------------------------------
    assert len(ef_gene_names) == len(wc_gene_names), (
        f"Gene count mismatch: EF={len(ef_gene_names)}, WC={len(wc_gene_names)}"
    )
    mismatches = np.sum(ef_gene_names != wc_gene_names)
    if mismatches > 0:
        print(f"WARNING: {mismatches} gene names differ between EF and WC!")
        # Use intersection in same order
        common = set(ef_gene_names) & set(wc_gene_names)
        mask = np.array([g in common for g in ef_gene_names])
        ef_gene_names = ef_gene_names[mask]
        wc_mask = np.array([g in common for g in wc_gene_names])
        wc_gene_names = wc_gene_names[wc_mask]
    else:
        print("Gene names match between EF and WC.")
    gene_names = list(ef_gene_names)

    # ----------------------------------------------------------------
    # 4. Transpose: genes-as-rows → genes-as-columns
    # ----------------------------------------------------------------
    # EF: extract count submatrix [genes x samples], transpose to [samples x genes]
    ef_counts = ef_raw.loc[:, ef_sample_cols].values.T  # [n_samples x n_genes]
    wc_counts = wc_raw.loc[:, wc_sample_cols].values.T

    # If we filtered genes above, apply the same mask
    if mismatches > 0:
        ef_counts = ef_counts[:, mask]
        wc_counts = wc_counts[:, wc_mask]

    # ----------------------------------------------------------------
    # 4a. Filter genes by minimum expression
    # ----------------------------------------------------------------
    # Genes with very low counts (mean < threshold) are sampling noise,
    # not real biological signal. Their fold-changes between conditions
    # are artifacts of random count fluctuations (0 vs 1) and must be
    # removed before any downstream analysis.
    # Require minimum mean count in BOTH tissues:
    #   - EF: needed to compute perturbation (MIA effect on neurons)
    #   - WC: needed for qPCR detection (bulk tissue assay)
    ef_gene_means = ef_counts.astype(float).mean(axis=0)  # mean across samples per gene
    wc_gene_means = wc_counts.astype(float).mean(axis=0)
    pass_filter = (ef_gene_means >= min_mean_count) & (wc_gene_means >= min_mean_count)

    n_before = len(gene_names)
    gene_names = [g for g, p in zip(gene_names, pass_filter) if p]
    ef_counts = ef_counts[:, pass_filter]
    wc_counts = wc_counts[:, pass_filter]
    n_after = len(gene_names)

    print(f"Gene filter: {n_before} -> {n_after} genes "
          f"(removed {n_before - n_after} genes with mean count < {min_mean_count} in either tissue)")

    # ----------------------------------------------------------------
    # 5. Build normalized sample-ID lookup from column names
    # ----------------------------------------------------------------
    ef_norm_ids = [normalize_sample_id(c) for c in ef_sample_cols]
    wc_norm_ids = [normalize_sample_id(c) for c in wc_sample_cols]

    # ----------------------------------------------------------------
    # 6. Load metadata, normalize its IDs, build lookup
    # ----------------------------------------------------------------
    print("Loading metadata...")
    meta_raw = pd.read_csv(os.path.join(raw_dir, "MIA_metadata.csv"))
    meta_raw["norm_id"] = meta_raw["sample_id"].apply(normalize_sample_id)

    # Rename fields to pipeline conventions
    condition_map = {"Saline": "saline", "PolyIC": "polyIC"}
    region_map = {"pyramidal neurons": "excitatory_frontal", "whole tissue": "whole_cortex"}
    age_map_meta = {"E15": "E15", "P0": "P0", "P13": "P13", "P70": "P70"}

    meta_raw["condition"] = meta_raw["treatment"].map(condition_map)
    meta_raw["region"] = meta_raw["sample_type"].map(region_map)
    meta_raw["timepoint"] = meta_raw["age"].map(age_map_meta)

    meta_lookup = {}
    for _, row in meta_raw.iterrows():
        meta_lookup[row["norm_id"]] = {
            "timepoint": row["timepoint"],
            "condition": row["condition"],
            "region": row["region"],
        }

    # ----------------------------------------------------------------
    # 7. Build EF expression DataFrame with metadata columns
    # ----------------------------------------------------------------
    print("Building expression_ef.csv...")
    ef_rows = []
    ef_matched = 0
    ef_derived = 0
    for i, (orig_col, norm_id) in enumerate(zip(ef_sample_cols, ef_norm_ids)):
        if norm_id in meta_lookup:
            info = meta_lookup[norm_id]
            ef_matched += 1
        else:
            info = parse_ef_column(orig_col)
            info["region"] = "excitatory_frontal"
            ef_derived += 1
            print(f"  EF derived metadata for: {orig_col} -> {norm_id}")
        ef_rows.append({
            "animal_id": norm_id,
            "timepoint": info["timepoint"],
            "condition": info["condition"],
        })

    ef_meta_df = pd.DataFrame(ef_rows)
    ef_expr_df = pd.DataFrame(ef_counts, columns=gene_names)
    ef_out = pd.concat([ef_meta_df, ef_expr_df], axis=1)
    print(f"  EF: {ef_matched} matched metadata, {ef_derived} derived from column names")

    # ----------------------------------------------------------------
    # 8. Build WC expression DataFrame with metadata columns
    # ----------------------------------------------------------------
    print("Building expression_wc.csv...")
    wc_rows = []
    wc_matched = 0
    wc_derived = 0
    for i, (orig_col, norm_id) in enumerate(zip(wc_sample_cols, wc_norm_ids)):
        if norm_id in meta_lookup:
            info = meta_lookup[norm_id]
            wc_matched += 1
        else:
            info = parse_wc_column(orig_col)
            info["region"] = "whole_cortex"
            wc_derived += 1
            print(f"  WC derived metadata for: {orig_col} -> {norm_id}")
        wc_rows.append({
            "animal_id": norm_id,
            "timepoint": info["timepoint"],
            "condition": info["condition"],
        })

    wc_meta_df = pd.DataFrame(wc_rows)
    wc_expr_df = pd.DataFrame(wc_counts, columns=gene_names)
    wc_out = pd.concat([wc_meta_df, wc_expr_df], axis=1)
    print(f"  WC: {wc_matched} matched metadata, {wc_derived} derived from column names")

    # ----------------------------------------------------------------
    # 9. Build unified metadata.csv
    # ----------------------------------------------------------------
    print("Building metadata.csv...")
    meta_entries = []
    for _, row in ef_out.iterrows():
        meta_entries.append({
            "animal_id": row["animal_id"],
            "timepoint": row["timepoint"],
            "condition": row["condition"],
            "region": "excitatory_frontal",
        })
    for _, row in wc_out.iterrows():
        meta_entries.append({
            "animal_id": row["animal_id"],
            "timepoint": row["timepoint"],
            "condition": row["condition"],
            "region": "whole_cortex",
        })
    meta_out = pd.DataFrame(meta_entries)

    # ----------------------------------------------------------------
    # 10. Save everything
    # ----------------------------------------------------------------
    ef_out.to_csv(os.path.join(out_dir, "expression_ef.csv"), index=False)
    wc_out.to_csv(os.path.join(out_dir, "expression_wc.csv"), index=False)
    meta_out.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)

    print(f"\nSaved to {out_dir}/:")
    print(f"  expression_ef.csv: {ef_out.shape[0]} samples x {ef_out.shape[1]} columns")
    print(f"  expression_wc.csv: {wc_out.shape[0]} samples x {wc_out.shape[1]} columns")
    print(f"  metadata.csv: {meta_out.shape[0]} rows")

    # ----------------------------------------------------------------
    # 11. Summary of timepoints/conditions
    # ----------------------------------------------------------------
    print("\n--- EF sample breakdown ---")
    print(ef_out.groupby(["timepoint", "condition"]).size().to_string())
    print("\n--- WC sample breakdown ---")
    print(wc_out.groupby(["timepoint", "condition"]).size().to_string())


if __name__ == "__main__":
    main()
