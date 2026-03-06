"""
integration/compare_to_existing.py

- Loads generalizable_core_panel.csv (new, cross-dataset)
- Loads outputs/top25_panel.csv          (existing pipeline, internal only)
- Loads outputs/all_genes_ranked.csv     (full internal ranking, for context)
- Produces:
    - cross_validation_report.csv:  per-gene comparison table
    - A console summary of overlap, novel genes, and missing genes

Columns in cross_validation_report.csv:
    gene_name, in_existing_top25, in_core_panel, combined_rank,
    evidence (both_streams / canales_only / kalish_only),
    internal_rank (rank in all_genes_ranked.csv, if present),
    category:
        CONFIRMED   — in both existing top25 AND core panel
        NOVEL       — in core panel, NOT in existing top25
        INTERNAL_ONLY — in existing top25, NOT in core panel

Usage:
    python -m integration.compare_to_existing
    python -m integration.compare_to_existing --out-dir outputs
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validate generalizable core panel against existing internal rankings"
    )
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = args.out_dir

    # --- load core panel ---
    core_path = os.path.join(out_dir, "generalizable_core_panel.csv")
    if not os.path.exists(core_path):
        raise FileNotFoundError(
            f"Missing: {core_path}\n"
            f"Run `python -m integration.combine_streams` first."
        )
    core = pd.read_csv(core_path, index_col=0)
    core.index.name = "combined_rank"
    print(f"Loaded {len(core)} genes from generalizable_core_panel.csv")

    # --- load existing top25 ---
    top25_path = os.path.join(out_dir, "top25_panel.csv")
    if not os.path.exists(top25_path):
        raise FileNotFoundError(
            f"Missing: {top25_path}\n"
            f"Run the main pipeline (inference/direct_score.py) first."
        )
    top25 = pd.read_csv(top25_path)
    # top25_panel.csv uses 'gene_name' column
    existing_genes = set(top25["gene_name"].dropna().tolist())
    print(f"Loaded {len(existing_genes)} genes from top25_panel.csv")

    # --- load full internal ranking (optional, for rank lookup) ---
    ranked_path = os.path.join(out_dir, "all_genes_ranked.csv")
    if os.path.exists(ranked_path):
        ranked = pd.read_csv(ranked_path)
        ranked = ranked.reset_index(drop=True)
        ranked.index = ranked.index + 1
        ranked.index.name = "internal_rank"
        ranked_reset = ranked.reset_index()[["internal_rank", "gene_name"]]
        print(f"Loaded {len(ranked)} genes from all_genes_ranked.csv")
    else:
        ranked_reset = None
        print(f"Note: {ranked_path} not found — internal_rank column will be empty")

    # -----------------------------------------------------------------------
    # Build comparison table
    # -----------------------------------------------------------------------
    core_genes = set(core["gene_name"].dropna().tolist())
    all_genes  = existing_genes | core_genes

    rows = []
    for gene in all_genes:
        in_existing   = gene in existing_genes
        in_core       = gene in core_genes

        if in_core:
            core_row      = core[core["gene_name"] == gene].iloc[0]
            combined_rank = int(core_row.name)
            evidence      = core_row.get("evidence", "")
            combined_score = core_row.get("combined_score", float("nan"))
        else:
            combined_rank  = None
            evidence       = None
            combined_score = float("nan")

        if ranked_reset is not None:
            r_match = ranked_reset[ranked_reset["gene_name"] == gene]
            internal_rank = int(r_match["internal_rank"].iloc[0]) if len(r_match) > 0 else None
        else:
            internal_rank = None

        if   in_existing and in_core:  category = "CONFIRMED"
        elif in_core and not in_existing: category = "NOVEL"
        else:                             category = "INTERNAL_ONLY"

        rows.append({
            "gene_name":       gene,
            "in_existing_top25": in_existing,
            "in_core_panel":   in_core,
            "combined_rank":   combined_rank,
            "combined_score":  combined_score,
            "evidence":        evidence,
            "internal_rank":   internal_rank,
            "category":        category,
        })

    report = pd.DataFrame(rows)

    # sort: CONFIRMED first (by combined_rank), then NOVEL, then INTERNAL_ONLY
    cat_order = {"CONFIRMED": 0, "NOVEL": 1, "INTERNAL_ONLY": 2}
    report["_cat_ord"] = report["category"].map(cat_order)
    report = report.sort_values(
        ["_cat_ord", "combined_rank"],
        na_position="last"
    ).drop(columns="_cat_ord").reset_index(drop=True)

    # --- save ---
    out_path = os.path.join(out_dir, "cross_validation_report.csv")
    report.to_csv(out_path, index=False)
    print(f"\nSaved cross-validation report → {out_path}")

    # --- summary ---
    confirmed      = report[report["category"] == "CONFIRMED"]
    novel          = report[report["category"] == "NOVEL"]
    internal_only  = report[report["category"] == "INTERNAL_ONLY"]

    print(f"\n{'='*60}")
    print(f"Cross-validation summary:")
    print(f"  Existing top-25 genes:       {len(existing_genes)}")
    print(f"  Generalizable core panel:    {len(core_genes)}")
    print(f"")
    print(f"  CONFIRMED (in both):         {len(confirmed)}")
    print(f"  NOVEL     (core only):       {len(novel)}")
    print(f"  INTERNAL_ONLY (top25 only):  {len(internal_only)}")
    print(f"")

    if len(confirmed) > 0:
        print(f"CONFIRMED genes (strongest cross-dataset evidence):")
        for _, row in confirmed.iterrows():
            ir = f"internal_rank={int(row['internal_rank'])}" if pd.notna(row.get("internal_rank")) else ""
            print(f"  {row['gene_name']:<18}  combined_rank={row['combined_rank']}  "
                  f"[{row['evidence']}]  {ir}")

    if len(novel) > 0:
        print(f"\nNOVEL genes (cross-dataset support, not in existing top-25):")
        for _, row in novel.head(20).iterrows():
            print(f"  {row['gene_name']:<18}  combined_rank={row['combined_rank']}  "
                  f"[{row['evidence']}]")
        if len(novel) > 20:
            print(f"  ... and {len(novel)-20} more (see cross_validation_report.csv)")

    if len(internal_only) > 0:
        print(f"\nINTERNAL_ONLY genes (in existing top-25 but not cross-validated):")
        for gene in internal_only["gene_name"].tolist():
            print(f"  {gene}")

    print("=" * 60)


if __name__ == "__main__":
    main()
