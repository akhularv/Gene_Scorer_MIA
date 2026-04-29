"""Compare the cross-dataset core panel against the internal ranked panel."""

import argparse
import os

import pandas as pd

from project_paths import resolve_path


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validate generalizable core panel against existing internal rankings"
    )
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = resolve_path(args.out_dir)

    core_path = os.path.join(out_dir, "generalizable_core_panel.csv")
    if not os.path.exists(core_path):
        raise FileNotFoundError(
            f"Missing: {core_path}\n"
            f"Run `python -m integration.combine_streams` first."
        )
    core = pd.read_csv(core_path, index_col=0)
    core.index.name = "combined_rank"
    print(f"Loaded {len(core)} genes from generalizable_core_panel.csv")

    panel_path = os.path.join(out_dir, "top_panel.csv")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(
            f"Missing: {panel_path}\n"
            f"Run the main pipeline (inference/direct_score.py) first."
        )
    panel = pd.read_csv(panel_path)
    existing_genes = set(panel["gene_name"].dropna().tolist())
    print(f"Loaded {len(existing_genes)} genes from top_panel.csv")

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
            "in_existing_panel": in_existing,
            "in_core_panel":   in_core,
            "combined_rank":   combined_rank,
            "combined_score":  combined_score,
            "evidence":        evidence,
            "internal_rank":   internal_rank,
            "category":        category,
        })

    report = pd.DataFrame(rows)

    cat_order = {"CONFIRMED": 0, "NOVEL": 1, "INTERNAL_ONLY": 2}
    report["_cat_ord"] = report["category"].map(cat_order)
    report = report.sort_values(
        ["_cat_ord", "combined_rank"],
        na_position="last"
    ).drop(columns="_cat_ord").reset_index(drop=True)

    out_path = os.path.join(out_dir, "cross_validation_report.csv")
    report.to_csv(out_path, index=False)
    print(f"\nSaved cross-validation report → {out_path}")

    confirmed      = report[report["category"] == "CONFIRMED"]
    novel          = report[report["category"] == "NOVEL"]
    internal_only  = report[report["category"] == "INTERNAL_ONLY"]

    print(f"\n{'='*60}")
    print(f"Cross-validation summary:")
    print(f"  Existing panel genes:        {len(existing_genes)}")
    print(f"  Generalizable core panel:    {len(core_genes)}")
    print(f"")
    print(f"  CONFIRMED (in both):         {len(confirmed)}")
    print(f"  NOVEL     (core only):       {len(novel)}")
    print(f"  INTERNAL_ONLY (panel only):  {len(internal_only)}")
    print(f"")

    if len(confirmed) > 0:
        print(f"CONFIRMED genes (strongest cross-dataset evidence):")
        for _, row in confirmed.iterrows():
            ir = f"internal_rank={int(row['internal_rank'])}" if pd.notna(row.get("internal_rank")) else ""
            print(f"  {row['gene_name']:<18}  combined_rank={row['combined_rank']}  "
                  f"[{row['evidence']}]  {ir}")

    if len(novel) > 0:
        print(f"\nNOVEL genes (cross-dataset support, not in existing panel):")
        for _, row in novel.head(20).iterrows():
            print(f"  {row['gene_name']:<18}  combined_rank={row['combined_rank']}  "
                  f"[{row['evidence']}]")
        if len(novel) > 20:
            print(f"  ... and {len(novel)-20} more (see cross_validation_report.csv)")

    if len(internal_only) > 0:
        print(f"\nINTERNAL_ONLY genes (in existing panel but not cross-validated):")
        for gene in internal_only["gene_name"].tolist():
            print(f"  {gene}")

    print("=" * 60)


if __name__ == "__main__":
    main()
