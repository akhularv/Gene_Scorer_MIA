"""Compare the measured panel with the predicted P3 and P7 panels."""

import argparse
import os

import pandas as pd
import yaml

from project_paths import resolve_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare the known panel to the predicted P3/P7 panels"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()

    with open(resolve_path(args.config)) as f:
        config = yaml.safe_load(f)

    output_dir = resolve_path(config["output_dir"])

    def load_required(filename: str) -> pd.DataFrame:
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Run the upstream script that generates it before running this script."
            )
        df = pd.read_csv(path)
        if "gene_name" not in df.columns:
            raise ValueError(
                f"Column 'gene_name' not found in {path}.\n"
                f"Available columns: {list(df.columns)}"
            )
        return df

    known = load_required("top_panel.csv")
    pred_p3 = load_required("predicted_P3_panel.csv")
    pred_p7 = load_required("predicted_P7_panel.csv")

    known_genes = set(known["gene_name"])
    p3_genes    = set(pred_p3["gene_name"])
    p7_genes    = set(pred_p7["gene_name"])

    all_genes = known_genes | p3_genes | p7_genes

    rows = []
    for gene in sorted(all_genes):
        in_known = gene in known_genes
        in_p3    = gene in p3_genes
        in_p7    = gene in p7_genes

        novel_p3 = in_p3 and not in_known
        novel_p7 = in_p7 and not in_known

        resolves = in_known and not in_p3 and not in_p7

        rows.append({
            "gene_name": gene,
            "in_known_panel": in_known,
            "in_predicted_P3_panel": in_p3,
            "in_predicted_P7_panel": in_p7,
            "novel_at_P3": novel_p3,
            "novel_at_P7": novel_p7,
            "resolves_by_P3": resolves,
        })

    table = pd.DataFrame(rows)

    out_path = os.path.join(output_dir, "comparison_table.csv")
    table.to_csv(out_path, index=False)
    print(f"Saved → {out_path}\n")

    overlap_p3     = known_genes & p3_genes
    overlap_p7     = known_genes & p7_genes
    novel_p3_genes = sorted(table.loc[table["novel_at_P3"],    "gene_name"])
    novel_p7_genes = sorted(table.loc[table["novel_at_P7"],    "gene_name"])
    resolving      = sorted(table.loc[table["resolves_by_P3"], "gene_name"])

    W = 65
    def section(label, genes):
        print(f"\n  {label} ({len(genes)} genes):")
        if genes:
            for g in genes:
                print(f"    • {g}")
        else:
            print("    (none)")

    print("=" * W)
    print("  KNOWN PANEL vs PREDICTED P3/P7 — COMPARISON SUMMARY")
    print("=" * W)
    print(f"\n  {'List':<35} {'Size':>5}")
    print(f"  {'-'*35} {'-'*5}")
    print(f"  {'Known panel (E15/P0/P13)':<35} {len(known_genes):>5}")
    print(f"  {'Predicted panel at P3':<35} {len(p3_genes):>5}")
    print(f"  {'Predicted panel at P7':<35} {len(p7_genes):>5}")

    print(f"\n  {'Overlap':<35} {'Count':>5}")
    print(f"  {'-'*35} {'-'*5}")
    print(f"  {'Known ∩ Predicted P3':<35} {len(overlap_p3):>5}")
    print(f"  {'Known ∩ Predicted P7':<35} {len(overlap_p7):>5}")

    section("Novel at P3 — not in known panel, predicted to emerge", novel_p3_genes)
    section("Novel at P7 — not in known panel, predicted to emerge", novel_p7_genes)
    section("Resolving by P3 — in known panel, fades from predictions", resolving)

    print("\n" + "=" * W)


if __name__ == "__main__":
    main()
