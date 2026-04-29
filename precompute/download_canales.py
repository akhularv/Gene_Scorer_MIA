"""
precompute/download_canales.py

- Downloads GSE166376 (Canales et al. 2021, eLife) from NCBI GEO FTP
- Retrieves: all supplementary files (raw counts + 4 DE tables), series matrix (metadata)
- Also downloads NordNeurogenomicsLab GitHub analysis scripts for reference
- Parses series matrix to produce a clean metadata.csv with timepoint/condition columns
- Skips files that already exist — safe to re-run
- If any download fails, prints the exact URL and manual download instructions

Usage:
    python -m precompute.download_canales
    python -m precompute.download_canales --out-dir data/canales_GSE166376
"""

import argparse
import ftplib
import gzip
import io
import os
import time
import urllib.request
import urllib.error

import pandas as pd

from project_paths import resolve_path


GEO_ACCESSION   = "GSE166376"
FTP_HOST        = "ftp.ncbi.nlm.nih.gov"
# NCBI FTP series path: first 6 chars + "nnn" is the parent dir
FTP_SERIES_PATH = "geo/series/GSE166nnn/GSE166376"

# GitHub repo with analysis scripts from the Nord lab (for reference only)
NORD_GITHUB_BASE = (
    "https://raw.githubusercontent.com/NordNeurogenomicsLab/Publications/"
    "master/Canales_eLife_2021"
)
# these are the known analysis script filenames in the repo (download for reference)
NORD_SCRIPTS = [
    "Canales_analysis.R",
    "Canales_analysis_functions.R",
]


# ---------------------------------------------------------------------------
# FTP / download helpers
# ---------------------------------------------------------------------------

def ftp_list_dir(ftp_path: str) -> list:
    """List filenames in an NCBI GEO FTP directory. Returns relative names only."""
    ftp = ftplib.FTP(FTP_HOST, timeout=60)
    ftp.login()  # anonymous
    try:
        ftp.cwd(ftp_path)
        names = ftp.nlst()
    except ftplib.error_perm as e:
        raise RuntimeError(
            f"FTP directory not found: ftp://{FTP_HOST}/{ftp_path}\n"
            f"Error: {e}\n"
            f"Check the accession number and try again."
        ) from e
    finally:
        ftp.quit()
    return names


def download_file(ftp_path: str, local_path: str, retries: int = 3) -> None:
    """Download one file via HTTPS mirror of the NCBI FTP. Retries on failure."""
    if os.path.exists(local_path):
        print(f"  [skip] already exists: {os.path.basename(local_path)}")
        return

    # NCBI provides an HTTPS mirror of their FTP at the same path
    url = f"https://{FTP_HOST}/{ftp_path}"
    tmp  = local_path + ".tmp"

    for attempt in range(1, retries + 1):
        try:
            print(f"  Downloading ({attempt}/{retries}): {os.path.basename(local_path)}")
            urllib.request.urlretrieve(url, tmp)
            os.rename(tmp, local_path)
            return
        except (urllib.error.URLError, OSError) as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == retries:
                raise RuntimeError(
                    f"\nDownload failed after {retries} attempts: {url}\n"
                    f"Error: {e}\n"
                    f"Manual download:\n"
                    f"  wget \"{url}\" -O \"{local_path}\"\n"
                    f"  curl -o \"{local_path}\" \"{url}\""
                ) from e
            print(f"  Retry in 5s ...")
            time.sleep(5)


def download_url(url: str, local_path: str) -> None:
    """Download from an arbitrary URL. Used for GitHub scripts."""
    if os.path.exists(local_path):
        print(f"  [skip] already exists: {os.path.basename(local_path)}")
        return
    try:
        print(f"  Downloading: {os.path.basename(local_path)}")
        urllib.request.urlretrieve(url, local_path)
    except Exception as e:
        print(f"  Warning: could not download {url}: {e}")
        print(f"  This file is optional (reference scripts). Continue.")


# ---------------------------------------------------------------------------
# Series matrix parsing → metadata.csv
# ---------------------------------------------------------------------------

def parse_series_matrix(gz_path: str) -> pd.DataFrame:
    """Extract per-sample metadata from a GEO series matrix .txt.gz file.

    The series matrix format has lines like:
      !Sample_geo_accession  "GSM1"  "GSM2"  ...
      !Sample_title          "name1" "name2" ...
      !Sample_characteristics_ch1  "treatment: PolyIC"  ...
    We stop reading at '!series_matrix_table_begin' (the expression data block).
    """
    # Use a list for characteristics (multiple rows share the same key)
    fields      = {}
    char_rows   = []    # list of value-lists for !Sample_characteristics_ch* lines
    char_counts = {}    # track how many times each key has appeared

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break  # expression data starts here — skip it
            if not line.startswith("!Sample_"):
                continue
            parts = line.rstrip("\n").split("\t")
            key   = parts[0]
            vals  = [v.strip('"') for v in parts[1:]]
            if "characteristics" in key:
                # give each characteristics row a unique key so none overwrites another
                count = char_counts.get(key, 0)
                char_counts[key] = count + 1
                unique_key = f"{key}__{count}" if count > 0 else key
                fields[unique_key] = vals
                char_rows.append(unique_key)
            elif key not in fields:
                fields[key] = vals

    n = len(fields.get("!Sample_geo_accession", []))
    if n == 0:
        raise ValueError(f"No sample data found in {gz_path}")

    rows = []
    for i in range(n):
        row = {}
        for k, vs in fields.items():
            row[k] = vs[i] if i < len(vs) else ""
        rows.append(row)

    df = pd.DataFrame(rows).rename(columns={
        "!Sample_geo_accession": "geo_accession",
        "!Sample_title":         "sample_title",
    })

    # flatten all characteristics rows into individual named columns
    # e.g. "treatment: PolyIC" → column "treatment" = "PolyIC"
    char_cols = [c for c in df.columns if "characteristics" in c]
    for col in char_cols:
        for _, val in df[col].items():
            if ": " in val:
                attr_key = val.split(": ", 1)[0].strip().lower().replace(" ", "_")
                if attr_key not in df.columns:
                    df[attr_key] = df[col].apply(
                        lambda v: v.split(": ", 1)[1] if ": " in v else ""
                    )

    # keep only the useful columns (drop raw characteristics_ columns)
    keep = [c for c in df.columns if "characteristics" not in c]
    return df[keep]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download GSE166376 (Canales 2021) from NCBI GEO"
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/canales_GSE166376",
        help="Directory to save downloaded files"
    )
    args = parser.parse_args()

    out_dir = resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nDownloading {GEO_ACCESSION} → {out_dir}/\n")

    # --- 1. list and download supplementary files ---
    print("Listing supplementary files ...")
    suppl_ftp = f"{FTP_SERIES_PATH}/suppl"
    try:
        suppl_files = ftp_list_dir(suppl_ftp)
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    print(f"Found {len(suppl_files)} supplementary file(s):")
    for f in suppl_files:
        print(f"  {f}")

    print("\nDownloading supplementary files ...")
    for fname in suppl_files:
        ftp_path  = f"{suppl_ftp}/{fname}"
        local     = os.path.join(out_dir, fname)
        download_file(ftp_path, local)

    # --- 2. download series matrix (contains sample metadata) ---
    print("\nDownloading series matrix (sample metadata) ...")
    matrix_ftp   = f"{FTP_SERIES_PATH}/matrix"
    matrix_fname = f"{GEO_ACCESSION}_series_matrix.txt.gz"
    matrix_local = os.path.join(out_dir, matrix_fname)
    download_file(f"{matrix_ftp}/{matrix_fname}", matrix_local)

    # --- 3. parse series matrix → metadata.csv ---
    print("\nParsing sample metadata ...")
    try:
        meta = parse_series_matrix(matrix_local)
        meta_path = os.path.join(out_dir, "metadata.csv")
        meta.to_csv(meta_path, index=False)
        print(f"Saved metadata → {meta_path}  ({len(meta)} samples)")
        print(f"Metadata columns: {list(meta.columns)}")
    except Exception as e:
        print(f"Warning: could not parse metadata: {e}")
        print("You can parse metadata.csv manually from the series matrix file.")

    # --- 4. download NordNeurogenomicsLab reference scripts (optional) ---
    print("\nDownloading NordNeurogenomicsLab analysis scripts (reference) ...")
    ref_dir = os.path.join(out_dir, "reference_scripts")
    os.makedirs(ref_dir, exist_ok=True)
    for script in NORD_SCRIPTS:
        url   = f"{NORD_GITHUB_BASE}/{script}"
        local = os.path.join(ref_dir, script)
        download_url(url, local)

    # --- summary ---
    downloaded = [f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))]
    print(f"\n{'='*60}")
    print(f"Done. {len(downloaded)} file(s) in {out_dir}/")
    print(f"\nExpected DE tables (one per timepoint):")
    for tp in ["E12", "E14", "E17", "P0"]:
        matches = [f for f in downloaded if tp in f]
        status  = "FOUND" if matches else "NOT FOUND — check supplementary file names"
        print(f"  {tp}: {status}  {matches}")
    print(f"\nNext step: python -m precompute.process_canales")
    print("=" * 60)


if __name__ == "__main__":
    main()
