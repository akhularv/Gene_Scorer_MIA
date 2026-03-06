"""
precompute/download_kalish.py

- Downloads GSE148237 (Kalish et al. 2021, Nature Neuroscience) from NCBI GEO
- Single-cell RNA-seq from fetal cortex: E14.5 and E18.5, MIA vs PBS, male + female
- Downloads all supplementary files (count matrices + cell metadata)
- Warns about large file sizes upfront — scRNA-seq matrices can be several GB
- Skips files already present — safe to re-run
- If download fails: prints exact URL and manual wget/curl command

Usage:
    python -m precompute.download_kalish
    python -m precompute.download_kalish --out-dir data/kalish_GSE148237
"""

import argparse
import ftplib
import os
import time
import urllib.request
import urllib.error

import pandas as pd


GEO_ACCESSION   = "GSE148237"
FTP_HOST        = "ftp.ncbi.nlm.nih.gov"
FTP_SERIES_PATH = "geo/series/GSE148nnn/GSE148237"

# Expected file patterns in the supplementary directory
# Kalish et al. likely deposited: count matrices per sample + cell metadata
# (actual names confirmed at download time)
EXPECTED_PATTERNS = [
    "E14",   # E14.5 count matrix/metadata
    "E18",   # E18.5 count matrix/metadata
    "meta",  # cell metadata (barcodes, annotations)
    "cell",  # cell type annotations
    "barcode",
    "feature",
    "matrix",
]


# ---------------------------------------------------------------------------
# FTP / download helpers (same pattern as download_canales.py)
# ---------------------------------------------------------------------------

def ftp_list_dir(ftp_path: str) -> list:
    """List filenames in an NCBI GEO FTP directory."""
    ftp = ftplib.FTP(FTP_HOST, timeout=60)
    ftp.login()
    try:
        ftp.cwd(ftp_path)
        names = ftp.nlst()
    except ftplib.error_perm as e:
        raise RuntimeError(
            f"FTP directory not found: ftp://{FTP_HOST}/{ftp_path}\n"
            f"Error: {e}"
        ) from e
    finally:
        ftp.quit()
    return names


def ftp_get_sizes(ftp_path: str, filenames: list) -> dict:
    """Return {filename: size_bytes} for a list of files. Best-effort."""
    sizes = {}
    try:
        ftp = ftplib.FTP(FTP_HOST, timeout=60)
        ftp.login()
        ftp.cwd(ftp_path)
        for fname in filenames:
            try:
                sizes[fname] = ftp.size(fname)
            except Exception:
                sizes[fname] = None
        ftp.quit()
    except Exception:
        pass
    return sizes


def download_file(ftp_path: str, local_path: str, retries: int = 3) -> None:
    """Download one file via HTTPS mirror of NCBI FTP. Skips if already present."""
    if os.path.exists(local_path):
        print(f"  [skip] already exists: {os.path.basename(local_path)}")
        return

    url = f"https://{FTP_HOST}/{ftp_path}"
    tmp = local_path + ".tmp"

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


def parse_series_matrix_minimal(gz_path: str) -> pd.DataFrame:
    """Quick parse of series matrix for sample title + accession only."""
    import gzip
    fields      = {}
    char_counts = {}
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            parts = line.rstrip("\n").split("\t")
            key   = parts[0]
            vals  = [v.strip('"') for v in parts[1:]]
            if "characteristics" in key:
                count = char_counts.get(key, 0)
                char_counts[key] = count + 1
                unique_key = f"{key}__{count}" if count > 0 else key
                fields[unique_key] = vals
            elif key not in fields:
                fields[key] = vals

    n = len(fields.get("!Sample_geo_accession", []))
    rows = []
    for i in range(n):
        row = {k: vs[i] if i < len(vs) else "" for k, vs in fields.items()}
        rows.append(row)

    df = pd.DataFrame(rows).rename(columns={
        "!Sample_geo_accession": "geo_accession",
        "!Sample_title":         "sample_title",
    })
    char_cols = [c for c in df.columns if "characteristics" in c]
    for col in char_cols:
        for _, val in df[col].items():
            if ": " in val:
                attr_key = val.split(": ", 1)[0].strip().lower().replace(" ", "_")
                if attr_key not in df.columns:
                    df[attr_key] = df[col].apply(
                        lambda v: v.split(": ", 1)[1] if ": " in v else ""
                    )
    keep = [c for c in df.columns if "characteristics" not in c]
    return df[keep]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download GSE148237 (Kalish 2021) from NCBI GEO"
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/kalish_GSE148237",
        help="Directory to save downloaded files"
    )
    parser.add_argument(
        "--skip-large", action="store_true",
        help="Skip files larger than 500MB (useful for testing)"
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nDownloading {GEO_ACCESSION} → {out_dir}/\n")
    print("Note: scRNA-seq count matrices can be 1–5 GB each.")
    print("      Ensure you have sufficient disk space before continuing.\n")

    # --- 1. list supplementary files ---
    print("Listing supplementary files ...")
    suppl_ftp = f"{FTP_SERIES_PATH}/suppl"
    try:
        suppl_files = ftp_list_dir(suppl_ftp)
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    print(f"Found {len(suppl_files)} supplementary file(s).")

    # get file sizes upfront so we can warn about large ones
    print("Checking file sizes (may take a moment) ...")
    sizes = ftp_get_sizes(suppl_ftp, suppl_files)

    print("\nSupplementary file manifest:")
    large_files = []
    for fname in suppl_files:
        sz = sizes.get(fname)
        sz_str = f"{sz / 1e6:.1f} MB" if sz else "size unknown"
        if sz and sz > 500e6:
            large_files.append(fname)
            sz_str += "  ← LARGE FILE"
        print(f"  {fname}  ({sz_str})")

    if large_files and args.skip_large:
        print(f"\n--skip-large: will skip {len(large_files)} file(s) > 500 MB")

    # --- 2. download each file ---
    print("\nDownloading supplementary files ...")
    for fname in suppl_files:
        sz = sizes.get(fname)
        if args.skip_large and sz and sz > 500e6:
            print(f"  [skip-large] {fname}")
            continue
        ftp_path  = f"{suppl_ftp}/{fname}"
        local     = os.path.join(out_dir, fname)
        download_file(ftp_path, local)

    # --- 3. download series matrix for metadata ---
    print("\nDownloading series matrix ...")
    matrix_ftp   = f"{FTP_SERIES_PATH}/matrix"
    matrix_fname = f"{GEO_ACCESSION}_series_matrix.txt.gz"
    matrix_local = os.path.join(out_dir, matrix_fname)
    try:
        download_file(f"{matrix_ftp}/{matrix_fname}", matrix_local)

        print("Parsing sample metadata ...")
        meta = parse_series_matrix_minimal(matrix_local)
        meta_path = os.path.join(out_dir, "metadata.csv")
        meta.to_csv(meta_path, index=False)
        print(f"Saved metadata → {meta_path}  ({len(meta)} samples)")
        print(f"Metadata columns: {list(meta.columns)}")
    except Exception as e:
        print(f"Warning: could not download or parse series matrix: {e}")

    # --- summary ---
    downloaded = [f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))]
    print(f"\n{'='*60}")
    print(f"Done. {len(downloaded)} file(s) in {out_dir}/")
    print("\nExpected data files (check if present):")
    for pat in ["E14", "E18", "barcode", "feature", "matrix", "meta"]:
        matches = [f for f in downloaded if pat.lower() in f.lower()]
        print(f"  '{pat}': {matches if matches else 'NOT FOUND'}")
    print(f"\nNext step: python -m precompute.process_kalish")
    print("=" * 60)


if __name__ == "__main__":
    main()
