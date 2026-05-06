"""
build_parquet.py – Merge all NOAA Storm Events CSVs into a single Parquet file.

Run this locally ONCE after fetching data:
    python build_parquet.py

Output: DataForProject/storms.parquet  (~300-400 MB, loads ~5x faster than CSVs)

Then upload storms.parquet to your Hugging Face dataset:
    huggingface-cli login
    huggingface-cli upload <your-hf-username>/noaa-storm-events \
        DataForProject/storms.parquet storms.parquet
"""

import glob
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "DataForProject"
OUT_PATH = DATA_DIR / "storms.parquet"


def parse_damage(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        v = val.strip().upper()
        if not v:
            return 0.0
        multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
        if v[-1] in multipliers and v[:-1]:
            try:
                return float(v[:-1]) * multipliers[v[-1]]
            except ValueError:
                return 0.0
        try:
            return float(v)
        except ValueError:
            return 0.0
    return 0.0


def main():
    csv_files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not csv_files:
        print("ERROR: No CSV files found in DataForProject/. Run fetch_data.py first.")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files. Reading...")
    dfs = []
    for i, f in enumerate(csv_files, 1):
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
            print(f"  [{i}/{len(csv_files)}] {Path(f).name}")
        except Exception as exc:
            print(f"  SKIP {Path(f).name}: {exc}")

    print("Concatenating...")
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Raw shape: {df.shape}")

    # ── drop columns with >80% missing ──────────────────────────────────────
    pct_missing = df.isnull().mean()
    drop_cols = pct_missing[pct_missing > 0.80].index.tolist()
    df.drop(columns=drop_cols, inplace=True)

    # ── drop rows with critical nulls ────────────────────────────────────────
    df.dropna(subset=["STATE", "STATE_FIPS", "DATA_SOURCE"], inplace=True)

    # ── imputation ───────────────────────────────────────────────────────────
    for col in ["CZ_NAME", "WFO", "MAGNITUDE_TYPE", "BEGIN_LOCATION", "END_LOCATION",
                "EPISODE_NARRATIVE", "EVENT_NARRATIVE"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")
    df["DAMAGE_PROPERTY"] = df["DAMAGE_PROPERTY"].fillna(0)
    df["DAMAGE_CROPS"] = df["DAMAGE_CROPS"].fillna(0)
    if "SOURCE" in df.columns:
        df["SOURCE"] = df["SOURCE"].fillna(df["SOURCE"].mode()[0])
    if "MAGNITUDE" in df.columns:
        df["MAGNITUDE"] = df["MAGNITUDE"].fillna(df["MAGNITUDE"].median())
    for col in ["BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    df.drop(columns=["BEGIN_RANGE", "BEGIN_AZIMUTH", "END_RANGE", "END_AZIMUTH"],
            errors="ignore", inplace=True)

    # ── parse damage strings ─────────────────────────────────────────────────
    df["DAMAGE_PROPERTY"] = df["DAMAGE_PROPERTY"].apply(parse_damage)
    df["DAMAGE_CROPS"] = df["DAMAGE_CROPS"].apply(parse_damage)

    # ── datetime conversion ──────────────────────────────────────────────────
    raw_year = pd.to_numeric(df["YEAR"], errors="coerce") if "YEAR" in df.columns else None
    df["BEGIN_DATE_TIME"] = pd.to_datetime(
        df["BEGIN_DATE_TIME"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
    )
    df["END_DATE_TIME"] = pd.to_datetime(
        df["END_DATE_TIME"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
    )
    df["YEAR"] = df["BEGIN_DATE_TIME"].dt.year
    if raw_year is not None:
        df["YEAR"] = df["YEAR"].fillna(raw_year)
    df["MONTH"] = df["BEGIN_DATE_TIME"].dt.month
    df["TOTAL_DAMAGE"] = df["DAMAGE_PROPERTY"] + df["DAMAGE_CROPS"]

    print(f"  Final shape: {df.shape}")
    print(f"Writing {OUT_PATH} ...")
    df.to_parquet(OUT_PATH, index=False, compression="snappy")
    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"Done! {size_mb:.1f} MB")
    print()
    print("Next steps:")
    print("  1. huggingface-cli login")
    print(f"  2. huggingface-cli upload <your-hf-username>/noaa-storm-events {OUT_PATH} storms.parquet")


if __name__ == "__main__":
    main()
