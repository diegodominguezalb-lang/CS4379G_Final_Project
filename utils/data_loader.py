"""Data loading, caching, and NOAA fetch logic for the NOAA Storm Events Dashboard."""

import glob
import gzip
import hashlib
import os
import re
import shutil
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    from huggingface_hub import hf_hub_download
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────
_DATA_DIR = "DataForProject"
_HF_DATASET = os.environ.get("HF_DATASET", "")
_PARQUET_LOCAL = os.path.join(_DATA_DIR, "storms.parquet")
_NOAA_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"


# ── Source detection ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def data_fingerprint(data_dir: str = "DataForProject") -> tuple[str, int, str]:
    """Return (short_hash, file_count, newest_file_date) for the data snapshot."""
    if _HF_AVAILABLE and _HF_DATASET:
        return _HF_DATASET, 1, "huggingface"
    if os.path.exists(_PARQUET_LOCAL):
        stat = os.stat(_PARQUET_LOCAL)
        short = hashlib.sha256(f"storms.parquet-{stat.st_size}".encode()).hexdigest()[:8]
        date_str = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
        return short, 1, date_str
    files = sorted(glob.glob(f"{data_dir}/*.csv"))
    if not files:
        return "no-data", 0, "n/a"
    hasher = hashlib.sha256()
    for f in files:
        stat = os.stat(f)
        hasher.update(f"{os.path.basename(f)}-{stat.st_size}".encode())
    short = hasher.hexdigest()[:8]
    newest = max(os.path.getmtime(f) for f in files)
    newest_str = datetime.fromtimestamp(newest).strftime("%Y-%m-%d")
    return short, len(files), newest_str


# ── HF Hub loader ─────────────────────────────────────────────────────────────
def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee YEAR, MONTH, and TOTAL_DAMAGE columns exist on any loaded df."""
    if "BEGIN_DATE_TIME" in df.columns:
        # Already datetime (parquet) or still a string — handle both
        if not pd.api.types.is_datetime64_any_dtype(df["BEGIN_DATE_TIME"]):
            df["BEGIN_DATE_TIME"] = pd.to_datetime(
                df["BEGIN_DATE_TIME"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
            )
        if "YEAR" not in df.columns or df["YEAR"].isna().all():
            df["YEAR"] = df["BEGIN_DATE_TIME"].dt.year
        if "MONTH" not in df.columns or df["MONTH"].isna().all():
            df["MONTH"] = df["BEGIN_DATE_TIME"].dt.month
    if "TOTAL_DAMAGE" not in df.columns:
        prop = pd.to_numeric(df.get("DAMAGE_PROPERTY", 0), errors="coerce").fillna(0)
        crop = pd.to_numeric(df.get("DAMAGE_CROPS", 0), errors="coerce").fillna(0)
        df["TOTAL_DAMAGE"] = prop + crop
    return df


@st.cache_data(show_spinner="Downloading dataset from Hugging Face…")
def _load_from_hf() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=_HF_DATASET,
        filename="storms.parquet",
        repo_type="dataset",
    )
    return _ensure_columns(pd.read_parquet(path))


def _load_local_parquet() -> pd.DataFrame:
    return _ensure_columns(pd.read_parquet(_PARQUET_LOCAL))


# ── NOAA auto-fetch ───────────────────────────────────────────────────────────
def _noaa_fetch_year(index_html: str, year: int) -> bool:
    """Download and decompress one year's CSV. Returns True on success."""
    pattern = rf"StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz"
    matches = re.findall(pattern, index_html)
    if not matches:
        return False
    filename = sorted(matches)[-1]
    gz_path = os.path.join(_DATA_DIR, filename)
    csv_path = gz_path.replace(".gz", "")
    try:
        with urllib.request.urlopen(f"{_NOAA_BASE}/{filename}", timeout=60) as resp, \
                open(gz_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        return True
    except Exception:
        return False


def _ensure_data() -> None:
    """Return immediately if data is available (HF, local parquet, or CSVs).
    Otherwise show a year-range picker and fetch from NOAA."""
    if _HF_AVAILABLE and _HF_DATASET:
        return
    if os.path.exists(_PARQUET_LOCAL):
        return
    os.makedirs(_DATA_DIR, exist_ok=True)
    if glob.glob(f"{_DATA_DIR}/*.csv"):
        return

    st.warning("No local data found. Fetch it directly from the NOAA FTP server below.")
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    start = col1.number_input("Start year", min_value=1950, max_value=current_year,
                              value=max(1950, current_year - 10), step=1)
    end = col2.number_input("End year", min_value=1950, max_value=current_year,
                            value=current_year, step=1)
    if not st.button("Download data from NOAA"):
        st.stop()

    try:
        with st.spinner("Fetching NOAA directory index…"):
            with urllib.request.urlopen(_NOAA_BASE + "/", timeout=30) as r:
                index_html = r.read().decode("utf-8", errors="replace")
    except Exception as exc:
        st.error(f"Could not reach NOAA server: {exc}")
        st.stop()

    years = list(range(int(start), int(end) + 1))
    progress = st.progress(0, text="Starting download…")
    ok, failed = 0, []
    for i, year in enumerate(years):
        progress.progress((i + 1) / len(years), text=f"Downloading {year}…")
        if _noaa_fetch_year(index_html, year):
            ok += 1
        else:
            failed.append(year)

    progress.empty()
    if failed:
        st.warning(f"Downloaded {ok}/{len(years)} years. Not available on server: {failed}")
    else:
        st.success(f"Downloaded {ok} year(s). Reloading…")
    st.cache_data.clear()
    st.rerun()


# ── Main data loader ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading storm event data…")
def load_data() -> pd.DataFrame:
    if _HF_AVAILABLE and _HF_DATASET:
        return _load_from_hf()

    if os.path.exists(_PARQUET_LOCAL):
        return _load_local_parquet()

    csv_files = glob.glob("DataForProject/*.csv")
    if not csv_files:
        st.error("No CSV files found in DataForProject/")
        st.stop()

    dfs = []
    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    df = pd.concat(dfs, ignore_index=True)

    # drop columns with >80% missing
    pct_missing = df.isnull().mean()
    df = df.drop(columns=pct_missing[pct_missing > 0.80].index.tolist())

    # drop rows with critical nulls
    df.dropna(subset=["STATE", "STATE_FIPS", "DATA_SOURCE"], inplace=True)

    # categorical imputation
    for col in ["CZ_NAME", "WFO", "MAGNITUDE_TYPE", "BEGIN_LOCATION", "END_LOCATION",
                "EPISODE_NARRATIVE", "EVENT_NARRATIVE"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    # numerical imputation
    df["DAMAGE_PROPERTY"] = df["DAMAGE_PROPERTY"].fillna(0)
    df["DAMAGE_CROPS"] = df["DAMAGE_CROPS"].fillna(0)
    if "SOURCE" in df.columns:
        df["SOURCE"] = df["SOURCE"].fillna(df["SOURCE"].mode()[0])
    if "MAGNITUDE" in df.columns:
        df["MAGNITUDE"] = df["MAGNITUDE"].fillna(df["MAGNITUDE"].median())
    for col in ["BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    df = df.drop(columns=["BEGIN_RANGE", "BEGIN_AZIMUTH", "END_RANGE", "END_AZIMUTH"], errors="ignore")

    # parse damage strings (e.g. "1.5K", "2M")
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

    df["DAMAGE_PROPERTY"] = df["DAMAGE_PROPERTY"].apply(parse_damage)
    df["DAMAGE_CROPS"] = df["DAMAGE_CROPS"].apply(parse_damage)

    # datetime conversion
    raw_year = pd.to_numeric(df["YEAR"], errors="coerce") if "YEAR" in df.columns else None

    if not pd.api.types.is_datetime64_any_dtype(df["BEGIN_DATE_TIME"]):
        df["BEGIN_DATE_TIME"] = pd.to_datetime(
            df["BEGIN_DATE_TIME"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
        )
    if "END_DATE_TIME" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["END_DATE_TIME"]):
        df["END_DATE_TIME"] = pd.to_datetime(
            df["END_DATE_TIME"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
        )
    df["YEAR"] = df["BEGIN_DATE_TIME"].dt.year
    if raw_year is not None:
        df["YEAR"] = df["YEAR"].fillna(raw_year)
    df["MONTH"] = df["BEGIN_DATE_TIME"].dt.month
    df["TOTAL_DAMAGE"] = df["DAMAGE_PROPERTY"] + df["DAMAGE_CROPS"]

    return df
