import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import gzip
import hashlib
import os
import re
import shutil
import urllib.request
from datetime import datetime, timezone

try:
    from huggingface_hub import hf_hub_download
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# ── Colorblind-safe palette (Okabe-Ito, Lecture 21 § HCI for Viz) ──────────────────
# Color is NEVER the only signal; charts pair color with text labels / position.
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999",
]

# ── Reproducibility: fix random seed (L4 · Compute) ──────────────────────────
np.random.seed(42)

# ── Data fingerprint helper (L3 · Data) ──────────────────────────────────────
@st.cache_data(show_spinner=False)
def data_fingerprint(data_dir: str = "DataForProject") -> tuple[str, int, str]:
    """Return (short_hash, file_count, newest_file_date) for the CSV snapshot."""
    # HF Hub source
    if _HF_AVAILABLE and _HF_DATASET:
        return _HF_DATASET, 1, "huggingface"
    # Local parquet
    if os.path.exists(_PARQUET_LOCAL):
        stat = os.stat(_PARQUET_LOCAL)
        short = hashlib.sha256(f"storms.parquet-{stat.st_size}".encode()).hexdigest()[:8]
        date_str = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
        return short, 1, date_str
    # Local CSVs
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

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NOAA Storm Events Dashboard",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Data loading & processing (cached)
# ──────────────────────────────────────────────────────────────────────────────
# Hugging Face dataset config — set HF_DATASET to your repo ID after uploading
# e.g. "your-hf-username/noaa-storm-events"
# ──────────────────────────────────────────────────────────────────────────────
_DATA_DIR = "DataForProject"
_HF_DATASET = os.environ.get("HF_DATASET", "")
_PARQUET_LOCAL = os.path.join(_DATA_DIR, "storms.parquet")

@st.cache_data(show_spinner="Downloading dataset from Hugging Face…")
def _load_from_hf() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=_HF_DATASET,
        filename="storms.parquet",
        repo_type="dataset",
    )
    return pd.read_parquet(path)

def _load_local_parquet() -> pd.DataFrame:
    return pd.read_parquet(_PARQUET_LOCAL)

# ──────────────────────────────────────────────────────────────────────────────
# NOAA auto-fetch (runs when DataForProject/ is empty, e.g. on Streamlit Cloud)
# ──────────────────────────────────────────────────────────────────────────────
_NOAA_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"

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
    # 1. Hugging Face dataset (Spaces deployment)
    if _HF_AVAILABLE and _HF_DATASET:
        return
    # 2. Local pre-built parquet
    if os.path.exists(_PARQUET_LOCAL):
        return
    # 3. Local CSVs
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

# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading storm event data…")
def load_data() -> pd.DataFrame:
    # ── Priority 1: Hugging Face Hub (Spaces deployment) ────────────────────
    if _HF_AVAILABLE and _HF_DATASET:
        return _load_from_hf()

    # ── Priority 2: local pre-built parquet ─────────────────────────────────
    if os.path.exists(_PARQUET_LOCAL):
        return _load_local_parquet()

    # ── Priority 3: raw CSVs (with full processing pipeline) ────────────────
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

    # ── drop columns with >80% missing ──────────────────────────────────────
    pct_missing = df.isnull().mean()
    df = df.drop(columns=pct_missing[pct_missing > 0.80].index.tolist())

    # ── drop rows with critical nulls ────────────────────────────────────────
    df.dropna(subset=["STATE", "STATE_FIPS", "DATA_SOURCE"], inplace=True)

    # ── categorical imputation ───────────────────────────────────────────────
    for col in ["CZ_NAME", "WFO", "MAGNITUDE_TYPE", "BEGIN_LOCATION", "END_LOCATION",
                "EPISODE_NARRATIVE", "EVENT_NARRATIVE"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    # ── numerical imputation ─────────────────────────────────────────────────
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

    # ── parse damage strings (e.g. "1.5K", "2M") ────────────────────────────
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

    # ── datetime conversion ──────────────────────────────────────────────────
    # Preserve the raw YEAR column from the CSV before overwriting
    raw_year = pd.to_numeric(df["YEAR"], errors="coerce") if "YEAR" in df.columns else None

    df["BEGIN_DATE_TIME"] = pd.to_datetime(
        df["BEGIN_DATE_TIME"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
    )
    df["END_DATE_TIME"] = pd.to_datetime(
        df["END_DATE_TIME"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
    )
    df["YEAR"] = df["BEGIN_DATE_TIME"].dt.year
    # Fall back to the CSV's YEAR column where datetime parsing failed
    if raw_year is not None:
        df["YEAR"] = df["YEAR"].fillna(raw_year)
    df["MONTH"] = df["BEGIN_DATE_TIME"].dt.month
    df["TOTAL_DAMAGE"] = df["DAMAGE_PROPERTY"] + df["DAMAGE_CROPS"]

    return df


_ensure_data()
df_all = load_data()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — global filters
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⛈️ Storm Events")
st.sidebar.markdown("---")

year_vals = df_all["YEAR"].dropna()
year_min = int(year_vals.min()) if len(year_vals) > 0 else 1950
year_max = int(year_vals.max()) if len(year_vals) > 0 else 2025
year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(1996, year_max),
    step=1,
)

all_states = sorted(df_all["STATE"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect(
    "States / Territories",
    options=all_states,
    default=[],
    placeholder="All states",
)

all_event_types = sorted(df_all["EVENT_TYPE"].dropna().unique().tolist())
selected_events = st.sidebar.multiselect(
    "Event types",
    options=all_event_types,
    default=[],
    placeholder="All event types",
)

st.sidebar.markdown("---")
_fp_hash, _fp_count, _fp_date = data_fingerprint()
st.sidebar.caption(
    f"**Data:** NOAA Storm Events Database  \n"
    f"**Source:** ncei.noaa.gov/pub/data/swdi/stormevents/  \n"
    f"**Files loaded:** {_fp_count}  \n"
    f"**Data snapshot:** `{_fp_hash}` ({_fp_date})"
)

# ── apply filters ────────────────────────────────────────────────────────────
df = df_all[df_all["YEAR"].between(year_range[0], year_range[1], inclusive="both")].copy()
if selected_states:
    df = df[df["STATE"].isin(selected_states)]
if selected_events:
    df = df[df["EVENT_TYPE"].isin(selected_events)]

# ──────────────────────────────────────────────────────────────────────────────
# Navigation tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_overview, tab_damage, tab_regional, tab_timeseries, tab_multivariate = st.tabs([
    "📊 Overview",
    "💰 Damage Analysis",
    "🗺️ Regional Analysis",
    "📈 Time Series",
    "🔍 Multivariate",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Overview")

    # ─ Empty-state guard (anti-pattern: silent blank charts) ───────────────────────────
    if df.empty:
        st.warning(
            "⚠️ No records match the current filters. "
            "Try widening the year range or clearing the state / event-type selections."
        )
        st.stop()

    # ─ 5-second headline (Lecture 19: primary message in ~5 s) ─────────────────────
    top_event = (
        df.groupby("EVENT_TYPE")["TOTAL_DAMAGE"].sum().idxmax()
        if not df.empty else "N/A"
    )
    top_event_share = (
        df[df["EVENT_TYPE"] == top_event]["TOTAL_DAMAGE"].sum()
        / df["TOTAL_DAMAGE"].sum() * 100
        if df["TOTAL_DAMAGE"].sum() > 0 else 0
    )
    st.info(
        f"💡 **Key insight:** **{top_event}** accounts for "
        f"**{top_event_share:.0f}%** of total economic damage in the selected period — "
        f"despite being far less frequent than wind or hail events."
    )

    # ─ KPI cards with prior-period delta (Lecture 19: show context, not just numbers) ──
    # Build a "prior" window of equal length immediately before the selected range
    _span = year_range[1] - year_range[0]
    _prior_start = year_range[0] - _span - 1
    _prior_end = year_range[0] - 1
    df_prior = df_all[
        df_all["YEAR"].between(_prior_start, _prior_end, inclusive="both")
    ].copy()
    if selected_states:
        df_prior = df_prior[df_prior["STATE"].isin(selected_states)]
    if selected_events:
        df_prior = df_prior[df_prior["EVENT_TYPE"].isin(selected_events)]

    def _delta(curr, prev):
        """Return delta string for st.metric (None when no prior data)."""
        if prev == 0 or df_prior.empty:
            return None
        pct = (curr - prev) / abs(prev) * 100
        return f"{pct:+.0f}% vs prior period"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Total Records",
        f"{len(df):,}",
        delta=_delta(len(df), len(df_prior)),
    )
    col2.metric(
        "Unique Event Types",
        df["EVENT_TYPE"].nunique(),
    )
    col3.metric(
        "Total Property Damage",
        f"${df['DAMAGE_PROPERTY'].sum() / 1e9:.1f}B",
        delta=_delta(
            df['DAMAGE_PROPERTY'].sum(),
            df_prior['DAMAGE_PROPERTY'].sum() if not df_prior.empty else 0,
        ),
    )
    col4.metric(
        "Total Crop Damage",
        f"${df['DAMAGE_CROPS'].sum() / 1e9:.1f}B",
        delta=_delta(
            df['DAMAGE_CROPS'].sum(),
            df_prior['DAMAGE_CROPS'].sum() if not df_prior.empty else 0,
        ),
    )

    st.markdown("---")
    col_left, col_right = st.columns(2)

    # Horizontal bar chart — replaces pie (Lecture 19 § Layer 6:
    # "Pie charts: use sparingly — prefer bar charts for ≥ 4 categories")
    with col_left:
        top_n = df["EVENT_TYPE"].value_counts().head(12).reset_index()
        top_n.columns = ["EVENT_TYPE", "COUNT"]
        top_n = top_n.sort_values("COUNT")  # ascending so longest bar is at top
        fig_bar_ev = px.bar(
            top_n,
            x="COUNT",
            y="EVENT_TYPE",
            orientation="h",
            title="Top 12 Event Types by Record Count",
            labels={"COUNT": "Number of Events", "EVENT_TYPE": ""},
            color="COUNT",
            color_continuous_scale="Blues",
        )
        fig_bar_ev.update_layout(
            coloraxis_showscale=False,
            height=420,
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_bar_ev, use_container_width=True)

    # Damage distribution (log histogram)
    with col_right:
        prop_nz = df[df["DAMAGE_PROPERTY"] > 0]["DAMAGE_PROPERTY"]
        if prop_nz.empty:
            st.info("No events with non-zero property damage in this selection.")
        else:
            fig_hist = px.histogram(
                np.log10(prop_nz),
                nbins=50,
                title="Property Damage Distribution (log₁₀, non-zero events)",
                labels={"value": "log₁₀(Damage, USD)", "count": "Events"},
                color_discrete_sequence=["steelblue"],
            )
            fig_hist.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig_hist, use_container_width=True)

    # Human impact summary
    st.markdown("### Human Impact Summary")
    hi_cols = ["INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT"]
    present = [c for c in hi_cols if c in df.columns]
    if present:
        hi = df[present].sum().reset_index()
        hi.columns = ["Metric", "Total"]
        hi["Metric"] = hi["Metric"].str.replace("_", " ").str.title()
        # Pair color with pattern to be color-blind safe (Lecture 19 anti-pattern fix)
        fig_hi = px.bar(
            hi,
            x="Metric",
            y="Total",
            color="Metric",
            text="Total",
            title="Total Injuries & Deaths (filtered selection)",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_hi.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_hi.update_layout(showlegend=False, uniformtext_minsize=8)
        st.plotly_chart(fig_hi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Damage Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_damage:
    st.header("Damage Analysis")

    if df.empty:
        st.warning("⚠️ No records match the current filters.")
        st.stop()

    top_n_slider = st.slider("Number of top event types to show", 5, 30, 15, key="top_n_damage")

    # Total damage by event type
    damage_by_type = (
        df.groupby("EVENT_TYPE")[["DAMAGE_PROPERTY", "DAMAGE_CROPS"]]
        .sum()
        .reset_index()
    )
    damage_by_type["TOTAL_DAMAGE"] = damage_by_type["DAMAGE_PROPERTY"] + damage_by_type["DAMAGE_CROPS"]
    damage_by_type = damage_by_type.sort_values("TOTAL_DAMAGE", ascending=False).head(top_n_slider)

    fig_total = px.bar(
        damage_by_type,
        x="EVENT_TYPE",
        y=["DAMAGE_PROPERTY", "DAMAGE_CROPS"],
        barmode="stack",
        title=f"Top {top_n_slider} Event Types — Total Damage (Property + Crops)",
        labels={"value": "Damage (USD)", "variable": "Damage Type", "EVENT_TYPE": "Event Type"},
        color_discrete_map={"DAMAGE_PROPERTY": "#1f77b4", "DAMAGE_CROPS": "#ff7f0e"},
    )
    fig_total.update_xaxes(tickangle=45)
    st.plotly_chart(fig_total, use_container_width=True)

    st.markdown("---")

    col_avg, col_scatter = st.columns(2)

    # Average damage per event
    with col_avg:
        avg_damage = (
            df.groupby("EVENT_TYPE")[["DAMAGE_PROPERTY", "DAMAGE_CROPS"]]
            .mean()
            .reset_index()
        )
        avg_damage["AVG_TOTAL"] = avg_damage["DAMAGE_PROPERTY"] + avg_damage["DAMAGE_CROPS"]
        avg_damage = avg_damage.sort_values("AVG_TOTAL", ascending=False).head(top_n_slider)

        fig_avg = px.bar(
            avg_damage,
            x="EVENT_TYPE",
            y="AVG_TOTAL",
            title=f"Top {top_n_slider} Event Types — Average Damage per Event",
            labels={"AVG_TOTAL": "Avg Total Damage (USD)", "EVENT_TYPE": "Event Type"},
            color="AVG_TOTAL",
            color_continuous_scale="Viridis",
        )
        fig_avg.update_xaxes(tickangle=45)
        fig_avg.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_avg, use_container_width=True)

    # Frequency vs total damage scatter
    with col_scatter:
        freq_df = df["EVENT_TYPE"].value_counts().reset_index()
        freq_df.columns = ["EVENT_TYPE", "COUNT"]
        scatter_df = pd.merge(freq_df, damage_by_type[["EVENT_TYPE", "TOTAL_DAMAGE"]], on="EVENT_TYPE")

        fig_scatter = px.scatter(
            scatter_df,
            x="COUNT",
            y="TOTAL_DAMAGE",
            text="EVENT_TYPE",
            title="Event Frequency vs. Total Damage",
            labels={"COUNT": "Number of Events", "TOTAL_DAMAGE": "Total Damage (USD)"},
            log_x=True,
            log_y=True,
            size="TOTAL_DAMAGE",
            size_max=40,
            color="TOTAL_DAMAGE",
            color_continuous_scale="Reds",
        )
        fig_scatter.update_traces(textposition="top center", textfont_size=9)
        fig_scatter.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Property vs crop damage scatter (event-level, sampled)
    st.markdown("---")
    st.subheader("Property Damage vs. Crop Damage (event level)")
    st.caption(
        "💬 **About this metric** — "
        "`DAMAGE_PROPERTY` = estimated USD value of structural / infrastructure damage. "
        "`DAMAGE_CROPS` = estimated USD value of agricultural crop losses. "
        "Both are NOAA estimates; individual event accuracy varies."
    )
    both_nz = df[(df["DAMAGE_PROPERTY"] > 0) & (df["DAMAGE_CROPS"] > 0)].copy()
    sample = both_nz.sample(min(5000, len(both_nz)), random_state=42) if len(both_nz) > 0 else both_nz
    if len(sample) > 0:
        fig_pvc = px.scatter(
            sample,
            x=np.log10(sample["DAMAGE_PROPERTY"]),
            y=np.log10(sample["DAMAGE_CROPS"]),
            color="EVENT_TYPE",
            opacity=0.5,
            color_discrete_sequence=OKABE_ITO,
            title="log₁₀(Property Damage) vs log₁₀(Crop Damage) — sampled events with both > 0",
            labels={"x": "log₁₀(Property Damage, USD)", "y": "log₁₀(Crop Damage, USD)"},
        )
        fig_pvc.update_traces(marker_size=5)
        st.plotly_chart(fig_pvc, use_container_width=True)
        st.caption(
            "📊 **Scatter plot** (log₁₀ scale, events with non-zero values in both columns). "
            "The diffuse cloud confirms property and crop damage are nearly uncorrelated — "
            "different storm types drive each. "
            "Colors use the Okabe-Ito colorblind-safe palette."
        )
    else:
        st.info("No events with both non-zero property and crop damage in current selection.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Regional Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_regional:
    st.header("Regional Analysis")

    if df.empty:
        st.warning("⚠️ No records match the current filters.")
        st.stop()

    # State abbreviation lookup
    state_abbrev = {
        "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
        "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
        "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
        "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
        "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
        "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
        "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
        "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
        "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
        "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
        "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
        "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
        "WISCONSIN": "WI", "WYOMING": "WY", "DISTRICT OF COLUMBIA": "DC",
        "PUERTO RICO": "PR", "GUAM": "GU", "VIRGIN ISLANDS": "VI",
    }

    state_damage = (
        df.groupby("STATE")
        .agg(
            TOTAL_DAMAGE=("TOTAL_DAMAGE", "sum"),
            PROPERTY_DAMAGE=("DAMAGE_PROPERTY", "sum"),
            CROP_DAMAGE=("DAMAGE_CROPS", "sum"),
            EVENT_COUNT=("EVENT_TYPE", "count"),
        )
        .reset_index()
    )
    state_damage["ABBREV"] = state_damage["STATE"].str.upper().map(state_abbrev)
    state_damage_us = state_damage.dropna(subset=["ABBREV"])

    map_metric = st.selectbox(
        "Map metric",
        ["TOTAL_DAMAGE", "PROPERTY_DAMAGE", "CROP_DAMAGE", "EVENT_COUNT"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    fig_map = px.choropleth(
        state_damage_us,
        locations="ABBREV",
        locationmode="USA-states",
        color=map_metric,
        scope="usa",
        hover_name="STATE",
        hover_data={"TOTAL_DAMAGE": ":,.0f", "EVENT_COUNT": ":,", "ABBREV": False},
        color_continuous_scale="OrRd",
        title=f"{map_metric.replace('_', ' ').title()} by State",
    )
    fig_map.update_layout(height=500)
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption(
        "🗺️ **Choropleth map** — color encodes the selected metric by U.S. state using a sequential "
        "palette (darker = higher). Hover for exact values. "
        "⚠️ **Caution:** total damage is not normalized by population or land area — "
        "coastal states appear larger partly due to hurricane exposure, not event frequency."
    )

    st.markdown("---")

    col_bar_state, col_corr_state = st.columns(2)

    # Top states by total damage bar
    with col_bar_state:
        top_states = state_damage.sort_values("TOTAL_DAMAGE", ascending=False).head(15)
        fig_state_bar = px.bar(
            top_states,
            x="STATE",
            y=["PROPERTY_DAMAGE", "CROP_DAMAGE"],
            barmode="stack",
            title="Top 15 States — Total Damage",
            labels={"value": "Damage (USD)", "variable": "Type", "STATE": "State"},
            color_discrete_map={"PROPERTY_DAMAGE": "#1f77b4", "CROP_DAMAGE": "#ff7f0e"},
        )
        fig_state_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_state_bar, use_container_width=True)

    # State-level frequency vs damage correlation histogram
    with col_corr_state:
        state_event_summary = df.groupby(["STATE", "EVENT_TYPE"]).agg(
            TOTAL_DAMAGE=("TOTAL_DAMAGE", "sum"),
            EVENT_COUNT=("EVENT_TYPE", "count"),
        ).reset_index()

        state_corrs = {}
        for state, grp in state_event_summary.groupby("STATE"):
            if len(grp) > 1:
                c = grp["EVENT_COUNT"].corr(grp["TOTAL_DAMAGE"])
                if not pd.isna(c):
                    state_corrs[state] = c

        if state_corrs:
            corr_df = pd.DataFrame(state_corrs.items(), columns=["STATE", "CORRELATION"])
            fig_corr = px.histogram(
                corr_df,
                x="CORRELATION",
                nbins=20,
                title="Distribution of Frequency–Damage Correlation Across States",
                labels={"CORRELATION": "Correlation Coefficient", "count": "States"},
                color_discrete_sequence=["teal"],
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough data for state correlations in this selection.")

    # Drill-down: top event types for a selected state
    st.markdown("---")
    st.subheader("Drill-Down: Top Event Types for a Single State")
    drill_state = st.selectbox("Select a state", sorted(df["STATE"].dropna().unique()), key="drill_state")
    state_df = df[df["STATE"] == drill_state]
    state_type_dmg = (
        state_df.groupby("EVENT_TYPE")["TOTAL_DAMAGE"]
        .sum()
        .reset_index()
        .sort_values("TOTAL_DAMAGE", ascending=False)
        .head(10)
    )
    fig_drill = px.bar(
        state_type_dmg,
        x="EVENT_TYPE",
        y="TOTAL_DAMAGE",
        title=f"Top 10 Event Types by Total Damage — {drill_state.title()}",
        labels={"TOTAL_DAMAGE": "Total Damage (USD)", "EVENT_TYPE": "Event Type"},
        color="TOTAL_DAMAGE",
        color_continuous_scale="Blues",
    )
    fig_drill.update_xaxes(tickangle=45)
    fig_drill.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_drill, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Time Series
# ══════════════════════════════════════════════════════════════════════════════
with tab_timeseries:
    st.header("Time Series Analysis")

    if df.empty:
        st.warning("⚠️ No records match the current filters.")
        st.stop()

    col_ts1, col_ts2 = st.columns(2)

    # Annual event count
    with col_ts1:
        yearly = df["YEAR"].value_counts().sort_index().reset_index()
        yearly.columns = ["YEAR", "EVENT_COUNT"]
        fig_ts = px.line(
            yearly,
            x="YEAR",
            y="EVENT_COUNT",
            markers=True,
            title="Storm Event Frequency Over Time",
            labels={"YEAR": "Year", "EVENT_COUNT": "Number of Events"},
        )
        fig_ts.add_vline(x=1996, line_dash="dash", line_color="red",
                         annotation_text="1996 methodology change", annotation_position="top right")
        st.plotly_chart(fig_ts, use_container_width=True)
        st.caption(
            "⚠️ **Methodology note:** The sharp increase starting ~1996 is primarily a "
            "reporting artifact — NOAA expanded the Storm Data publication format and event "
            "categorization, not necessarily a true increase in storm frequency. "
            "Analyses comparing pre- and post-1996 counts should account for this."
        )

    # Annual total damage
    with col_ts2:
        yearly_dmg = df.groupby("YEAR")["TOTAL_DAMAGE"].sum().reset_index()
        fig_dmg_ts = px.bar(
            yearly_dmg,
            x="YEAR",
            y="TOTAL_DAMAGE",
            title="Total Economic Damage Per Year",
            labels={"YEAR": "Year", "TOTAL_DAMAGE": "Total Damage (USD)"},
            color="TOTAL_DAMAGE",
            color_continuous_scale="Reds",
        )
        fig_dmg_ts.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_dmg_ts, use_container_width=True)
        st.caption(
            "📊 **Bar chart** — total property + crop damage (USD) per year. "
            "Large spikes correspond to major hurricane landfalls (e.g., 2005 Katrina season, 2017 Harvey/Irma/Maria, 2024). "
            "Damage values are nominal USD; not inflation-adjusted."
        )

    # Monthly seasonality
    st.markdown("---")
    st.subheader("Seasonal Patterns")
    col_month1, col_month2 = st.columns(2)

    with col_month1:
        monthly = df["MONTH"].value_counts().sort_index().reset_index()
        monthly.columns = ["MONTH", "COUNT"]
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly["MONTH_NAME"] = monthly["MONTH"].map(month_names)
        fig_month = px.bar(
            monthly,
            x="MONTH_NAME",
            y="COUNT",
            title="Event Count by Month",
            labels={"MONTH_NAME": "Month", "COUNT": "Number of Events"},
            category_orders={"MONTH_NAME": list(month_names.values())},
            color="COUNT",
            color_continuous_scale="Blues",
        )
        fig_month.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_month, use_container_width=True)

    with col_month2:
        monthly_dmg = df.groupby("MONTH")["TOTAL_DAMAGE"].sum().reset_index()
        monthly_dmg["MONTH_NAME"] = monthly_dmg["MONTH"].map(month_names)
        fig_month_dmg = px.bar(
            monthly_dmg,
            x="MONTH_NAME",
            y="TOTAL_DAMAGE",
            title="Total Damage by Month",
            labels={"MONTH_NAME": "Month", "TOTAL_DAMAGE": "Total Damage (USD)"},
            category_orders={"MONTH_NAME": list(month_names.values())},
            color="TOTAL_DAMAGE",
            color_continuous_scale="Oranges",
        )
        fig_month_dmg.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_month_dmg, use_container_width=True)

    # Top event types trend (animated)
    st.markdown("---")
    st.subheader("Top Event Types — Cumulative Damage Over Time")
    top_types_for_trend = (
        df.groupby("EVENT_TYPE")["TOTAL_DAMAGE"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    trend_df = (
        df[df["EVENT_TYPE"].isin(top_types_for_trend)]
        .groupby(["YEAR", "EVENT_TYPE"])["TOTAL_DAMAGE"]
        .sum()
        .reset_index()
    )
    fig_trend = px.line(
        trend_df,
        x="YEAR",
        y="TOTAL_DAMAGE",
        color="EVENT_TYPE",
        color_discrete_sequence=OKABE_ITO,
        title="Annual Damage by Top 8 Event Types",
        labels={"YEAR": "Year", "TOTAL_DAMAGE": "Total Damage (USD)", "EVENT_TYPE": "Event Type"},
        markers=True,
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption(
        "📊 **Line chart** — annual total economic damage for the 8 highest-damage event types. "
        "Hurricane / Tropical Storm events produce large isolated spikes; "
        "frequent events (Hail, Thunderstorm Wind) produce a lower, consistent baseline. "
        "Colors use the Okabe-Ito colorblind-safe palette."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Multivariate
# ══════════════════════════════════════════════════════════════════════════════
with tab_multivariate:
    st.header("Multivariate Analysis")

    if df.empty:
        st.warning("⚠️ No records match the current filters.")
        st.stop()

    damage_factor_cols = [c for c in [
        "INJURIES_DIRECT", "INJURIES_INDIRECT",
        "DEATHS_DIRECT", "DEATHS_INDIRECT",
        "DAMAGE_PROPERTY", "DAMAGE_CROPS", "MAGNITUDE",
    ] if c in df.columns]

    corr_matrix = df[damage_factor_cols].corr()

    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Matrix — Key Damage Factors",
        aspect="auto",
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown(
        """
        **Key observations:**
        - `INJURIES_DIRECT` ↔ `DEATHS_DIRECT`: moderate positive correlation — events that injure also kill.
        - `DAMAGE_PROPERTY`: weakly correlated with injuries/deaths; property harm and human harm operate through different mechanisms.
        - `MAGNITUDE`: negligible correlation with all damage factors — event type and location matter more than raw magnitude.
        - `DAMAGE_CROPS`: nearly uncorrelated with all other factors; driven by seasonal and geographic agricultural exposure.
        """
    )

    st.markdown("---")

    # Parallel coordinates — top event types
    st.subheader("Parallel Coordinates — Event-Level Damage Profile")
    top8 = (
        df.groupby("EVENT_TYPE")["TOTAL_DAMAGE"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    pc_df = df[df["EVENT_TYPE"].isin(top8)].copy()
    pc_sample = pc_df.sample(min(3000, len(pc_df)), random_state=42)
    pc_cols = [c for c in ["DAMAGE_PROPERTY", "DAMAGE_CROPS", "INJURIES_DIRECT", "DEATHS_DIRECT"] if c in pc_sample.columns]

    # log-transform for readability
    for col in ["DAMAGE_PROPERTY", "DAMAGE_CROPS"]:
        if col in pc_sample.columns:
            pc_sample[f"log_{col}"] = np.log10(pc_sample[col].clip(lower=1))
    display_cols = [f"log_{c}" if c in ["DAMAGE_PROPERTY", "DAMAGE_CROPS"] else c
                    for c in pc_cols]

    event_type_codes = {et: i for i, et in enumerate(top8)}
    pc_sample["ET_CODE"] = pc_sample["EVENT_TYPE"].map(event_type_codes)

    fig_pc = px.parallel_coordinates(
        pc_sample,
        dimensions=display_cols,
        color="ET_CODE",
        color_continuous_scale=px.colors.qualitative.Safe,
        title="Parallel Coordinates — Top 8 Event Types (sampled, damage in log₁₀)",
        labels={f: f.replace("log_DAMAGE_", "log₁₀(").replace("_", " ").title() + (")" if f.startswith("log_") else "")
                for f in display_cols},
    )
    st.plotly_chart(fig_pc, use_container_width=True)

    # Event type legend for the parallel coordinates
    st.caption("Event type color codes: " + " | ".join(f"{i}={et}" for et, i in event_type_codes.items()))

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility footer  (checklist: "footer shows data version + deploy time")
# ──────────────────────────────────────────────────────────────────────────────
_fp_hash2, _fp_count2, _fp_date2 = data_fingerprint()
_deploy_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.markdown("---")
st.caption(
    f"**NOAA Storm Events Dashboard** · "
    f"Data snapshot `{_fp_hash2}` ({_fp_count2} files, newest {_fp_date2}) · "
    f"Rendered {_deploy_time} · "
    f"[Source](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/)"
)
