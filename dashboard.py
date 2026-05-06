import numpy as np
import streamlit as st
from datetime import datetime, timezone

from utils.data_loader import (
    _ensure_data,
    load_data,
    data_fingerprint,
)
from tabs.overview import render_overview
from tabs.damage import render_damage
from tabs.regional import render_regional
from tabs.timeseries import render_timeseries
from tabs.multivariate import render_multivariate

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NOAA Storm Events Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data ──────────────────────────────────────────────────────────────────────
_ensure_data()
df_all = load_data()

# ── Sidebar — global filters ──────────────────────────────────────────────────
st.sidebar.title(" Storm Events")
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

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_all[df_all["YEAR"].between(year_range[0], year_range[1], inclusive="both")].copy()
if selected_states:
    df = df[df["STATE"].isin(selected_states)]
if selected_events:
    df = df[df["EVENT_TYPE"].isin(selected_events)]

# ── Navigation tabs ───────────────────────────────────────────────────────────
tab_overview, tab_damage, tab_regional, tab_timeseries, tab_multivariate = st.tabs([
    " Overview",
    " Damage Analysis",
    " Regional Analysis",
    " Time Series",
    " Multivariate",
])

with tab_overview:
    render_overview(df, df_all, year_range, selected_states, selected_events)

with tab_damage:
    render_damage(df)

with tab_regional:
    render_regional(df, df_all)

with tab_timeseries:
    render_timeseries(df)

with tab_multivariate:
    render_multivariate(df)

# ── Footer ────────────────────────────────────────────────────────────────────
_fp_hash2, _fp_count2, _fp_date2 = data_fingerprint()
_deploy_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.markdown("---")
st.caption(
    f"**NOAA Storm Events Dashboard** · "
    f"Data snapshot `{_fp_hash2}` ({_fp_count2} files, newest {_fp_date2}) · "
    f"Rendered {_deploy_time} · "
    f"[Source](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/)"
)