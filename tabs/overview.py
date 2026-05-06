"""Tab 1 — Overview."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.formatting import OKABE_ITO


def render_overview(
    df: pd.DataFrame,
    df_all: pd.DataFrame,
    year_range: tuple[int, int],
    selected_states: list[str],
    selected_events: list[str],
) -> None:
    st.header("Overview")

    if df.empty:
        st.warning(
            "⚠️ No records match the current filters. "
            "Try widening the year range or clearing the state / event-type selections."
        )
        st.stop()

    # ─ 5-second headline (Lecture 19: primary message in ~5 s) ──────────────
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

    # ─ Prior-period delta window ─────────────────────────────────────────────
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
        if prev == 0 or df_prior.empty:
            return None
        pct = (curr - prev) / abs(prev) * 100
        return f"{pct:+.0f}% vs prior period"

    # ─ KPI cards ────────────────────────────────────────────────────────────
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
            df["DAMAGE_PROPERTY"].sum(),
            df_prior["DAMAGE_PROPERTY"].sum() if not df_prior.empty else 0,
        ),
    )
    col4.metric(
        "Total Crop Damage",
        f"${df['DAMAGE_CROPS'].sum() / 1e9:.1f}B",
        delta=_delta(
            df["DAMAGE_CROPS"].sum(),
            df_prior["DAMAGE_CROPS"].sum() if not df_prior.empty else 0,
        ),
    )

    st.markdown("---")
    col_left, col_right = st.columns(2)

    # Horizontal bar chart (top 12 event types by count)
    with col_left:
        top_n = df["EVENT_TYPE"].value_counts().head(12).reset_index()
        top_n.columns = ["EVENT_TYPE", "COUNT"]
        top_n = top_n.sort_values("COUNT")
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
