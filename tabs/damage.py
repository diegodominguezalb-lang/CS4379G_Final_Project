"""Tab 2 — Damage Analysis."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.formatting import OKABE_ITO, _dollar_yaxis, _fmt_usd


def render_damage(df: pd.DataFrame) -> None:
    st.header("Damage Analysis")

    if df.empty:
        st.warning("⚠️ No records match the current filters.")
        st.stop()

    top_n_slider = st.slider("Number of top event types to show", 5, 30, 15, key="top_n_damage")

    # ── Total damage by event type ────────────────────────────────────────────
    damage_by_type = (
        df.groupby("EVENT_TYPE")[["DAMAGE_PROPERTY", "DAMAGE_CROPS"]]
        .sum()
        .reset_index()
    )
    damage_by_type["TOTAL_DAMAGE"] = damage_by_type["DAMAGE_PROPERTY"] + damage_by_type["DAMAGE_CROPS"]
    damage_by_type = damage_by_type.sort_values("TOTAL_DAMAGE", ascending=False).head(top_n_slider)

    damage_by_type_melted = damage_by_type.melt(
        id_vars="EVENT_TYPE",
        value_vars=["DAMAGE_PROPERTY", "DAMAGE_CROPS"],
        var_name="Damage Type",
        value_name="Damage",
    )
    damage_by_type_melted["DMG_LABEL"] = damage_by_type_melted["Damage"].apply(_fmt_usd)
    fig_total = px.bar(
        damage_by_type_melted,
        x="EVENT_TYPE",
        y="Damage",
        color="Damage Type",
        barmode="stack",
        title=f"Top {top_n_slider} Event Types — Total Damage (Property + Crops)",
        labels={"Damage": "Damage (USD)", "Damage Type": "Damage Type", "EVENT_TYPE": "Event Type"},
        color_discrete_map={"DAMAGE_PROPERTY": "#1f77b4", "DAMAGE_CROPS": "#ff7f0e"},
        custom_data=["DMG_LABEL"],
    )
    fig_total.update_xaxes(tickangle=45)
    _dollar_yaxis(fig_total, damage_by_type[["DAMAGE_PROPERTY", "DAMAGE_CROPS"]].sum(axis=1).max())
    fig_total.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{customdata[0]}<extra></extra>")
    st.plotly_chart(fig_total, use_container_width=True)

    st.markdown("---")

    col_avg, col_scatter = st.columns(2)

    # ── Average damage per event ──────────────────────────────────────────────
    with col_avg:
        avg_damage = (
            df.groupby("EVENT_TYPE")[["DAMAGE_PROPERTY", "DAMAGE_CROPS"]]
            .mean()
            .reset_index()
        )
        avg_damage["AVG_TOTAL"] = avg_damage["DAMAGE_PROPERTY"] + avg_damage["DAMAGE_CROPS"]
        avg_damage = avg_damage.sort_values("AVG_TOTAL", ascending=False).head(top_n_slider)
        avg_damage["DMG_LABEL"] = avg_damage["AVG_TOTAL"].apply(_fmt_usd)

        fig_avg = px.bar(
            avg_damage,
            x="EVENT_TYPE",
            y="AVG_TOTAL",
            title=f"Top {top_n_slider} Event Types — Average Damage per Event",
            labels={"AVG_TOTAL": "Avg Total Damage (USD)", "EVENT_TYPE": "Event Type"},
            color="AVG_TOTAL",
            color_continuous_scale="Viridis",
            custom_data=["DMG_LABEL"],
        )
        fig_avg.update_xaxes(tickangle=45)
        _dollar_yaxis(fig_avg, avg_damage["AVG_TOTAL"].max())
        fig_avg.update_traces(hovertemplate="<b>%{x}</b><br>Avg Damage: %{customdata[0]}<extra></extra>")
        fig_avg.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_avg, use_container_width=True)

    # ── Frequency vs total damage scatter ────────────────────────────────────
    with col_scatter:
        freq_df = df["EVENT_TYPE"].value_counts().reset_index()
        freq_df.columns = ["EVENT_TYPE", "COUNT"]
        scatter_df = pd.merge(freq_df, damage_by_type[["EVENT_TYPE", "TOTAL_DAMAGE"]], on="EVENT_TYPE")
        scatter_df["DMG_LABEL"] = scatter_df["TOTAL_DAMAGE"].apply(_fmt_usd)

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
            custom_data=["DMG_LABEL"],
        )
        fig_scatter.update_traces(
            textposition="top center",
            textfont_size=9,
            hovertemplate="<b>%{text}</b><br>Events: %{x:,}<br>Total Damage: %{customdata[0]}<extra></extra>",
        )
        fig_scatter.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Property vs crop damage scatter (event-level, sampled) ───────────────
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
