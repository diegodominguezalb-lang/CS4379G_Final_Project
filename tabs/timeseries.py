"""Tab 4 — Time Series Analysis."""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.formatting import OKABE_ITO, _dollar_yaxis, _fmt_usd

_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def render_timeseries(df: pd.DataFrame) -> None:
    st.header("Time Series Analysis")

    if df.empty:
        st.warning("⚠️ No records match the current filters.")
        st.stop()

    col_ts1, col_ts2 = st.columns(2)

    # ── Annual event count ────────────────────────────────────────────────────
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
        fig_ts.add_vline(
            x=1996, line_dash="dash", line_color="red",
            annotation_text="1996 methodology change",
            annotation_position="top right",
        )
        fig_ts.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_ts, use_container_width=True)
        st.caption(
            "⚠️ **Methodology note:** The sharp increase starting ~1996 is primarily a "
            "reporting artifact — NOAA expanded the Storm Data publication format and event "
            "categorization, not necessarily a true increase in storm frequency. "
            "Analyses comparing pre- and post-1996 counts should account for this."
        )

    # ── Annual total damage ───────────────────────────────────────────────────
    with col_ts2:
        yearly_dmg = df.groupby("YEAR")["TOTAL_DAMAGE"].sum().reset_index()
        yearly_dmg["DMG_LABEL"] = yearly_dmg["TOTAL_DAMAGE"].apply(_fmt_usd)
        fig_dmg_ts = px.bar(
            yearly_dmg,
            x="YEAR",
            y="TOTAL_DAMAGE",
            title="Total Economic Damage Per Year",
            labels={"YEAR": "Year", "TOTAL_DAMAGE": "Total Damage (USD)"},
            color="TOTAL_DAMAGE",
            color_continuous_scale="Reds",
            custom_data=["DMG_LABEL"],
        )
        _dollar_yaxis(fig_dmg_ts, yearly_dmg["TOTAL_DAMAGE"].max())
        fig_dmg_ts.update_traces(hovertemplate="<b>%{x}</b><br>Total Damage: %{customdata[0]}<extra></extra>")
        fig_dmg_ts.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_dmg_ts, use_container_width=True)
        st.caption(
            "📊 **Bar chart** — total property + crop damage (USD) per year. "
            "Large spikes correspond to major hurricane landfalls (e.g., 2005 Katrina season, 2017 Harvey/Irma/Maria, 2024). "
            "Damage values are nominal USD; not inflation-adjusted."
        )

    # ── Monthly seasonality ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Seasonal Patterns")
    col_month1, col_month2 = st.columns(2)

    with col_month1:
        monthly = df["MONTH"].value_counts().sort_index().reset_index()
        monthly.columns = ["MONTH", "COUNT"]
        monthly["MONTH_NAME"] = monthly["MONTH"].map(_MONTH_NAMES)
        fig_month = px.bar(
            monthly,
            x="MONTH_NAME",
            y="COUNT",
            title="Event Count by Month",
            labels={"MONTH_NAME": "Month", "COUNT": "Number of Events"},
            category_orders={"MONTH_NAME": list(_MONTH_NAMES.values())},
            color="COUNT",
            color_continuous_scale="Blues",
        )
        fig_month.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_month, use_container_width=True)

    with col_month2:
        monthly_dmg = df.groupby("MONTH")["TOTAL_DAMAGE"].sum().reset_index()
        monthly_dmg["MONTH_NAME"] = monthly_dmg["MONTH"].map(_MONTH_NAMES)
        fig_month_dmg = px.bar(
            monthly_dmg,
            x="MONTH_NAME",
            y="TOTAL_DAMAGE",
            title="Total Damage by Month",
            labels={"MONTH_NAME": "Month", "TOTAL_DAMAGE": "Total Damage (USD)"},
            category_orders={"MONTH_NAME": list(_MONTH_NAMES.values())},
            color="TOTAL_DAMAGE",
            color_continuous_scale="Oranges",
        )
        _dollar_yaxis(fig_month_dmg, monthly_dmg["TOTAL_DAMAGE"].max())
        fig_month_dmg.update_traces(hovertemplate="<b>%{x}</b><br>Total Damage: $%{y:,.0f}<extra></extra>")
        fig_month_dmg.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_month_dmg, use_container_width=True)

    # ── Top event types trend over time ───────────────────────────────────────
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
    trend_df["DMG_LABEL"] = trend_df["TOTAL_DAMAGE"].apply(_fmt_usd)
    fig_trend = px.line(
        trend_df,
        x="YEAR",
        y="TOTAL_DAMAGE",
        color="EVENT_TYPE",
        color_discrete_sequence=OKABE_ITO,
        title="Annual Damage by Top 8 Event Types",
        labels={"YEAR": "Year", "TOTAL_DAMAGE": "Total Damage (USD)", "EVENT_TYPE": "Event Type"},
        markers=True,
        custom_data=["DMG_LABEL"],
    )
    _dollar_yaxis(fig_trend, trend_df["TOTAL_DAMAGE"].max())
    fig_trend.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Damage: %{customdata[0]}<extra></extra>")
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption(
        "📊 **Line chart** — annual total economic damage for the 8 highest-damage event types. "
        "Hurricane / Tropical Storm events produce large isolated spikes; "
        "frequent events (Hail, Thunderstorm Wind) produce a lower, consistent baseline. "
        "Colors use the Okabe-Ito colorblind-safe palette."
    )
