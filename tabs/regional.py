"""Tab 3 — Regional Analysis."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.formatting import _dollar_yaxis

_STATE_ABBREV = {
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


def render_regional(df: pd.DataFrame) -> None:
    st.header("Regional Analysis")

    if df.empty:
        st.warning("⚠️ No records match the current filters.")
        st.stop()

    # ── State aggregation ─────────────────────────────────────────────────────
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
    state_damage["ABBREV"] = state_damage["STATE"].str.upper().map(_STATE_ABBREV)
    state_damage_us = state_damage.dropna(subset=["ABBREV"])

    # ── Choropleth map ────────────────────────────────────────────────────────
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
    if map_metric != "EVENT_COUNT":
        _max_map = state_damage_us[map_metric].max()
        if _max_map >= 1e12:   _cb_div, _cb_sfx = 1e12, "T"
        elif _max_map >= 1e9:  _cb_div, _cb_sfx = 1e9,  "B"
        elif _max_map >= 1e6:  _cb_div, _cb_sfx = 1e6,  "M"
        else:                   _cb_div, _cb_sfx = 1e3,  "K"
        _cb_vals = np.linspace(0, _max_map, 5)
        fig_map.update_coloraxes(
            colorbar_tickvals=_cb_vals.tolist(),
            colorbar_ticktext=[f"${v/_cb_div:.0f}{_cb_sfx}" for v in _cb_vals],
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

    # ── Top 15 states by damage ───────────────────────────────────────────────
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
        _dollar_yaxis(fig_state_bar, top_states[["PROPERTY_DAMAGE", "CROP_DAMAGE"]].sum(axis=1).max())
        fig_state_bar.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name}: $%{y:,.0f}<extra></extra>")
        st.plotly_chart(fig_state_bar, use_container_width=True)

    # ── State-level frequency–damage correlation histogram ───────────────────
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

    # ── Drill-down: top event types for a selected state ─────────────────────
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
    _dollar_yaxis(fig_drill, state_type_dmg["TOTAL_DAMAGE"].max())
    fig_drill.update_traces(hovertemplate="<b>%{x}</b><br>Total Damage: $%{y:,.0f}<extra></extra>")
    fig_drill.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_drill, use_container_width=True)
