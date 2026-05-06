"""Tab 3 — Regional Analysis."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.formatting import _dollar_yaxis, _fmt_usd

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


def render_regional(df: pd.DataFrame, df_all: pd.DataFrame | None = None) -> None:
    st.header("Regional Analysis")

    if df.empty:
        st.warning(" No records match the current filters.")
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
        " **Choropleth map** — color encodes the selected metric by U.S. state using a sequential "
        "palette (darker = higher). Hover for exact values. "
        " **Caution:** total damage is not normalized by population or land area — "
        "coastal states appear larger partly due to hurricane exposure, not event frequency."
    )

    st.markdown("---")

    col_bar_state, col_corr_state = st.columns(2)

    # ── Top 15 states by damage ───────────────────────────────────────────────
    with col_bar_state:
        top_states = state_damage.sort_values("TOTAL_DAMAGE", ascending=False).head(15)
        top_states_melted = top_states.melt(
            id_vars="STATE",
            value_vars=["PROPERTY_DAMAGE", "CROP_DAMAGE"],
            var_name="Type",
            value_name="Damage",
        )
        top_states_melted["DMG_LABEL"] = top_states_melted["Damage"].apply(_fmt_usd)
        fig_state_bar = px.bar(
            top_states_melted,
            x="STATE",
            y="Damage",
            color="Type",
            barmode="stack",
            title="Top 15 States — Total Damage",
            labels={"Damage": "Damage (USD)", "Type": "Type", "STATE": "State"},
            color_discrete_map={"PROPERTY_DAMAGE": "#1f77b4", "CROP_DAMAGE": "#ff7f0e"},
            custom_data=["DMG_LABEL"],
        )
        fig_state_bar.update_xaxes(tickangle=45)
        _dollar_yaxis(fig_state_bar, top_states[["PROPERTY_DAMAGE", "CROP_DAMAGE"]].sum(axis=1).max())
        fig_state_bar.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{customdata[0]}<extra></extra>")
        st.plotly_chart(fig_state_bar, use_container_width=True)
        st.caption(
            " **Stacked bar chart** — top 15 states by total economic damage, "
            "split into property (blue) and crop (orange). "
            "Gulf Coast and Atlantic states dominate due to hurricane exposure; "
            "Great Plains states show higher crop damage shares from drought and hail."
        )

    # ── State-level frequency–damage correlation histogram ───────────────────
    with col_corr_state:
        _corr_source = df_all if df_all is not None else df
        state_event_summary = _corr_source.groupby(["STATE", "EVENT_TYPE"]).agg(
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
            _US_STATES = {
                "ALABAMA","ALASKA","ARIZONA","ARKANSAS","CALIFORNIA","COLORADO",
                "CONNECTICUT","DELAWARE","FLORIDA","GEORGIA","HAWAII","IDAHO",
                "ILLINOIS","INDIANA","IOWA","KANSAS","KENTUCKY","LOUISIANA",
                "MAINE","MARYLAND","MASSACHUSETTS","MICHIGAN","MINNESOTA",
                "MISSISSIPPI","MISSOURI","MONTANA","NEBRASKA","NEVADA",
                "NEW HAMPSHIRE","NEW JERSEY","NEW MEXICO","NEW YORK",
                "NORTH CAROLINA","NORTH DAKOTA","OHIO","OKLAHOMA","OREGON",
                "PENNSYLVANIA","RHODE ISLAND","SOUTH CAROLINA","SOUTH DAKOTA",
                "TENNESSEE","TEXAS","UTAH","VERMONT","VIRGINIA","WASHINGTON",
                "WEST VIRGINIA","WISCONSIN","WYOMING","DISTRICT OF COLUMBIA",
            }
            corr_df = pd.DataFrame(state_corrs.items(), columns=["STATE", "CORRELATION"])
            corr_df = corr_df[corr_df["STATE"].str.upper().isin(_US_STATES)]
            corr_df = corr_df.sort_values("CORRELATION", ascending=True)
            fig_corr = px.bar(
                corr_df,
                x="CORRELATION",
                y="STATE",
                orientation="h",
                title="Frequency–Damage Correlation by State",
                labels={"CORRELATION": "Pearson r (frequency vs. damage)", "STATE": ""},
                color="CORRELATION",
                color_continuous_scale="RdYlGn",
                range_color=[-1, 1],
            )
            fig_corr.update_layout(
                coloraxis_showscale=False,
                height=len(corr_df) * 18 + 80,
                margin=dict(l=10, r=10, t=50, b=10),
                yaxis=dict(tickfont=dict(size=10)),
                xaxis=dict(side="top", title=dict(standoff=8)),
            )
            fig_corr.update_traces(
                hovertemplate="<b>%{y}</b><br>r = %{x:.2f}<extra></extra>"
            )
            with st.container(height=480):
                st.plotly_chart(fig_corr, use_container_width=True)
            st.caption(
                " **Sorted bar chart** — each bar is one state; length and color show the Pearson r "
                "between event-type frequency and total damage within that state. "
                "Green (r ≈ 1): frequent event types are also the most damaging. "
                "Red (r ≈ 0 or negative): damage is decoupled from frequency — rare events (e.g., hurricanes) dominate costs."
            )
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
    state_type_dmg["DMG_LABEL"] = state_type_dmg["TOTAL_DAMAGE"].apply(_fmt_usd)
    fig_drill = px.bar(
        state_type_dmg,
        x="EVENT_TYPE",
        y="TOTAL_DAMAGE",
        title=f"Top 10 Event Types by Total Damage — {drill_state.title()}",
        labels={"TOTAL_DAMAGE": "Total Damage (USD)", "EVENT_TYPE": "Event Type"},
        color="TOTAL_DAMAGE",
        color_continuous_scale="Blues",
        custom_data=["DMG_LABEL"],
    )
    fig_drill.update_xaxes(tickangle=45)
    _dollar_yaxis(fig_drill, state_type_dmg["TOTAL_DAMAGE"].max())
    fig_drill.update_traces(hovertemplate="<b>%{x}</b><br>Total Damage: %{customdata[0]}<extra></extra>")
    fig_drill.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_drill, use_container_width=True)
    st.caption(
        " **Bar chart** — top 10 event types by total damage for the selected state. "
        "Use this to compare which hazards are most costly in a specific region, "
        "and how that differs from the national picture above."
    )
