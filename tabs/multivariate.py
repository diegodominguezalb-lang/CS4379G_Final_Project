"""Tab 5 — Multivariate Analysis."""

import pandas as pd
import plotly.express as px
import streamlit as st


def render_multivariate(df: pd.DataFrame) -> None:
    st.header("Multivariate Analysis")

    if df.empty:
        st.warning("⚠️ No records match the current filters.")
        st.stop()

    # ── Correlation heatmap ───────────────────────────────────────────────────
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
    st.caption(
        "📊 **Correlation heatmap** — Pearson correlations between key numeric columns. "
        "Red = strong positive, blue = strong negative, white = no linear relationship. "
        "Values shown to 2 decimal places; only linear relationships are captured here."
    )

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

    # ── Event-type profile heatmap ────────────────────────────────────────────
    st.subheader("Event-Type Risk Profile")
    profile_cols = {c: c for c in [
        "DAMAGE_PROPERTY", "DAMAGE_CROPS",
        "INJURIES_DIRECT", "DEATHS_DIRECT",
    ] if c in df.columns}

    top15 = (
        df.groupby("EVENT_TYPE")["TOTAL_DAMAGE"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .index.tolist()
    )
    agg = df[df["EVENT_TYPE"].isin(top15)].groupby("EVENT_TYPE").agg(
        Event_Count=("EVENT_TYPE", "count"),
        **{k: (k, "sum") for k in profile_cols}
    ).reindex(top15)

    # Rename columns for display
    agg.columns = ["Event Count", "Property Damage", "Crop Damage", "Injuries", "Deaths"]

    # Min-max normalize each column so all metrics share the same 0-1 scale
    norm = agg.copy().astype(float)
    for col in norm.columns:
        mn, mx = norm[col].min(), norm[col].max()
        norm[col] = (norm[col] - mn) / (mx - mn) if mx > mn else 0.0

    # Build hover text showing the raw values with USD formatting
    from utils.formatting import _fmt_usd
    hover = norm.copy().astype(str)
    for i, et in enumerate(top15):
        raw = agg.loc[et]
        hover.loc[et, "Event Count"]      = f"{int(raw['Event Count']):,}"
        hover.loc[et, "Property Damage"]  = _fmt_usd(raw["Property Damage"])
        hover.loc[et, "Crop Damage"]      = _fmt_usd(raw["Crop Damage"])
        hover.loc[et, "Injuries"]         = f"{int(raw['Injuries']):,}"
        hover.loc[et, "Deaths"]           = f"{int(raw['Deaths']):,}"

    fig_profile = px.imshow(
        norm,
        text_auto=False,
        color_continuous_scale="YlOrRd",
        zmin=0, zmax=1,
        title="Normalized Risk Profile — Top 15 Event Types",
        aspect="auto",
        labels={"color": "Relative intensity"},
        custom_data=[hover.values],
    )
    # Overlay the raw-value text
    for col_idx, col_name in enumerate(norm.columns):
        for row_idx, et in enumerate(top15):
            fig_profile.add_annotation(
                x=col_idx, y=row_idx,
                text=hover.loc[et, col_name],
                showarrow=False,
                font=dict(size=9, color="black"),
            )
    fig_profile.update_layout(height=520, coloraxis_showscale=True)
    st.plotly_chart(fig_profile, use_container_width=True)
    st.caption(
        "📊 **Normalized heatmap** — each column is min-max scaled to 0–1 so metrics with "
        "very different units (dollars vs. deaths) are visually comparable. "
        "Cell text shows the actual raw value. Darker = relatively higher within that metric. "
        "Rows are ordered by total damage (highest at top)."
    )
