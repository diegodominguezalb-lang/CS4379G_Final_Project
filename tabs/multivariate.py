"""Tab 5 — Multivariate Analysis."""

import numpy as np
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

    # ── Parallel coordinates ──────────────────────────────────────────────────
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
    pc_cols = [c for c in ["DAMAGE_PROPERTY", "DAMAGE_CROPS", "INJURIES_DIRECT", "DEATHS_DIRECT"]
               if c in pc_sample.columns]

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
    st.caption("Event type color codes: " + " | ".join(f"{i}={et}" for et, i in event_type_codes.items()))
