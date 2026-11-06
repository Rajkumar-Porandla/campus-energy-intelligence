"""Building Intelligence View for the CEI Dashboard.

Presents comprehensive facility intelligence profiles, multi-dimensional
operational metrics, and defensible Energy Risk Index (ERI, 0–100) scoring.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data_loader import (
    get_all_building_ids,
    get_building_peak_analysis,
    get_building_series,
    get_portfolio_risk_scores,
    load_all_reports,
    load_building_metadata,
)
from app.theme import (
    COLOR_BORDER,
    COLOR_CRIMSON,
    COLOR_DUSTY_BLUE,
    COLOR_EMERALD,
    COLOR_MUSTARD,
    COLOR_ORANGE,
    COLOR_SURFACE,
    COLOR_TEAL,
    COLOR_TEXT_PRIMARY,
    get_plotly_layout,
    render_kpi,
    render_mission_header,
)


def render_building_intelligence():
    render_mission_header(
        "MISSION CONTROL // BUILDING INTELLIGENCE",
        "Constructed facility operational profiles, multi-dimensional demand volatility metrics, and defensible Energy Risk Index (ERI) scoring.",
        status_tag="ACTIVE",
    )

    core_ids = get_all_building_ids()
    meta = load_building_metadata()

    # Disclaimer Callout Box (Strictly enforced)
    st.markdown(
        f"""
        <div class="callout-box orange">
            <b>METHODOLOGICAL RISK SCORING DISCLAIMER:</b> The <b>Energy Risk Index (ERI)</b> is a constructed operational index (0–100)
            quantifying electrical demand volatility, peak stress, zero-shift regime exposure, anomaly frequency, and forecast uncertainty.
            <b>It is strictly an operational volatility and predictive uncertainty index, NOT a calculated probability of mechanical or equipment failure.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Risk Table calculation
    risk_df = get_portfolio_risk_scores(feature_mode="seasonal_deviation", operating_point="1.0pct")

    # Building Selector & Filter Controls
    c1, c2 = st.columns([1.5, 1.5])
    with c1:
        default_idx = core_ids.index("Panther_education_Vincent") if "Panther_education_Vincent" in core_ids else 0
        selected_bldg = st.selectbox("Inspect Facility Intelligence Profile:", core_ids, index=default_idx)

    with c2:
        tier_filter = st.selectbox(
            "Portfolio Risk Tier Filter:",
            ["All Facilities (29 Core)", "High Risk (Score ≥ 65)", "Moderate Risk (35 ≤ Score < 65)", "Low Risk (Score < 35)"],
            index=0,
        )

    # Selected Building Metrics
    b_row = risk_df[risk_df["building_id"] == selected_bldg].iloc[0]
    b_meta = meta.loc[selected_bldg] if selected_bldg in meta.index else None

    sqm_val = f"{b_meta['sqm']:,.0f} m²" if b_meta is not None and pd.notna(b_meta.get("sqm")) else "N/A"
    yr_val = f"{int(b_meta['yearbuilt'])}" if b_meta is not None and pd.notna(b_meta.get("yearbuilt")) else "N/A"

    score_val = b_row["energy_risk_index"]
    tier_name = b_row["risk_tier"]
    tier_color = "orange" if tier_name in ("High", "Elevated") else ("mustard" if tier_name == "Moderate" else "teal")

    # Top Facility KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi("Energy Risk Index", f"{score_val:.1f} / 100", f"Volatility Tier: {tier_name.upper()} (Not Failure Prob)", tier_color)
    with k2:
        render_kpi("Peak Stress Subscore", f"{b_row['peak_stress_pts']:.1f} / 20", "Peak-to-Mean Ratio", "orange")
    with k3:
        render_kpi("Volatility Subscore", f"{b_row['volatility_pts']:.1f} / 20", "Load Variance (CV)", "mustard")
    with k4:
        render_kpi("Zero Shift Subscore", f"{b_row['zero_shift_pts']:.1f} / 20", "2016 Zero Prevalence", "dusty")
    with k5:
        render_kpi("Forecast Subscore", f"{b_row['forecast_pts']:.1f} / 20", "Daily MASE Uncertainty", "teal")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🎯 Risk Component Decomposition",
        "📊 Portfolio Risk Benchmark (29 Buildings)",
        "📋 Full Portfolio Intelligence Table",
    ])

    with tab1:
        st.markdown(f"#### Subcomponent Breakdown — {selected_bldg}")
        st.markdown(
            "The constructed Energy Risk Index aggregates 5 distinct operational dimensions (each normalized to 0–20 points, strictly measuring demand volatility and forecast uncertainty, not mechanical failure probability):"
        )

        subcomponents_data = pd.DataFrame({
            "Dimension": [
                "1. Peak Demand Stress",
                "2. Load Volatility (CV)",
                "3. Zero-Shift Exposure",
                "4. Anomaly Alert Frequency",
                "5. Forecast Uncertainty",
            ],
            "Score": [
                b_row["peak_stress_pts"],
                b_row["volatility_pts"],
                b_row["zero_shift_pts"],
                b_row["anomaly_pts"],
                b_row["forecast_pts"],
            ],
            "Max Score": [20, 20, 20, 20, 20],
            "Operational Meaning": [
                "High peak demand spikes requiring utility demand management",
                "High coefficient of variation in hourly energy usage",
                "Historical 2016 zero-meter reading prevalence",
                "High frequency of statistical outlier alerts in validation",
                "High forecast errors (MASE) challenging baseline control",
            ]
        })

        fig_bar = px.bar(
            subcomponents_data,
            x="Score",
            y="Dimension",
            orientation="h",
            text="Score",
            color="Score",
            color_continuous_scale=["#568B87", "#D9AE54", "#C76B43"],
            range_x=[0, 22],
        )
        fig_bar.update_traces(texttemplate="%{text:.1f} pts", textposition="outside")
        fig_bar.update_layout(
            get_plotly_layout(
                title=f"Energy Risk Subcomponent Contributions (Max 20 pts each) — {selected_bldg}",
                xaxis_title="Assigned Points",
                yaxis_title="",
                height=320,
            ),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(subcomponents_data[["Dimension", "Score", "Operational Meaning"]], use_container_width=True)

    with tab2:
        st.markdown("#### Portfolio Risk Index Distribution across 29 Core Facilities")

        # Sort by risk score
        sorted_risk = risk_df.sort_values("energy_risk_index", ascending=False).copy()

        # Highlight selected building in the ranking chart
        sorted_risk["is_selected"] = sorted_risk["building_id"] == selected_bldg
        
        fig_dist = px.bar(
            sorted_risk,
            x="building_id",
            y="energy_risk_index",
            color="risk_tier",
            color_discrete_map={"High": COLOR_ORANGE, "Moderate": COLOR_MUSTARD, "Low": COLOR_TEAL},
            labels={"building_id": "Facility", "energy_risk_index": "Energy Risk Index (0–100)"},
        )
        fig_dist.add_hline(y=65, line_dash="dash", line_color=COLOR_ORANGE, annotation_text="High Risk (≥65)")
        fig_dist.add_hline(y=35, line_dash="dash", line_color=COLOR_MUSTARD, annotation_text="Moderate Risk (≥35)")
        
        # Add annotation for the selected building
        sel_row = sorted_risk[sorted_risk["building_id"] == selected_bldg]
        if not sel_row.empty:
            sel_score = sel_row["energy_risk_index"].values[0]
            fig_dist.add_annotation(
                x=selected_bldg,
                y=sel_score,
                text=f"Selected: {selected_bldg.replace('Panther_education_', '')} ({sel_score:.1f})",
                showarrow=True,
                arrowhead=2,
                arrowcolor=COLOR_TEXT_PRIMARY,
                font=dict(color=COLOR_TEXT_PRIMARY, size=10, family="IBM Plex Mono"),
                ax=0,
                ay=-45,
            )

        fig_dist.update_layout(
            get_plotly_layout(
                title=f"Cross-Facility Energy Risk Index Ranking (Selected: {selected_bldg})",
                xaxis_title="Facility Identifier",
                yaxis_title="Energy Risk Index (0–100)",
                height=420,
            ),
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab3:
        st.markdown("#### Full Portfolio Risk & Operational Metrics")
        st.markdown("Detailed tabular ledger of risk scores, subcomponents, and empirical metrics for all 29 core facilities.")

        filtered_table = sorted_risk.copy()
        if tier_filter == "High Risk (Score ≥ 65)":
            filtered_table = filtered_table[filtered_table["risk_tier"] == "High"]
        elif tier_filter == "Moderate Risk (35 ≤ Score < 65)":
            filtered_table = filtered_table[filtered_table["risk_tier"] == "Moderate"]
        elif tier_filter == "Low Risk (Score < 35)":
            filtered_table = filtered_table[filtered_table["risk_tier"] == "Low"]

        display_cols = [
            "building_id",
            "energy_risk_index",
            "risk_tier",
            "peak_stress_pts",
            "volatility_pts",
            "zero_shift_pts",
            "anomaly_pts",
            "forecast_pts",
        ]
        
        st.dataframe(
            filtered_table[display_cols].style.format({
                "energy_risk_index": "{:.1f}",
                "peak_stress_pts": "{:.1f}",
                "volatility_pts": "{:.1f}",
                "zero_shift_pts": "{:.1f}",
                "anomaly_pts": "{:.1f}",
                "forecast_pts": "{:.1f}",
            }),
            use_container_width=True,
            height=400,
        )


# Alias for backward compatibility
render_research_findings = render_building_intelligence
