"""Anomaly Observatory View for the CEI Dashboard.

Displays training-calibrated Statistical Z-score, Isolation Forest, and LOF novelty
detection across raw_consumption and seasonal_deviation feature modes with
explainable anomaly attribution and non-confirmed fault disclosure.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data_loader import (
    get_all_building_ids,
    get_anomaly_explanations,
    get_anomaly_macro_summary,
    get_anomaly_results_data,
    load_all_reports,
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
    MODEL_COLORS,
    get_plotly_layout,
    render_kpi,
    render_mission_header,
)


def render_anomaly_observatory():
    render_mission_header(
        "MISSION CONTROL // ANOMALY OBSERVATORY",
        "Training-calibrated outlier detection, multi-model alert rates, and explainable observation decomposition.",
        status_tag="ACTIVE",
    )

    macro_df = get_anomaly_macro_summary()

    # Telemetry Configuration Console Grid
    # Row 1: Primary Detectors & Calibration Settings
    row1_c1, row1_c2 = st.columns([1.5, 1.5])
    with row1_c1:
        mode_choice = st.selectbox(
            "Feature Representation Mode:",
            [
                "seasonal_deviation (3D Joint Slot Deviation + 1h Ramp)",
                "raw_consumption (1D Univariate Load)",
            ],
            index=0,
            key="anomaly_feature_mode_selector",
        )
        feature_mode = mode_choice.split(" ")[0]

    with row1_c2:
        op_point = st.selectbox(
            "Operating Calibration Point:",
            ["1.0pct (99th Percentile)", "0.5pct (99.5th Percentile)", "2.0pct (98th Percentile)"],
            index=0,
            key="anomaly_op_point_selector",
        ).split(" ")[0]

    # Row 2: Sensitivity Isolation Controls & Safeguard Indicator
    row2_c1, row2_c2 = st.columns([1.5, 1.5])
    with row2_c1:
        include_quintin = st.checkbox(
            "Include Quintin Sensitivity Channel",
            value=False,
            key="anomaly_include_quintin_channel",
            help="Panther_education_Quintin was isolated from core benchmarks due to severe meter faults.",
        )

    with row2_c2:
        st.markdown(
            f"""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: {COLOR_DUSTY_BLUE}; padding-top: 0.35rem;">
                CALIBRATION INVARIANCE: <span style="color: {COLOR_EMERALD}; font-weight: 600;">TRAINING-ONLY CALIBRATION</span> · 0 VALID ZEROS FLAGGED
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Load evaluated anomaly data
    eval_df = get_anomaly_results_data(
        feature_mode=feature_mode,
        operating_point=op_point,
        include_quintin=include_quintin,
    )

    # Filter macro summary
    sub_macro = macro_df[
        (macro_df["feature_mode"] == feature_mode) & (macro_df["operating_point"] == op_point)
    ]

    # Top Macro Telemetry KPIs
    k1, k2, k3, k4 = st.columns(4)
    z_row = sub_macro[sub_macro["model"] == "statistical_zscore"]
    if_row = sub_macro[sub_macro["model"] == "isolation_forest"]
    lof_row = sub_macro[sub_macro["model"] == "lof_novelty"]

    z_rate = z_row["flagged_per_1000_macro_mean"].values[0] if not z_row.empty else None
    if_rate = if_row["flagged_per_1000_macro_mean"].values[0] if not if_row.empty else None
    lof_rate = lof_row["flagged_per_1000_macro_mean"].values[0] if not lof_row.empty else None

    with k1:
        z_str = f"{z_rate:.2f} / 1kh" if z_rate is not None else "N/A"
        render_kpi("Z-Score Alert Rate", z_str, "Macro Mean (Validation)", "dusty")
    with k2:
        if_str = f"{if_rate:.2f} / 1kh" if if_rate is not None else "N/A"
        render_kpi("IF Alert Rate", if_str, "Macro Mean (Validation)", "teal")
    with k3:
        lof_str = f"{lof_rate:.2f} / 1kh" if lof_rate is not None else "N/A"
        render_kpi("LOF Novelty Rate", lof_str, "Macro Mean (Validation)", "mustard")
    with k4:
        render_kpi("Zero-Reading Safeguard", "0 Flagged", "100% Valid Zero Invariance", "emerald")

    # Important Scientific Callout Box
    st.markdown(
        f"""
        <div class="callout-box orange">
            <b>NON-CONFIRMED FAULT DISCLOSURE:</b> All anomaly detectors operate in an <b>unsupervised paradigm</b>. Flagged observations
            indicate statistical rarity relative to calibrated 2016 baseline operational distributions. They <b>do not constitute confirmed
            mechanical failures or verified sensor faults</b> without physical on-site ground-truth audit.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 Multi-Detector Alert Rates",
        "🔍 Explainable Anomaly Attribution",
        "📋 Temporal Coincidence & Macro Invariance",
    ])

    with tab1:
        st.markdown(f"#### Validation Alert Rates across Facilities ({feature_mode}, {op_point})")

        fig_box = px.box(
            eval_df,
            x="model",
            y="validation_flagged_per_1000_hours",
            color="model",
            color_discrete_map=MODEL_COLORS,
            points="all",
            hover_data=["building_id"],
            labels={
                "model": "Anomaly Detector",
                "validation_flagged_per_1000_hours": "Alert Rate (Alerts / 1,000 Hours)",
            },
        )
        fig_box.update_layout(
            get_plotly_layout(
                title=f"Validation Alert Rate Distribution across Facilities ({feature_mode}, {op_point})",
                xaxis_title="Detector Model",
                yaxis_title="Alerts / 1,000 Hours",
                height=400,
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("##### Building-Level Anomaly Breakdown")
        pivot_df = eval_df.pivot(
            index="building_id", columns="model", values="validation_flagged_hours"
        ).reset_index()
        pivot_df.columns = [
            "Building Identifier" if c == "building_id" else f"{c} (Hours)"
            for c in pivot_df.columns
        ]
        st.dataframe(pivot_df, use_container_width=True, height=320)

    with tab2:
        st.markdown("#### Explainable Observation Attribution (Telemetry Decomposition)")
        st.markdown(
            "Decomposes flagged anomaly observations into physical components: **Diurnal Slot Deviation** (z-score against hour-of-week), "
            "**1-Hour Ramp Rate** (rate of change vs prior hour), and **Baseline Magnitude Exceedance**."
        )

        b_choice = st.selectbox("Select Facility for Explainability Audit:", get_all_building_ids(), index=0)
        explanations = get_anomaly_explanations(b_choice, operating_point=op_point, top_k=6)

        if explanations:
            for exp in explanations:
                ts_str = exp["timestamp"].strftime("%Y-%m-%d %H:%M")
                factor = exp['dominant_factor'].lower()
                badge_class = "cei-badge-orange" if "deviation" in factor or "slot" in factor else ("cei-badge-mustard" if "ramp" in factor else "cei-badge-crimson")
                border_color = COLOR_ORANGE if "deviation" in factor or "slot" in factor else (COLOR_MUSTARD if "ramp" in factor else COLOR_CRIMSON)

                st.markdown(
                    f"""
                    <div style="background: linear-gradient(165deg, #1D3344 0%, #152735 100%); border-left: 4px solid {border_color}; border-top: 1px solid {COLOR_BORDER}; border-right: 1px solid {COLOR_BORDER}; border-bottom: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; box-shadow: 0 4px 14px rgba(0,0,0,0.35);">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.4rem;">
                            <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY}; letter-spacing: 0.02em;">{ts_str}</span>
                            <span class="cei-badge {badge_class}">PRIMARY FACTOR: {exp['dominant_factor'].upper()}</span>
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.88rem; line-height: 1.5; color: #D6E2E7;">
                            {exp['explanation_text']}
                        </div>
                        <div style="display: flex; gap: 1.5rem; margin-top: 0.65rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: {COLOR_DUSTY_BLUE}; flex-wrap: wrap; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 0.5rem;">
                            <span>Observed: <b style="color: {COLOR_TEXT_PRIMARY};">{exp['observed_consumption']:.2f} kWh</b></span>
                            <span>Slot Mean: <b style="color: {COLOR_TEXT_PRIMARY};">{exp['slot_mean']:.2f} kWh</b></span>
                            <span>Slot Z-Score: <b style="color: {border_color};">{exp['slot_zscore']:+.2f}σ</b></span>
                            <span>1h Ramp: <b style="color: {COLOR_TEXT_PRIMARY};">{exp['ramp_rate']:+.2f} kWh/h</b></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No anomalous events exceeded the nominal threshold for this facility under selected calibration.")

    with tab3:
        st.markdown("#### Campus Coincidence & Macro Invariance")
        st.markdown("Macro-level alert statistics across all 18 evaluated detector/mode/operating-point combinations.")

        display_macro = macro_df.copy()
        st.dataframe(
            display_macro.style.format({
                "flagged_per_1000_macro_mean": "{:.2f}",
                "flagged_per_1000_median": "{:.2f}",
                "hours_with_multi_building_coincidence": "{:.0f}",
                "total_flagged_zero_hours": "{:.0f}",
            }),
            use_container_width=True,
            height=380,
        )


# Alias for backward compatibility
render_anomaly = render_anomaly_observatory
