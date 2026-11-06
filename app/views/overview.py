"""Command Center View for the CEI Dashboard.

Displays portfolio KPIs, campus partition timeline, observed electricity
telemetry, and building inventory table with 1960s Mission Control styling.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data_loader import (
    load_all_reports,
    load_building_metadata,
    load_electricity_core,
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


def render_command_center():
    render_mission_header(
        "MISSION CONTROL // COMMAND CENTER",
        "Operational telemetry, chronological partitions, and portfolio inventory across 29 UCF Panther educational facilities.",
        status_tag="NOMINAL",
    )

    reports = load_all_reports()
    p3c = reports["phase3c"]
    cov = p3c["coverage_and_regime_synthesis"]["coverage_macro_summary"]
    zs = p3c["coverage_and_regime_synthesis"]["zero_frequency_shift_summary"]
    meta = load_building_metadata()
    profiles = pd.DataFrame(p3c["cross_building_heterogeneity_synthesis"]["building_profiles"])

    # 1. Primary Mission Control Telemetry Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi("Core Portfolio", "29 Buildings", "UCF Panther Educational Site", "teal")
    with col2:
        render_kpi("Time Horizon", "17,544 Hours", "2 Full Years (2016–2017)", "dusty")
    with col3:
        render_kpi("Telemetry Volume", "508,776 Records", "0.00% Missing in Core Data", "mustard")
    with col4:
        render_kpi("2016 Zero Shift", "381.02 / 1,000h", "Drops to 0.00 in H1 2017", "orange")

    # 2. Portfolio Total Consumption Notice & Aggregation Caveat
    st.markdown(
        f"""
        <div class="callout-box dusty">
            <b>PORTFOLIO AGGREGATION ADVISORY:</b> Campus-wide total energy consumption represents the unweighted sum of hourly meter
            readings across disparate facilities (ranging from 1,280 m² to 29,788 m²). It is provided strictly as a macro-descriptive
            reference. Research evaluations, forecasting MASE, and anomaly alert rates are calculated <b>strictly per-building</b> without
            cross-facility averaging of raw kWh.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Chronological Experiment Partition Architecture
    st.markdown("### Chronological Experiment Partition Architecture")
    st.markdown(
        """
        The CEI research framework enforces strict temporal partitioning to prevent data leakage and benchmark
        models under realistic operational deployment conditions.
        """
    )

    timeline_cols = st.columns(3)
    with timeline_cols[0]:
        st.markdown(
            f"""
            <div style="background-color: #1A2F3E; border-top: 3px solid {COLOR_TEAL}; border-radius: 4px; padding: 1rem; border: 1px solid {COLOR_BORDER};">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {COLOR_TEAL}; font-weight: 700;">PARTITION 01</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY}; margin: 0.2rem 0;">Training Set (2016)</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: {COLOR_DUSTY_BLUE};">2016-01-01 to 2016-12-31</div>
                <div style="margin-top: 0.6rem; font-size: 0.84rem; color: #CBD5E1;">
                    ● <b>8,784 Hours</b> (Leap Year)<br>
                    ● Baseline scaling denominator derivation<br>
                    ● High zero-reading prevalence (381.02/1,000h)<br>
                    ● Seasonal slot & quantile threshold calibration
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with timeline_cols[1]:
        st.markdown(
            f"""
            <div style="background-color: #1A2F3E; border-top: 3px solid {COLOR_MUSTARD}; border-radius: 4px; padding: 1rem; border: 1px solid {COLOR_BORDER};">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {COLOR_MUSTARD}; font-weight: 700;">PARTITION 02</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY}; margin: 0.2rem 0;">Validation Set (H1 2017)</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: {COLOR_DUSTY_BLUE};">2017-01-01 to 2017-06-30</div>
                <div style="margin-top: 0.6rem; font-size: 0.84rem; color: #CBD5E1;">
                    ● <b>4,344 Hours</b> (Primary Benchmark)<br>
                    ● Forecast models: Persistence, Naive, SARIMA, RF, XGB<br>
                    ● Anomaly detectors: Z-score, IF, LOF Novelty<br>
                    ● Regime shift: 0 valid zeros observed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with timeline_cols[2]:
        st.markdown(
            f"""
            <div style="background-color: #1A2F3E; border-top: 3px solid {COLOR_ORANGE}; border-radius: 4px; padding: 1rem; border: 1px solid {COLOR_BORDER};">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {COLOR_ORANGE}; font-weight: 700;">PARTITION 03 (SEALED)</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY}; margin: 0.2rem 0;">Held-Out Test Set (H2 2017)</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: {COLOR_DUSTY_BLUE};">2017-07-01 to 2017-12-31</div>
                <div style="margin-top: 0.6rem; font-size: 0.84rem; color: #CBD5E1;">
                    ● <b>4,416 Hours</b> (Strictly Preserved)<br>
                    ● Execution status: <code>not_run</code><br>
                    ● Zero contamination guarantee<br>
                    ● Reserved for pre-registered final deployment
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Research Verification Status Grid
    st.markdown("### Research Verification & Integrity Ledger")
    status_cols = st.columns(6)
    with status_cols[0]:
        st.markdown(
            f"""
            <div style="background-color: #152533; border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 0.6rem; text-align: center;">
                <div style="font-size: 0.7rem; color: {COLOR_DUSTY_BLUE};">PHASE 3A</div>
                <div style="font-weight: 700; font-size: 0.82rem; color: {COLOR_TEXT_PRIMARY};">Canonical Data</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #8CEAB8; margin-top: 0.3rem;">● PASSED</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_cols[1]:
        st.markdown(
            f"""
            <div style="background-color: #152533; border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 0.6rem; text-align: center;">
                <div style="font-size: 0.7rem; color: {COLOR_DUSTY_BLUE};">PHASE 3B.1</div>
                <div style="font-weight: 700; font-size: 0.82rem; color: {COLOR_TEXT_PRIMARY};">Baselines Frozen</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #8CEAB8; margin-top: 0.3rem;">● VALIDATED</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_cols[2]:
        st.markdown(
            f"""
            <div style="background-color: #152533; border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 0.6rem; text-align: center;">
                <div style="font-size: 0.7rem; color: {COLOR_DUSTY_BLUE};">PHASE 3B.2</div>
                <div style="font-weight: 700; font-size: 0.82rem; color: {COLOR_TEXT_PRIMARY};">Candidate ML</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #8CEAB8; margin-top: 0.3rem;">● RECURSIVE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_cols[3]:
        st.markdown(
            f"""
            <div style="background-color: #152533; border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 0.6rem; text-align: center;">
                <div style="font-size: 0.7rem; color: {COLOR_DUSTY_BLUE};">PHASE 3B.3</div>
                <div style="font-weight: 700; font-size: 0.82rem; color: {COLOR_TEXT_PRIMARY};">Anomaly Detect</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #8CEAB8; margin-top: 0.3rem;">● ZERO-LEAKAGE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_cols[4]:
        st.markdown(
            f"""
            <div style="background-color: #152533; border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 0.6rem; text-align: center;">
                <div style="font-size: 0.7rem; color: {COLOR_DUSTY_BLUE};">PHASE 3C</div>
                <div style="font-weight: 700; font-size: 0.82rem; color: {COLOR_TEXT_PRIMARY};">Synthesis</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #8CEAB8; margin-top: 0.3rem;">● REMEDIATED</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_cols[5]:
        st.markdown(
            f"""
            <div style="background-color: #152533; border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 0.6rem; text-align: center;">
                <div style="font-size: 0.7rem; color: {COLOR_DUSTY_BLUE};">PHASE 3D</div>
                <div style="font-weight: 700; font-size: 0.82rem; color: {COLOR_TEXT_PRIMARY};">Reproducibility</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #8CEAB8; margin-top: 0.3rem;">● 100% SHA-256</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Core Building Telemetry & Characteristics Table
    st.markdown("### Campus Building Portfolio Telemetry")
    st.markdown(
        "Interactive inventory and physical scale analysis of all 29 core educational facilities with physical metadata, load metrics, "
        "and regime-shift statistics."
    )

    merged_table = profiles.merge(
        meta[["sqm", "yearbuilt"]], left_on="building_id", right_index=True, how="left"
    )
    display_df = merged_table[[
        "building_id",
        "sqm",
        "yearbuilt",
        "daily_baseline_mae",
        "train_valid_zero_per_1000_hours",
        "daily_best_mase",
        "anom_1pct_seasonal_lof_per_1000h",
    ]].copy()

    display_df.columns = [
        "Building Identifier",
        "Gross Area (m²)",
        "Year Built",
        "Daily Baseline MAE",
        "2016 Zero/1kh",
        "Daily Best MASE",
        "LOF Alert / 1kh",
    ]

    t1, t2 = st.tabs(["📋 Portfolio Facility Inventory", "📊 Scale & Vintage Architecture"])

    with t1:
        st.dataframe(
            display_df.style.format({
                "Gross Area (m²)": "{:,.0f}",
                "Year Built": "{:.0f}",
                "Daily Baseline MAE": "{:.2f}",
                "2016 Zero/1kh": "{:.1f}",
                "Daily Best MASE": "{:.3f}",
                "LOF Alert / 1kh": "{:.2f}",
            }),
            use_container_width=True,
            height=380,
        )

    with t2:
        fig_scale = px.scatter(
            merged_table,
            x="sqm",
            y="daily_baseline_mae",
            size="sqm",
            color="yearbuilt",
            hover_name="building_id",
            hover_data=["daily_best_mase", "anom_1pct_seasonal_lof_per_1000h"],
            color_continuous_scale=["#568B87", "#D9AE54", "#C76B43"],
            labels={
                "sqm": "Gross Floor Area (m²)",
                "daily_baseline_mae": "Daily Baseline MAE (kWh)",
                "yearbuilt": "Construction Year",
            },
        )
        fig_scale.update_layout(
            get_plotly_layout(
                title="Facility Scale vs. Baseline Operational Energy Scale (Bubble Size: Area, Color: Vintage)",
                xaxis_title="Gross Floor Area (m²)",
                yaxis_title="Daily Seasonal Naive Baseline MAE (kWh)",
                height=380,
            )
        )
        st.plotly_chart(fig_scale, use_container_width=True)


# Alias for backward compatibility
render_overview = render_command_center
