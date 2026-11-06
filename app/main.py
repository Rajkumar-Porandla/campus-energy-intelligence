"""Campus Energy Consumption Intelligence (CEI) Interactive Dashboard.

Primary Streamlit application entrypoint implementing the 1960s Space Age
Mission Control console with 6 comprehensive views:
1. Command Center: Portfolio telemetry, chronological timeline, research status.
2. Consumption Analytics: Building telemetry, diurnal curves, peak detection, multi-facility overlay.
3. Forecasting Lab: Benchmark matrix, actual-vs-forecast curves, residual diagnostics.
4. Anomaly Observatory: Unsupervised detectors, explainable attribution, non-fault disclosure.
5. Building Intelligence: Facility profiles, peak demand stress, Energy Risk Index (0–100).
6. Energy Insights: Outlier leverage diagnostics, research synthesis, cryptographic ledger.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

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
    apply_theme,
)
from app.views.anomaly import render_anomaly_observatory
from app.views.building_intelligence import render_building_intelligence
from app.views.energy_insights import render_energy_insights
from app.views.explorer import render_consumption_analytics
from app.views.forecasting import render_forecasting_lab
from app.views.overview import render_command_center


def main():
    st.set_page_config(
        page_title="CEI Mission Control // Campus Energy Intelligence",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply 1960s Space Age Mission Control theme
    apply_theme()

    # Sidebar Navigation & Telemetry Console
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0.6rem 0 1.2rem 0;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY}; letter-spacing: -0.01em;">
                    🛰️ CEI MISSION CONTROL
                </div>
                <div style="font-size: 0.8rem; color: {COLOR_DUSTY_BLUE}; margin-top: 0.25rem;">
                    Campus Energy Intelligence & Anomaly Detection
                </div>
                <div style="margin-top: 0.75rem;">
                    <span class="cei-badge cei-badge-emerald">● SYSTEM: NOMINAL</span>
                    <span class="cei-badge cei-badge-teal">3D-REMEDIATED-V1</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"<hr style='border-color: {COLOR_BORDER}; margin: 0.5rem 0 1rem 0;'>", unsafe_allow_html=True)

        nav_choice = st.radio(
            "Mission Console Navigation:",
            [
                "1. Command Center",
                "2. Consumption Analytics",
                "3. Forecasting Lab",
                "4. Anomaly Observatory",
                "5. Building Intelligence",
                "6. Energy Insights",
            ],
            index=0,
        )

        st.markdown(f"<hr style='border-color: {COLOR_BORDER}; margin: 1.2rem 0 1rem 0;'>", unsafe_allow_html=True)

        with st.expander("📌 Dataset Architecture", expanded=False):
            st.markdown(
                """
                - **Site:** UCF Panther (BDG2)
                - **Core Facilities:** 29 Educational Buildings
                - **Sensitivity Facility:** Panther_education_Quintin
                - **Time Horizon:** 2016-01-01 to 2017-12-31 (17,544h)
                - **Training:** 2016 (8,784h)
                - **Validation:** H1 2017 (4,344h)
                - **Held-Out Test:** H2 2017 (4,416h, `not_run`)
                """
            )

        with st.expander("🛡️ Methodological Guardrails", expanded=False):
            st.markdown(
                """
                - **No Ground-Truth Fault Claims** (Unsupervised only)
                - **No Test Set Leakage** (H2 2017 strictly held out)
                - **No Quintin Distortion** (Sensitivity evaluated separately)
                - **Valid-Zero Suppression** (0 valid zeros flagged)
                - **Defensible ERI** (Operational volatility, NOT failure probability)
                """
            )

        st.markdown(
            f"""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.70rem; color: {COLOR_DUSTY_BLUE}; padding-top: 1rem; text-align: center;">
                CEI CONSOLE v2.0 // DEEPMIND PAIR<br>
                SHA-256 LEDGER: 100% IN-MEMORY
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Route View
    if nav_choice == "1. Command Center":
        render_command_center()
    elif nav_choice == "2. Consumption Analytics":
        render_consumption_analytics()
    elif nav_choice == "3. Forecasting Lab":
        render_forecasting_lab()
    elif nav_choice == "4. Anomaly Observatory":
        render_anomaly_observatory()
    elif nav_choice == "5. Building Intelligence":
        render_building_intelligence()
    elif nav_choice == "6. Energy Insights":
        render_energy_insights()


if __name__ == "__main__":
    main()
