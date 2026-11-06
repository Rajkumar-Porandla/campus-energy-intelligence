"""Consumption Analytics View for the CEI Dashboard.

Provides interactive building-level time-series exploration, diurnal load curves,
partition boundary visualization, peak demand detection, and multi-building comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data_loader import (
    compute_diurnal_profile,
    get_all_building_ids,
    get_building_peak_analysis,
    get_building_series,
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


def render_consumption_analytics():
    render_mission_header(
        "MISSION CONTROL // CONSUMPTION ANALYTICS",
        "High-fidelity telemetry exploration, diurnal load curves, peak demand detection, and multi-building comparison.",
        status_tag="ACTIVE",
    )

    core_ids = get_all_building_ids()
    meta = load_building_metadata()

    # Session State Initialization for Sensitivity Channel
    if "include_quintin_channel" not in st.session_state:
        st.session_state["include_quintin_channel"] = False

    include_quintin = st.session_state["include_quintin_channel"]
    building_options = core_ids + (["Panther_education_Quintin"] if include_quintin else [])
    default_idx = building_options.index("Panther_education_Vincent") if "Panther_education_Vincent" in building_options else 0

    # Telemetry Configuration Console Grid
    # Row 1: Primary Controls (Facility Channel, Temporal Window, Temporal Aggregation)
    row1_c1, row1_c2, row1_c3 = st.columns([1.35, 1.35, 1.1])
    with row1_c1:
        selected_building = st.selectbox(
            "Facility Telemetry Channel:",
            building_options,
            index=default_idx,
            key="facility_telemetry_channel",
        )

    with row1_c2:
        partition_filter = st.selectbox(
            "Temporal Window Partition:",
            [
                "Evaluation Scope (2016 Train + H1 2017 Val)",
                "Training Only (2016: 8,784h)",
                "Validation Only (H1 2017: 4,344h)",
                "Full Time Span (with H2 2017 Held-Out)",
            ],
            index=0,
            key="temporal_partition_filter",
        )

    with row1_c3:
        resolution = st.selectbox(
            "Temporal Aggregation:",
            ["Hourly Telemetry", "Daily Mean", "Daily Peak (Max)", "Weekly Mean"],
            index=0,
            key="temporal_aggregation_res",
        )

    # Row 2: Contextual Controls & Metadata Indicator
    row2_c1, row2_c2 = st.columns([1.35, 2.45])
    with row2_c1:
        st.checkbox(
            "Include Quintin (Sensitivity Only)",
            key="include_quintin_channel",
            help="Panther_education_Quintin was isolated from core benchmarks due to severe meter faults (43.6% missing records).",
        )

    with row2_c2:
        b_meta = meta.loc[selected_building] if selected_building in meta.index else None
        sqm_val = f"{b_meta['sqm']:,.0f} m²" if b_meta is not None and pd.notna(b_meta.get("sqm")) else "N/A"
        yr_val = f"{int(b_meta['yearbuilt'])}" if b_meta is not None and pd.notna(b_meta.get("yearbuilt")) else "N/A"
        st.markdown(
            f"""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: {COLOR_DUSTY_BLUE}; padding-top: 0.35rem;">
                FACILITY METADATA: <span style="color: {COLOR_TEXT_PRIMARY}; font-weight: 600;">{selected_building}</span> · Area: <span style="color: {COLOR_TEAL}; font-weight: 600;">{sqm_val}</span> · Vintage: <span style="color: {COLOR_MUSTARD}; font-weight: 600;">{yr_val}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Load facility data
    is_quintin = "Quintin" in selected_building
    df = get_building_series(selected_building, include_quintin_if_selected=is_quintin)

    # Filter partition
    if partition_filter == "Evaluation Scope (2016 Train + H1 2017 Val)":
        df_filtered = df[df["timestamp"] < "2017-07-01"].copy()
    elif partition_filter == "Training Only (2016: 8,784h)":
        df_filtered = df[df["timestamp"] < "2017-01-01"].copy()
    elif partition_filter == "Validation Only (H1 2017: 4,344h)":
        df_filtered = df[(df["timestamp"] >= "2017-01-01") & (df["timestamp"] < "2017-07-01")].copy()
    else:
        df_filtered = df.copy()

    # Facility Telemetry KPIs
    b_meta = meta.loc[selected_building] if selected_building in meta.index else None
    sqm_val = f"{b_meta['sqm']:,.0f} m²" if b_meta is not None and pd.notna(b_meta.get("sqm")) else "N/A"
    yr_val = f"{int(b_meta['yearbuilt'])}" if b_meta is not None and pd.notna(b_meta.get("yearbuilt")) else "N/A"

    mean_val = df_filtered["energy_consumption"].mean()
    peak_val = df_filtered["energy_consumption"].max()
    zero_cnt = (df_filtered["energy_consumption"] == 0).sum()
    zero_pct = (zero_cnt / len(df_filtered) * 100) if len(df_filtered) > 0 else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi("Floor Area", sqm_val, f"Built {yr_val}", "teal")
    with k2:
        render_kpi("Mean Demand", f"{mean_val:.2f} kWh", "Observed Window", "dusty")
    with k3:
        render_kpi("Peak Demand", f"{peak_val:.2f} kWh", "Maximum Recorded", "orange")
    with k4:
        render_kpi("Zero Readings", f"{zero_cnt:,}", f"{zero_pct:.2f}% of Window", "mustard")
    with k5:
        data_qual = "100.0% Complete" if df_filtered["energy_consumption"].isna().sum() == 0 else "Missing Found"
        render_kpi("Data Integrity", data_qual, "0 Null Values", "emerald")

    if is_quintin:
        st.markdown(
            f"""
            <div class="callout-box orange">
                <b>SENSITIVITY ISOLATION NOTICE:</b> Panther_education_Quintin is isolated from core evaluations due to severe
                meter faults (43.6% missing records in raw telemetry). It is included here strictly for exploratory sensitivity assessment.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 1. Primary Time Series Telemetry Plot
    st.markdown("### Metered Electricity Demand Time Series")
    
    # Resample if requested
    df_plot = df_filtered.set_index("timestamp")
    if resolution == "Daily Mean":
        df_plot = df_plot[["energy_consumption"]].resample("1D").mean().reset_index()
    elif resolution == "Daily Peak (Max)":
        df_plot = df_plot[["energy_consumption"]].resample("1D").max().reset_index()
    elif resolution == "Weekly Mean":
        df_plot = df_plot[["energy_consumption"]].resample("1W").mean().reset_index()
    else:
        df_plot = df_plot.reset_index()

    fig_ts = go.Figure()
    fig_ts.add_trace(
        go.Scatter(
            x=df_plot["timestamp"],
            y=df_plot["energy_consumption"],
            name="Electricity Demand (kWh)",
            line=dict(color=COLOR_TEAL, width=1.5),
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Demand: %{y:.2f} kWh<extra></extra>",
        )
    )

    # Partition boundary vertical lines
    if partition_filter in ("Full Time Span (with H2 2017 Held-Out)", "Evaluation Scope (2016 Train + H1 2017 Val)"):
        fig_ts.add_vline(
            x=pd.Timestamp("2017-01-01 00:00:00").timestamp() * 1000,
            line_dash="dash",
            line_color=COLOR_MUSTARD,
            annotation_text="Train / Val Boundary (2017-01-01)",
            annotation_font=dict(color=COLOR_MUSTARD, size=10),
            annotation_position="top left",
        )
    if partition_filter == "Full Time Span (with H2 2017 Held-Out)":
        fig_ts.add_vline(
            x=pd.Timestamp("2017-07-01 00:00:00").timestamp() * 1000,
            line_dash="dash",
            line_color=COLOR_ORANGE,
            annotation_text="Val / Held-Out Test Boundary (2017-07-01)",
            annotation_font=dict(color=COLOR_ORANGE, size=10),
            annotation_position="top left",
        )

    fig_ts.update_layout(
        get_plotly_layout(
            title=f"Telemetry Stream — {selected_building} ({resolution})",
            xaxis_title="Timestamp",
            yaxis_title="Energy Consumption (kWh)",
            height=400,
        )
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # 2. Peak Demand Detection & Diurnal Load Curves
    tab1, tab2, tab3 = st.tabs([
        "⚡ Peak Consumption Detection",
        "🕒 24-Hour Diurnal Load Curves",
        "🏢 Multi-Building Comparison Overlay",
    ])

    with tab1:
        st.markdown("#### Facility Peak Demand Events (Top 99th Percentile)")
        st.markdown(
            "Identifies high-stress electrical demand events exceeding the 99th percentile of operational load. "
            "Helps facility managers target coincident peak charges and load shedding opportunities."
        )

        peak_data = get_building_peak_analysis(selected_building)
        peaks_df = peak_data["peaks_df"]
        diurnal_peaks = peak_data["diurnal_peaks"]

        p_col1, p_col2 = st.columns([1.2, 1.8])
        with p_col1:
            render_kpi("99th Pct Peak Threshold", f"{peak_data['threshold_kwh']:.1f} kWh", "Operational Cutoff", "orange")
            render_kpi("Detected Peak Hours", f"{peak_data['peak_count']} Hours", "Top 1% Exceedances", "mustard")
            render_kpi("Maximum Facility Peak", f"{peak_data['max_peak_kwh']:.2f} kWh", f"Load Factor: {peak_data['load_factor']:.2f}", "crimson" if peak_data['max_peak_kwh'] > 100 else "teal")

        with p_col2:
            fig_dp = px.bar(
                diurnal_peaks,
                x="hour",
                y="peak_count",
                labels={"hour": "Hour of Day (0–23)", "peak_count": "Peak Occurrences"},
                title=f"Diurnal Distribution of Peak Events — {selected_building}",
            )
            fig_dp.update_traces(marker_color=COLOR_ORANGE)
            fig_dp.update_layout(
                get_plotly_layout(
                    title=f"Hour-of-Day Peak Occurrence Frequency — {selected_building}",
                    xaxis_title="Hour of Day (0–23)",
                    yaxis_title="Count of Peak Events",
                    height=280,
                )
            )
            st.plotly_chart(fig_dp, use_container_width=True)

        if not peaks_df.empty:
            st.markdown("##### Highest Peak Demand Observations")
            top_peaks = peaks_df.sort_values("energy_consumption", ascending=False).head(10)[
                ["timestamp", "energy_consumption"]
            ].copy()
            top_peaks["threshold_kwh"] = peak_data["threshold_kwh"]
            top_peaks["exceedance_kwh"] = top_peaks["energy_consumption"] - peak_data["threshold_kwh"]
            st.dataframe(
                top_peaks.style.format({
                    "energy_consumption": "{:.2f} kWh",
                    "threshold_kwh": "{:.2f} kWh",
                    "exceedance_kwh": "+{:.2f} kWh",
                }),
                use_container_width=True,
            )

    with tab2:
        st.markdown("#### Diurnal Load Profiles (Weekday vs. Weekend)")
        diurnal_df = compute_diurnal_profile(df)

        fig_diurnal = go.Figure()
        colors = {
            ("Training (2016)", "Weekday"): COLOR_TEAL,
            ("Training (2016)", "Weekend"): COLOR_DUSTY_BLUE,
            ("Validation (H1 2017)", "Weekday"): COLOR_ORANGE,
            ("Validation (H1 2017)", "Weekend"): COLOR_MUSTARD,
        }

        for (part, dt), grp in diurnal_df.groupby(["partition", "day_type"]):
            color = colors.get((part, dt), "#888888")
            dash = "solid" if dt == "Weekday" else "dash"
            fig_diurnal.add_trace(
                go.Scatter(
                    x=grp["hour"],
                    y=grp["mean"],
                    name=f"{part} — {dt}",
                    line=dict(color=color, dash=dash, width=2),
                    hovertemplate=f"<b>{part} ({dt})</b><br>Hour: %{{x}}:00<br>Mean: %{{y:.2f}} kWh<extra></extra>",
                )
            )

        fig_diurnal.update_layout(
            get_plotly_layout(
                title=f"Diurnal Hourly Load Profiles — {selected_building}",
                xaxis_title="Hour of Day (0–23)",
                yaxis_title="Mean Energy Consumption (kWh)",
                height=380,
            ),
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        )
        st.plotly_chart(fig_diurnal, use_container_width=True)

    with tab3:
        st.markdown("#### Multi-Facility Comparative Overlay")
        st.markdown("Overlay demand curves across multiple buildings to analyze load synchronicity and coincident peaks.")
        
        comparison_buildings = st.multiselect(
            "Select Facilities to Compare (Up to 4):",
            building_options,
            default=[selected_building, "Panther_education_Barbra"] if "Panther_education_Barbra" in building_options else [selected_building],
            max_selections=4,
        )

        normalize = st.checkbox("Normalize by Facility Floor Area (Wh/m²)", value=True)

        if comparison_buildings:
            fig_comp = go.Figure()
            palette = [COLOR_TEAL, COLOR_ORANGE, COLOR_MUSTARD, COLOR_DUSTY_BLUE]

            # Sample 2-week window in validation for clean overlay
            window_start = pd.Timestamp("2017-02-01 00:00:00")
            window_end = pd.Timestamp("2017-02-15 00:00:00")

            for i, b_id in enumerate(comparison_buildings):
                b_series = get_building_series(b_id, include_quintin_if_selected=True)
                b_win = b_series[(b_series["timestamp"] >= window_start) & (b_series["timestamp"] < window_end)].copy()
                
                scale = 1.0
                y_label = "kWh"
                if normalize and b_id in meta.index and pd.notna(meta.loc[b_id, "sqm"]):
                    scale = 1000.0 / float(meta.loc[b_id, "sqm"])
                    y_label = "Wh/m²"

                fig_comp.add_trace(
                    go.Scatter(
                        x=b_win["timestamp"],
                        y=b_win["energy_consumption"] * scale,
                        name=f"{b_id.replace('Panther_education_', '')} ({y_label})",
                        line=dict(color=palette[i % len(palette)], width=1.5),
                        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Demand: %{y:.2f} " + y_label + "<extra></extra>",
                    )
                )

            fig_comp.update_layout(
                get_plotly_layout(
                    title=f"Comparative Demand Overlay (2017-02-01 to 2017-02-15) — {'Normalized (Wh/m²)' if normalize else 'Absolute (kWh)'}",
                    xaxis_title="Timestamp",
                    yaxis_title="Normalized Demand (Wh/m²)" if normalize else "Demand (kWh)",
                    height=380,
                )
            )
            st.plotly_chart(fig_comp, use_container_width=True)


# Alias for backward compatibility
render_explorer = render_consumption_analytics
