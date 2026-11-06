"""Forecasting Lab View for the CEI Dashboard.

Compares Persistence, Seasonal-Naive, Random Forest, SARIMA, and XGBoost models
across hourly, daily, and weekly horizons and 24h, 48h, and 168h lookbacks with
actual-vs-predicted curves and residual diagnostics.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data_loader import (
    get_all_building_ids,
    get_forecasting_macro_summary,
    get_forecasting_models_data,
    get_validation_forecast_sample,
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


def render_forecasting_lab():
    render_mission_header(
        "MISSION CONTROL // FORECASTING LAB",
        "Multi-horizon model benchmark, common-lookback invariance, and residual diagnostics across 29 facilities.",
        status_tag="EVALUATED",
    )

    macro_df = get_forecasting_macro_summary()

    # Controls Row
    c1, c2, c3 = st.columns([1.5, 1.2, 1.3])
    with c1:
        task_choice = st.selectbox(
            "Forecasting Horizon / Task:",
            ["daily_profile (H=24, Primary Task)", "hourly (H=1, Secondary)", "weekly_profile (H=168, Exploratory)"],
            index=0,
        )
        task = task_choice.split(" ")[0]

    with c2:
        lookback = st.selectbox("Lookback Window (Hours):", [168, 48, 24], index=0)

    with c3:
        metric = st.selectbox(
            "Evaluation Metric:",
            ["mase (Mean Absolute Scaled Error)", "mae (kWh)", "rmse (kWh)", "mean_error (Bias kWh)"],
            index=0,
        )
        metric_col = metric.split(" ")[0]

    # Load evaluated models for this task and lookback
    eval_df = get_forecasting_models_data(task=task, lookback_hours=lookback)

    # Filter macro summary for this slice
    sub_macro = macro_df[(macro_df["task"] == task) & (macro_df["lookback_hours"] == lookback)]

    # Top Macro Telemetry KPIs
    k1, k2, k3, k4 = st.columns(4)
    naive_row = sub_macro[sub_macro["model"].isin(["daily_seasonal_naive", "persistence", "weekly_seasonal_naive"])]
    naive_mase = naive_row["mase_macro_mean"].values[0] if not naive_row.empty else None
    naive_raw_name = naive_row["model"].values[0] if not naive_row.empty else "Baseline"
    naive_label = naive_raw_name.replace("_", " ").title()

    best_row = sub_macro.sort_values("mase_macro_mean").iloc[0] if not sub_macro.empty else None
    best_model_name = best_row["model"].replace("_", " ").title() if best_row is not None else "Benchmark"
    best_mase_val = best_row["mase_macro_mean"] if best_row is not None else 0.0
    best_mae_val = best_row["mae_macro_mean"] if best_row is not None else 0.0

    # Determine horizon dynamic
    is_ml_winner = "Forest" in best_model_name or "Xgboost" in best_model_name
    dynamic_verdict = "ML Beats Baseline" if is_ml_winner else "Baseline Wins Macro"

    with k1:
        naive_str = f"{naive_mase:.3f}" if naive_mase is not None else "N/A"
        render_kpi(f"{naive_label} MASE", naive_str, "Benchmark Baseline", "dusty")
    with k2:
        render_kpi("Lowest Macro MASE", f"{best_mase_val:.3f}", f"Top Model: {best_model_name}", "teal")
    with k3:
        render_kpi(f"{best_model_name} MAE", f"{best_mae_val:.2f} kWh", "Unweighted Macro Mean", "mustard")
    with k4:
        render_kpi("Macro Dynamic", dynamic_verdict, f"Slice: {task} (L={lookback}h)", "orange" if is_ml_winner else "teal")

    # Important Scientific Callout
    st.markdown(
        f"""
        <div class="callout-box dusty">
            <b>SCIENTIFIC CONTEXT — MACRO PORTFOLIO MEAN VS. PER-BUILDING WINS:</b><br>
            ● <b>Macro Portfolio Mean:</b> On the unweighted macro mean across all 29 core facilities, simple seasonal baselines achieve
            the lowest MASE for multi-step profiles (<b>Daily Seasonal Naive</b> achieves <b>MASE = 1.644</b> on H=24 vs. RF: 1.909; 
            <b>Weekly Seasonal Naive</b> achieves <b>MASE = 1.937</b> on H=168 vs. RF: 2.491). <b>Random Forest</b> achieves the lowest
            macro MASE strictly on the immediate 1-hour horizon (<b>H=1 MASE = 0.937</b> vs. Persistence: 1.017).<br>
            ● <b>Per-Building Heterogeneity:</b> Pairwise per-building comparisons show Random Forest outperforming naive baselines on specific
            high-load predictable facilities, but large forecasting errors on volatile or regime-shifting buildings elevate its unweighted macro mean.
            <b>No single model is universally superior across all horizons and buildings.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 Cross-Building Distribution",
        "📈 Actual vs Forecast Curves",
        "📋 Full Macro Benchmark Matrix",
    ])

    with tab1:
        st.markdown(f"#### Model Comparison: {metric_col.upper()} across 29 Core Buildings")

        fig_box = px.box(
            eval_df,
            x="model",
            y=metric_col,
            color="model",
            color_discrete_map=MODEL_COLORS,
            points="all",
            hover_data=["building_id"],
            labels={"model": "Model", metric_col: metric_col.upper()},
        )
        fig_box.update_layout(
            get_plotly_layout(
                title=f"{metric_col.upper()} Distribution across 29 Buildings ({task}, L={lookback}h)",
                xaxis_title="Model",
                yaxis_title=metric_col.upper(),
                height=420,
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("##### Building-Level Performance Breakdown")
        pivot_df = eval_df.pivot(index="building_id", columns="model", values=metric_col).reset_index()
        st.dataframe(
            pivot_df.style.format({col: "{:.3f}" for col in pivot_df.columns if col != "building_id"}),
            use_container_width=True,
            height=320,
        )

    with tab2:
        st.markdown("#### Actual vs. Baseline Forecast Demonstration (Validation Partition)")
        st.markdown(
            "Visualizes true metered electricity consumption against Daily Seasonal Naive (lag 24h) and Persistence (lag 1h) "
            "forecast baselines across a representative 7-day validation window (2017-03-01 to 2017-03-08)."
        )

        b_select = st.selectbox("Select Facility for Forecast Inspection:", get_all_building_ids(), index=0)
        fc_sample = get_validation_forecast_sample(b_select, start_date="2017-03-01", days=7)

        fig_fc = go.Figure()
        fig_fc.add_trace(
            go.Scatter(
                x=fc_sample["timestamp"],
                y=fc_sample["energy_consumption"],
                name="Actual Telemetry (Ground Truth)",
                line=dict(color=COLOR_TEXT_PRIMARY, width=2.5),
                hovertemplate="<b>Actual: %{y:.2f} kWh</b><extra></extra>",
            )
        )
        fig_fc.add_trace(
            go.Scatter(
                x=fc_sample["timestamp"],
                y=fc_sample["forecast_daily_seasonal_naive"],
                name="Daily Seasonal Naive Baseline (Lag 24h)",
                line=dict(color=COLOR_TEAL, width=1.8, dash="dash"),
                hovertemplate="<b>Daily Naive: %{y:.2f} kWh</b><extra></extra>",
            )
        )
        fig_fc.add_trace(
            go.Scatter(
                x=fc_sample["timestamp"],
                y=fc_sample["forecast_persistence"],
                name="Persistence Baseline (Lag 1h)",
                line=dict(color=COLOR_MUSTARD, width=1.2, dash="dot"),
                hovertemplate="<b>Persistence: %{y:.2f} kWh</b><extra></extra>",
            )
        )

        fig_fc.update_layout(
            get_plotly_layout(
                title=f"Actual vs. Forecast Trajectory — {b_select} (Validation Window: 2017-03-01 to 2017-03-08)",
                xaxis_title="Timestamp",
                yaxis_title="Energy Consumption (kWh)",
                height=380,
            )
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Residual Diagnostics for the backtest window
        fc_sample["residual_naive"] = fc_sample["energy_consumption"] - fc_sample["forecast_daily_seasonal_naive"]
        valid_res = fc_sample["residual_naive"].dropna()
        sample_mae = valid_res.abs().mean()
        sample_rmse = (valid_res ** 2).mean() ** 0.5
        sample_bias = valid_res.mean()

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            render_kpi("Sample Window MAE", f"{sample_mae:.2f} kWh", "Daily Naive 7-Day Window", "teal")
        with rc2:
            render_kpi("Sample Window RMSE", f"{sample_rmse:.2f} kWh", "Root Mean Squared Error", "dusty")
        with rc3:
            render_kpi("Sample Mean Bias", f"{sample_bias:+.2f} kWh", "Unbiased: ~0.0 kWh", "mustard")

        fig_res = go.Figure()
        fig_res.add_trace(
            go.Bar(
                x=fc_sample["timestamp"],
                y=fc_sample["residual_naive"],
                name="Forecast Error (Actual - Naive)",
                marker_color=fc_sample["residual_naive"].map(lambda v: COLOR_ORANGE if v > 0 else COLOR_TEAL),
                hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Residual: %{y:+.2f} kWh<extra></extra>",
            )
        )
        fig_res.add_hline(y=0, line_dash="solid", line_color=COLOR_BORDER, line_width=1)
        fig_res.update_layout(
            get_plotly_layout(
                title=f"Forecast Residual Trajectory (Actual − Daily Naive) — {b_select}",
                xaxis_title="Timestamp",
                yaxis_title="Residual Error (kWh)",
                height=240,
            )
        )
        st.plotly_chart(fig_res, use_container_width=True)

        st.markdown(
            f"""
            <div style="font-size: 0.8rem; color: {COLOR_DUSTY_BLUE}; font-family: 'IBM Plex Mono', monospace;">
                STATUS: VALIDATION BACKTEST EVALUATION // STRICT DATA SPLIT PRESERVED (NOT A FORWARD FUTURE RUN)
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab3:
        st.markdown("#### Full Macro Benchmark Summary Matrix")
        st.markdown("Unweighted building-level macro metrics across all 3 tasks, 3 lookbacks, and 4 models (36 configurations).")

        display_macro = macro_df.copy()
        st.dataframe(
            display_macro.style.format({
                "mae_macro_mean": "{:.2f}",
                "mae_median": "{:.2f}",
                "rmse_macro_mean": "{:.2f}",
                "rmse_median": "{:.2f}",
                "mase_macro_mean": "{:.3f}",
                "mase_median": "{:.3f}",
                "mean_error_macro_mean": "{:.2f}",
                "mean_error_median": "{:.2f}",
            }),
            use_container_width=True,
            height=400,
        )


# Alias for backward compatibility
render_forecasting = render_forecasting_lab
