"""Comprehensive automated interaction testing across all 6 views of the CEI Dashboard."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

MAIN_PY = (Path(__file__).resolve().parents[1] / "app/main.py").as_posix()


@pytest.fixture(scope="module")
def app_runner():
    """Initializes AppTest runner for app/main.py."""
    at = AppTest.from_file(MAIN_PY, default_timeout=30)
    at.run()
    return at


def test_view_1_command_center(app_runner):
    at = app_runner
    at.sidebar.radio[0].set_value("1. Command Center").run()
    assert len(at.exception) == 0
    # Check title
    titles = [m.value for m in at.markdown if "COMMAND CENTER" in m.value]
    assert len(titles) > 0


def test_view_2_consumption_analytics(app_runner):
    at = app_runner
    at.sidebar.radio[0].set_value("2. Consumption Analytics").run()
    assert len(at.exception) == 0

    # Test selecting different building
    building_sb = at.selectbox[0]
    building_sb.set_value("Panther_education_Alecia").run()
    assert len(at.exception) == 0

    # Test selecting different partition
    partition_sb = at.selectbox[1]
    partition_sb.set_value("Training Only (2016: 8,784h)").run()
    assert len(at.exception) == 0

    # Test selecting different resolution
    res_sb = at.selectbox[2]
    res_sb.set_value("Daily Peak (Max)").run()
    assert len(at.exception) == 0


def test_view_3_forecasting_lab(app_runner):
    at = app_runner
    at.sidebar.radio[0].set_value("3. Forecasting Lab").run()
    assert len(at.exception) == 0

    # Test changing task
    task_sb = at.selectbox[0]
    task_sb.set_value("hourly (H=1, Secondary)").run()
    assert len(at.exception) == 0

    # Test changing lookback
    lb_sb = at.selectbox[1]
    lb_sb.set_value(48).run()
    assert len(at.exception) == 0


def test_view_4_anomaly_observatory(app_runner):
    at = app_runner
    at.sidebar.radio[0].set_value("4. Anomaly Observatory").run()
    assert len(at.exception) == 0

    # Test changing feature mode
    mode_sb = at.selectbox[0]
    mode_sb.set_value("raw_consumption (1D Univariate Load)").run()
    assert len(at.exception) == 0

    # Test changing operating point
    op_sb = at.selectbox[1]
    op_sb.set_value("0.5pct (99.5th Percentile)").run()
    assert len(at.exception) == 0


def test_view_5_building_intelligence(app_runner):
    at = app_runner
    at.sidebar.radio[0].set_value("5. Building Intelligence").run()
    assert len(at.exception) == 0
    # Check that Energy Risk Index and Disclaimer are rendered
    disclaimers = [m.value for m in at.markdown if "METHODOLOGICAL RISK SCORING DISCLAIMER" in m.value]
    assert len(disclaimers) > 0


def test_view_6_energy_insights(app_runner):
    at = app_runner
    at.sidebar.radio[0].set_value("6. Energy Insights").run()
    assert len(at.exception) == 0
    # Check that Vincent outlier content is rendered
    headings = [m.value for m in at.markdown if "The Outlier-Leverage Investigation" in m.value]
    assert len(headings) > 0
