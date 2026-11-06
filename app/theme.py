"""UI theme, styling, and Plotly visualization defaults for the CEI Dashboard.

Implements the 1960s Space Age Retro-Futurism / Mission Control aesthetic
elevated with Apple Interface Design clarity and Emil Kowalski motion engineering:
- Deep Navy (#172B3A) console chassis and surfaces
- Warm Ivory (#F1E8D5) primary typography and indicators
- Burnt Orange (#C76B43), Mustard (#D9AE54), Teal (#568B87), and Dusty Blue (#91AFB5) accents
- Typography: Space Grotesk (headers), IBM Plex Mono (telemetry/metrics), Inter (body)
- GPU-accelerated micro-interactions: transform/opacity only, tactile press states, no ease-in
- Full reduced-motion and touch device accessibility support
- Mission Control bezel cards, telemetry pills, and high-contrast dark Plotly layouts.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Color Palette Constants (1960s Space Age / Mission Control Palette)
COLOR_BG = "#172B3A"           # Deep Navy chassis background
COLOR_SURFACE = "#1E3547"      # Console panel surface
COLOR_SURFACE_HOVER = "#254157" # Illuminated surface
COLOR_SURFACE_ALT = "#142533"   # Recessed chassis panel
COLOR_BORDER = "#365369"       # Bezel console border
COLOR_BORDER_SUBTLE = "#233E54"# Recessed divider line
COLOR_BORDER_ACCENT = "#568B87" # Instrument frame teal

COLOR_TEXT_PRIMARY = "#F1E8D5" # Warm Ivory
COLOR_TEXT_MUTED = "#91AFB5"   # Dusty Blue
COLOR_TEXT_DIM = "#6A8592"     # Low-intensity console readout

# Mission Control Accents
COLOR_TEAL = "#568B87"         # Subsea telemetry teal
COLOR_ORANGE = "#C76B43"       # Burnt Orange ignition/alert
COLOR_MUSTARD = "#D9AE54"      # Launch status mustard
COLOR_DUSTY_BLUE = "#91AFB5"   # Avionics dusty blue
COLOR_CRIMSON = "#D05353"      # Critical telemetry alarm
COLOR_EMERALD = "#4E9B78"      # Nominal system green

# Model and Category Colors
MODEL_COLORS = {
    "persistence": "#91AFB5",             # Dusty Blue
    "daily_seasonal_naive": "#568B87",     # Teal
    "weekly_seasonal_naive": "#D9AE54",    # Mustard
    "sarima": "#8B729E",                  # Vintage Plum
    "random_forest": "#4E9B78",           # Instrument Green
    "xgboost": "#C76B43",                 # Burnt Orange
    "statistical_zscore": "#91AFB5",      # Dusty Blue
    "isolation_forest": "#568B87",        # Teal
    "lof_novelty": "#D9AE54",             # Mustard
}


def apply_theme():
    """Inject custom CSS for the 1960s Space Age Mission Control theme."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        /* Root Design System Variables */
        :root {{
            --cei-bg: {COLOR_BG};
            --cei-surface: {COLOR_SURFACE};
            --cei-surface-hover: {COLOR_SURFACE_HOVER};
            --cei-border: {COLOR_BORDER};
            --cei-border-subtle: {COLOR_BORDER_SUBTLE};
            --cei-text-primary: {COLOR_TEXT_PRIMARY};
            --cei-text-muted: {COLOR_TEXT_MUTED};
            --cei-teal: {COLOR_TEAL};
            --cei-orange: {COLOR_ORANGE};
            --cei-mustard: {COLOR_MUSTARD};
            --cei-emerald: {COLOR_EMERALD};
            --cei-crimson: {COLOR_CRIMSON};
            --cei-dusty: {COLOR_DUSTY_BLUE};
            --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
            --ease-press: cubic-bezier(0.2, 0, 0, 1);
        }}

        /* Base Application Chassis */
        .stApp {{
            background-color: var(--cei-bg);
            color: var(--cei-text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}

        /* Header Typography - Space Grotesk */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Space Grotesk', -apple-system, sans-serif !important;
            color: var(--cei-text-primary) !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }}

        /* Sidebar Chassis & Navigation Styling */
        section[data-testid="stSidebar"] {{
            background-color: #11202B !important;
            border-right: 1px solid var(--cei-border-subtle) !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.45);
        }}
        section[data-testid="stSidebar"] .stMarkdown {{
            color: var(--cei-text-primary);
        }}

        /* Styled Sidebar Radio Channels */
        div[data-testid="stRadio"] div[role="radiogroup"] {{
            gap: 0.35rem;
        }}
        div[data-testid="stRadio"] div[role="radiogroup"] > label {{
            background: #162938;
            border: 1px solid #28445A;
            border-radius: 6px;
            padding: 0.6rem 0.85rem;
            margin-bottom: 0.25rem;
            color: var(--cei-text-muted) !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.86rem !important;
            font-weight: 500 !important;
            cursor: pointer;
            transition: transform 120ms var(--ease-press), background-color 160ms var(--ease-out), border-color 160ms var(--ease-out);
        }}
        @media (hover: hover) and (pointer: fine) {{
            div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
                background-color: #1F384C;
                border-color: var(--cei-teal);
                color: var(--cei-text-primary) !important;
                transform: translateY(-1px);
            }}
        }}
        div[data-testid="stRadio"] div[role="radiogroup"] > label:active {{
            transform: scale(0.98);
        }}
        div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
        div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
            background: linear-gradient(90deg, #1D3A4B 0%, #172D3B 100%) !important;
            border-left: 3px solid var(--cei-orange) !important;
            border-color: var(--cei-teal) !important;
            color: var(--cei-text-primary) !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
        }}

        /* Interactive Buttons - Emil Kowalski Active Feedback */
        div.stButton > button {{
            background-color: #1D3447 !important;
            color: var(--cei-text-primary) !important;
            border: 1px solid var(--cei-border) !important;
            border-radius: 5px !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.80rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            padding: 0.45rem 0.95rem !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
            transition: transform 120ms var(--ease-press), background-color 160ms var(--ease-out), border-color 160ms var(--ease-out) !important;
        }}
        @media (hover: hover) and (pointer: fine) {{
            div.stButton > button:hover {{
                background-color: #27455C !important;
                border-color: var(--cei-teal) !important;
                color: var(--cei-text-primary) !important;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
            }}
        }}
        div.stButton > button:active {{
            transform: scale(0.97) !important;
        }}

        /* Styled Dropdowns / Selectboxes */
        div[data-baseweb="select"] > div {{
            background-color: #162938 !important;
            border: 1px solid #2C485E !important;
            border-radius: 5px !important;
            color: var(--cei-text-primary) !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.85rem !important;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.2) !important;
            transition: border-color 150ms var(--ease-out) !important;
        }}
        @media (hover: hover) and (pointer: fine) {{
            div[data-baseweb="select"] > div:hover {{
                border-color: var(--cei-teal) !important;
            }}
        }}
        div[data-baseweb="select"] > div:focus-within {{
            border-color: var(--cei-orange) !important;
            box-shadow: 0 0 0 1px var(--cei-orange) !important;
        }}

        /* Styled Navigation Tabs */
        div[data-testid="stTabs"] button[data-baseweb="tab"] {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            color: var(--cei-text-muted) !important;
            padding: 0.65rem 1.1rem !important;
            border-bottom: 2px solid transparent !important;
            transition: color 150ms var(--ease-out), border-color 150ms var(--ease-out) !important;
        }}
        @media (hover: hover) and (pointer: fine) {{
            div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {{
                color: var(--cei-text-primary) !important;
            }}
        }}
        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--cei-text-primary) !important;
            border-bottom: 3px solid var(--cei-orange) !important;
        }}

        /* Mission Control KPI Telemetry Card */
        .kpi-container {{
            background: linear-gradient(165deg, #1E3547 0%, #152735 100%);
            border: 1px solid var(--cei-border);
            border-top: 3px solid var(--cei-teal);
            border-radius: 6px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 0.85rem;
            min-height: 110px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 4px 16px rgba(8, 16, 24, 0.5);
            transition: transform 160ms var(--ease-out), border-color 160ms var(--ease-out), box-shadow 160ms var(--ease-out);
            position: relative;
            overflow: hidden;
        }}
        .kpi-container.orange {{
            border-top-color: var(--cei-orange);
        }}
        .kpi-container.mustard {{
            border-top-color: var(--cei-mustard);
        }}
        .kpi-container.dusty {{
            border-top-color: var(--cei-dusty);
        }}
        .kpi-container.emerald {{
            border-top-color: var(--cei-emerald);
        }}
        .kpi-container.crimson {{
            border-top-color: var(--cei-crimson);
        }}
        @media (hover: hover) and (pointer: fine) {{
            .kpi-container:hover {{
                border-color: var(--cei-border-accent);
                transform: translateY(-2px);
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12), 0 8px 24px rgba(8, 16, 24, 0.65);
            }}
        }}
        .kpi-container:active {{
            transform: scale(0.99);
        }}
        .kpi-title {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: var(--cei-text-muted);
            margin-bottom: 0.4rem;
            font-weight: 600;
        }}
        .kpi-value {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.95rem;
            font-weight: 700;
            color: var(--cei-text-primary);
            line-height: 1.15;
            font-feature-settings: "tnum";
            font-variant-numeric: tabular-nums;
        }}
        .kpi-value.teal {{ color: var(--cei-teal); }}
        .kpi-value.orange {{ color: var(--cei-orange); }}
        .kpi-value.mustard {{ color: var(--cei-mustard); }}
        .kpi-value.dusty {{ color: var(--cei-dusty); }}
        .kpi-value.emerald {{ color: var(--cei-emerald); }}
        .kpi-value.crimson {{ color: var(--cei-crimson); }}
        .kpi-subtitle {{
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            color: var(--cei-text-muted);
            margin-top: 0.45rem;
            line-height: 1.35;
        }}

        /* Retro Mission Control Telemetry Badge Pills */
        .cei-badge {{
            display: inline-flex;
            align-items: center;
            font-family: 'IBM Plex Mono', monospace;
            padding: 0.22rem 0.6rem;
            border-radius: 4px;
            font-size: 0.70rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-right: 0.4rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        }}
        .cei-badge-teal {{
            background-color: rgba(86, 139, 135, 0.22);
            color: #C2ECE8;
            border: 1px solid var(--cei-teal);
        }}
        .cei-badge-orange {{
            background-color: rgba(199, 107, 67, 0.22);
            color: #FFC0A3;
            border: 1px solid var(--cei-orange);
        }}
        .cei-badge-mustard {{
            background-color: rgba(217, 174, 84, 0.22);
            color: #FFE699;
            border: 1px solid var(--cei-mustard);
        }}
        .cei-badge-dusty {{
            background-color: rgba(145, 175, 181, 0.22);
            color: #D3E7EC;
            border: 1px solid var(--cei-dusty);
        }}
        .cei-badge-emerald {{
            background-color: rgba(78, 155, 120, 0.22);
            color: #A3F3CA;
            border: 1px solid var(--cei-emerald);
        }}
        .cei-badge-crimson {{
            background-color: rgba(208, 83, 83, 0.22);
            color: #FFAEAE;
            border: 1px solid var(--cei-crimson);
        }}

        /* Mission Control Callout Console Frame */
        .callout-box {{
            background: linear-gradient(180deg, #1A2F3E 0%, #152735 100%);
            border-left: 4px solid var(--cei-teal);
            border-top: 1px solid var(--cei-border-subtle);
            border-right: 1px solid var(--cei-border-subtle);
            border-bottom: 1px solid var(--cei-border-subtle);
            padding: 0.95rem 1.25rem;
            border-radius: 4px;
            margin: 0.9rem 0;
            font-size: 0.88rem;
            line-height: 1.55;
            color: var(--cei-text-primary);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }}
        .callout-box.orange {{ border-left-color: var(--cei-orange); }}
        .callout-box.mustard {{ border-left-color: var(--cei-mustard); }}
        .callout-box.dusty {{ border-left-color: var(--cei-dusty); }}
        .callout-box.crimson {{ border-left-color: var(--cei-crimson); }}

        /* Mission Control Header Banner */
        .mission-header {{
            background: linear-gradient(90deg, #182E3E 0%, #142433 100%);
            border: 1px solid var(--cei-border);
            border-left: 5px solid var(--cei-orange);
            border-radius: 6px;
            padding: 1.15rem 1.45rem;
            margin-bottom: 1.35rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 4px 18px rgba(0,0,0,0.35);
        }}
        .mission-header-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--cei-text-primary);
            letter-spacing: -0.015em;
            margin: 0;
            line-height: 1.2;
        }}
        .mission-header-sub {{
            font-family: 'Inter', sans-serif;
            font-size: 0.88rem;
            color: var(--cei-text-muted);
            margin-top: 0.35rem;
            line-height: 1.4;
        }}

        /* Table & Dataframe Styling */
        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--cei-border-subtle) !important;
            border-radius: 6px !important;
            background-color: #142533 !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.35);
        }}

        /* Expander Frame */
        details[data-testid="stExpander"] {{
            background-color: #142533 !important;
            border: 1px solid var(--cei-border-subtle) !important;
            border-radius: 6px !important;
            margin-bottom: 0.75rem !important;
        }}
        details[data-testid="stExpander"] summary {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.90rem !important;
            font-weight: 600 !important;
            color: var(--cei-text-primary) !important;
        }}

        /* Streamlit widget adjustments */
        div[data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', monospace !important;
            color: var(--cei-text-primary) !important;
            font-feature-settings: "tnum";
            font-variant-numeric: tabular-nums;
        }}

        /* Standardized Widget Label Height & Alignment */
        div[data-testid="stWidgetLabel"] label,
        div[data-testid="stWidgetLabel"] p {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            color: var(--cei-text-primary) !important;
            margin-bottom: 0.35rem !important;
            min-height: 1.35rem;
            display: flex;
            align-items: flex-end;
        }}

        /* Checkbox Focus & Typography Alignment */
        div[data-testid="stCheckbox"] {{
            padding-top: 0.4rem;
        }}
        div[data-testid="stCheckbox"] label span {{
            font-family: 'Inter', sans-serif !important;
            font-size: 0.85rem !important;
            color: var(--cei-text-primary) !important;
        }}
        div[data-testid="stCheckbox"] input:focus-visible {{
            outline: 2px solid var(--cei-orange) !important;
            outline-offset: 2px !important;
        }}

        /* Responsive Mobile / Tablet Grid System */
        @media (max-width: 768px) {{
            .mission-header {{
                padding: 0.85rem 1rem !important;
                margin-bottom: 1rem !important;
            }}
            .mission-header-title {{
                font-size: 1.25rem !important;
            }}
            .kpi-container {{
                padding: 0.8rem 1rem !important;
                min-height: 85px !important;
                margin-bottom: 0.5rem !important;
            }}
            .kpi-value {{
                font-size: 1.5rem !important;
            }}
            div[data-testid="column"] {{
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
                margin-bottom: 0.5rem !important;
            }}
        }}

        /* Reduced Motion Accessibility - Emil Kowalski & Apple Accessibility standard */
        @media (prefers-reduced-motion: reduce) {{
            *, ::before, ::after {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
                transform: none !important;
            }}
            .kpi-container:hover {{
                transform: none !important;
            }}
            div.stButton > button:hover, div.stButton > button:active {{
                transform: none !important;
            }}
            div[data-testid="stRadio"] div[role="radiogroup"] > label:hover,
            div[data-testid="stRadio"] div[role="radiogroup"] > label:active {{
                transform: none !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_plotly_layout(
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 420
) -> dict:
    """Standardized retro-futuristic dark Plotly layout dict."""
    return dict(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color=COLOR_TEXT_PRIMARY, family="Space Grotesk, sans-serif"),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#162938",
        height=height,
        margin=dict(l=50, r=30, t=50 if title else 30, b=50),
        font=dict(color=COLOR_TEXT_MUTED, size=11, family="IBM Plex Mono, monospace"),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(color=COLOR_TEXT_PRIMARY, size=11, family="Inter, sans-serif")) if xaxis_title else None,
            gridcolor="#233E54",
            zerolinecolor="#233E54",
            tickfont=dict(color=COLOR_TEXT_MUTED, size=10, family="IBM Plex Mono, monospace"),
            showline=True,
            linecolor=COLOR_BORDER,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(color=COLOR_TEXT_PRIMARY, size=11, family="Inter, sans-serif")) if yaxis_title else None,
            gridcolor="#233E54",
            zerolinecolor="#233E54",
            tickfont=dict(color=COLOR_TEXT_MUTED, size=10, family="IBM Plex Mono, monospace"),
            showline=True,
            linecolor=COLOR_BORDER,
        ),
        legend=dict(
            bgcolor="rgba(19, 36, 49, 0.90)",
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            font=dict(color=COLOR_TEXT_PRIMARY, size=10, family="IBM Plex Mono, monospace"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(
            bgcolor=COLOR_SURFACE,
            bordercolor=COLOR_TEAL,
            font=dict(color=COLOR_TEXT_PRIMARY, size=11, family="IBM Plex Mono, monospace"),
        ),
    )


def render_kpi(label: str, value: str, subtitle: str = "", variant: str = "teal"):
    """Render a styled Mission Control KPI telemetry block."""
    variant_border = variant if variant in ("orange", "mustard", "dusty", "emerald", "crimson") else "teal"
    st.markdown(
        f"""
        <div class="kpi-container {variant_border}">
            <div class="kpi-title">{label}</div>
            <div class="kpi-value {variant}">{value}</div>
            {f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mission_header(title: str, subtitle: str, status_tag: str = "NOMINAL"):
    """Render a top mission control title header with telemetry status."""
    st.markdown(
        f"""
        <div class="mission-header">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <h2 class="mission-header-title">{title}</h2>
                    <div class="mission-header-sub">{subtitle}</div>
                </div>
                <div style="display: flex; gap: 0.4rem; align-items: center;">
                    <span class="cei-badge cei-badge-emerald">● STATUS: {status_tag}</span>
                    <span class="cei-badge cei-badge-teal">FRAMEWORK 3D</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
