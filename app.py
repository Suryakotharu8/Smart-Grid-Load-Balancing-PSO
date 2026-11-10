"""
app.py - "Substation" UI for Smart Grid Load Balancing Using PSO.
Dark (night substation) and light (daylight schematic) themes, switchable from the sidebar.
Application logic is unchanged; only the presentation layer was redesigned.
"""
from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

from grid_model import (
    GridConfig, Area, DistributionLine,
    compute_line_stats, create_initial_allocation,
    get_normal_scenario, get_peak_demand_scenario, get_unbalanced_scenario,
    DEFAULT_AREA_NAMES,
)
from fitness import FitnessWeights, compute_fitness_components
from pso import PSOConfig
from optimizer import run_optimization
from utils import (
    fmt_mw, fmt_pct, status_emoji, status_text,
    validate_pso_params, check_feasibility,
    allocation_to_dataframe, results_summary_dataframe,
    dataframe_to_csv_bytes, generate_pdf_report,
)
from visualization import (
    set_theme,
    plot_line_loads_comparison, plot_capacity_vs_load, plot_area_demand,
    plot_allocation_heatmap, plot_convergence, plot_line_utilization,
    plot_distribution_breakdown, plot_before_after_radar,
    plot_utilization_comparison, plot_loss_analysis,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Page Configuration
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Smart Grid | Load Optimization Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Session State Management
# ═══════════════════════════════════════════════════════════════════════════════

def _init_session_state():
    if "grid_config" not in st.session_state:
        st.session_state.grid_config = get_normal_scenario()
    if "opt_results" not in st.session_state:
        st.session_state.opt_results = None
    if "optimization_run" not in st.session_state:
        st.session_state.optimization_run = False
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

_init_session_state()


def _get_config() -> GridConfig:
    return st.session_state.grid_config


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Theme System  (dark = "Night Substation", light = "Daylight Schematic")
# ═══════════════════════════════════════════════════════════════════════════════

THEMES = {
    "dark": {
        "scheme":      "dark",
        "bg":          "#090B16",
        "surface":     "#121730",
        "surface-2":   "#1A2040",
        "sidebar-bg":  "linear-gradient(180deg, #0D1128 0%, #090B16 100%)",
        "line":        "#262E56",
        "ink":         "#ECEEFF",
        "muted":       "#8F97C4",
        "accent":      "#8B7BFF",
        "accent-2":    "#C26BFF",
        "accent-soft": "rgba(139, 123, 255, 0.16)",
        "amber":       "#FFB020",
        "ok":          "#4ADE9C",
        "warn":        "#FFB020",
        "bad":         "#FF5C7A",
        "glow-a":      "rgba(124, 92, 255, 0.22)",
        "glow-b":      "rgba(255, 176, 32, 0.07)",
        "grid-line":   "rgba(143, 151, 196, 0.055)",
        "shadow":      "0 8px 28px rgba(0, 0, 0, 0.35)",
        "shadow-hover": "0 14px 38px rgba(139, 123, 255, 0.22)",
        "badge-ok-bg":   "rgba(74, 222, 156, 0.14)",
        "badge-warn-bg": "rgba(255, 176, 32, 0.15)",
        "badge-bad-bg":  "rgba(255, 92, 122, 0.16)",
    },
    "light": {
        "scheme":      "light",
        "bg":          "#F3F1FB",
        "surface":     "#FFFFFF",
        "surface-2":   "#F7F5FE",
        "sidebar-bg":  "linear-gradient(180deg, #FFFFFF 0%, #F6F3FF 100%)",
        "line":        "#DDD8F0",
        "ink":         "#1A1640",
        "muted":       "#6B6791",
        "accent":      "#5B3DF5",
        "accent-2":    "#A24CF0",
        "accent-soft": "rgba(91, 61, 245, 0.09)",
        "amber":       "#E08600",
        "ok":          "#0E9F6E",
        "warn":        "#C77100",
        "bad":         "#E11D48",
        "glow-a":      "rgba(124, 92, 255, 0.16)",
        "glow-b":      "rgba(255, 176, 32, 0.12)",
        "grid-line":   "rgba(91, 61, 245, 0.055)",
        "shadow":      "0 6px 22px rgba(60, 40, 140, 0.08)",
        "shadow-hover": "0 14px 34px rgba(91, 61, 245, 0.18)",
        "badge-ok-bg":   "rgba(14, 159, 110, 0.12)",
        "badge-warn-bg": "rgba(224, 134, 0, 0.14)",
        "badge-bad-bg":  "rgba(225, 29, 72, 0.11)",
    },
}

_theme_name = "dark" if st.session_state.dark_mode else "light"
_T = THEMES[_theme_name]
set_theme(_theme_name)  # keep Plotly charts in sync with the UI theme

# Semantic colour tokens (resolved by CSS variables, so they follow the active theme)
C_INK = "var(--ink)"
C_OK = "var(--ok)"
C_WARN = "var(--warn)"
C_BAD = "var(--bad)"
C_MUTED = "var(--muted)"

_STATIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ── Base ───────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color-scheme: var(--scheme);
}
.stApp { background: var(--bg) !important; color: var(--ink) !important; }

/* Blueprint grid + ambient glow behind everything */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    background-image:
        radial-gradient(900px 520px at 88% -8%, var(--glow-a), transparent 62%),
        radial-gradient(760px 520px at -6% 108%, var(--glow-b), transparent 60%),
        linear-gradient(var(--grid-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-line) 1px, transparent 1px) !important;
    background-size: auto, auto, 46px 46px, 46px 46px !important;
    background-attachment: fixed !important;
}
.main .block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px;
}
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden !important; }

/* Text colours that Streamlit would otherwise leave dark */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stCaptionContainer"],
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp label { color: var(--ink) !important; }

/* ── Sidebar ────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] > div { background: transparent !important; }
section[data-testid="stSidebar"] div.block-container,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 1.4rem !important;
}

.brand-container {
    display: flex; align-items: center; gap: 13px;
    padding: 0.25rem 0.25rem 1.3rem 0.25rem;
    margin-bottom: 1.2rem;
    border-bottom: 1px solid var(--line);
}
.brand-icon {
    width: 42px; height: 42px; border-radius: 13px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 0 0 var(--accent-soft), 0 6px 18px rgba(124, 92, 255, 0.35);
    animation: brandPulse 3.2s ease-in-out infinite;
}
.stApp .brand-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.22rem; font-weight: 700; color: var(--ink) !important;
    line-height: 1.15; letter-spacing: -0.02em;
}
.stApp .brand-subtitle {
    font-size: 0.76rem; font-weight: 500; color: var(--muted) !important;
    margin-top: 2px;
}
.sidebar-note {
    font-size: 0.74rem; color: var(--muted); margin: 0.4rem 0 0.2rem 0.2rem;
}

/* Navigation (radio rebuilt as a menu) */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] div[role="radiogroup"] { gap: 6px !important; }
div[data-testid="stRadio"] label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    width: 100%;
    cursor: pointer !important;
    transition: background .18s ease, border-color .18s ease, transform .18s ease !important;
}
div[data-testid="stRadio"] label > div:first-child { display: none !important; }
div[data-testid="stRadio"] label p {
    color: var(--muted) !important; font-weight: 600 !important; font-size: 0.9rem !important;
}
div[data-testid="stRadio"] label:hover {
    background: var(--surface-2) !important;
    border-color: var(--line) !important;
    transform: translateX(3px);
}
div[data-testid="stRadio"] label:hover p { color: var(--ink) !important; }
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(90deg, var(--accent-soft), transparent 130%) !important;
    border-color: var(--line) !important;
    box-shadow: inset 3px 0 0 var(--accent) !important;
}
div[data-testid="stRadio"] label[data-checked="true"] p,
div[data-testid="stRadio"] label:has(input:checked) p { color: var(--ink) !important; font-weight: 700 !important; }

/* ── Page header ────────────────────────────────────────────────────── */
.page-header {
    position: relative;
    display: flex; justify-content: space-between; align-items: flex-start;
    padding-bottom: 1.35rem; margin-bottom: 1.6rem;
    border-bottom: 1px solid var(--line);
}
/* a pulse of "current" travelling along the header rule */
.page-header::after {
    content: ""; position: absolute; left: 0; bottom: -1px; width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent 0%, var(--accent) 22%, var(--amber) 50%, var(--accent-2) 78%, transparent 100%);
    background-size: 240% 100%;
    animation: currentFlow 5.5s linear infinite;
    opacity: .9;
}
.stApp .page-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; color: var(--ink) !important;
    margin: 0; line-height: 1.12; letter-spacing: -0.03em;
}
.stApp .page-subtitle {
    font-size: 0.93rem; color: var(--muted) !important;
    margin: 6px 0 0 0; max-width: 62ch;
}
.status-pill {
    display: inline-flex; align-items: center; gap: 9px;
    padding: 7px 16px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 700; color: var(--ink);
    border: 1px solid var(--line);
    background: var(--surface);
    box-shadow: var(--shadow);
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; position: relative; }
.status-dot::after {
    content: ""; position: absolute; inset: -4px; border-radius: 50%;
    border: 1.5px solid currentColor; opacity: 0; animation: ping 2.2s ease-out infinite;
}
.status-dot-green { background: var(--ok);   color: var(--ok); }
.status-dot-amber { background: var(--warn); color: var(--warn); }
.status-dot-red   { background: var(--bad);  color: var(--bad); }

/* ── KPI cards ──────────────────────────────────────────────────────── */
.kpi-card {
    --rail: var(--accent);
    position: relative; overflow: hidden;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1.1rem 1.25rem 1.1rem 1.45rem;
    display: flex; flex-direction: column; justify-content: space-between;
    box-shadow: var(--shadow);
    height: 100%; min-height: 118px;
    transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
    animation: riseIn .55s cubic-bezier(.2, .8, .2, 1) both;
}
/* signal rail */
.kpi-card::before {
    content: ""; position: absolute; left: 0; top: 14px; bottom: 14px; width: 4px;
    border-radius: 0 4px 4px 0; background: var(--rail);
    box-shadow: 0 0 14px var(--rail);
}
/* soft corner glow */
.kpi-card::after {
    content: ""; position: absolute; right: -40px; top: -40px; width: 120px; height: 120px;
    border-radius: 50%; background: radial-gradient(circle, var(--rail) 0%, transparent 70%);
    opacity: 0.10; transition: opacity .25s ease;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-hover); border-color: var(--rail); }
.kpi-card:hover::after { opacity: 0.22; }

[data-testid="column"]:nth-child(2) .kpi-card { animation-delay: .06s; }
[data-testid="column"]:nth-child(3) .kpi-card { animation-delay: .12s; }
[data-testid="column"]:nth-child(4) .kpi-card { animation-delay: .18s; }
[data-testid="column"]:nth-child(5) .kpi-card { animation-delay: .24s; }

.kpi-label-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
.kpi-label { font-size: 0.8rem; font-weight: 600; color: var(--muted); }
.kpi-icon {
    width: 30px; height: 30px; border-radius: 9px; font-size: 0.95rem;
    display: inline-flex; align-items: center; justify-content: center;
    background: var(--accent-soft);
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem; font-weight: 700; color: var(--ink);
    line-height: 1.15; letter-spacing: -0.02em;
}
.kpi-subtext { font-size: 0.76rem; font-weight: 500; color: var(--muted); margin-top: 5px; }

/* ── Panels & section headers ───────────────────────────────────────── */
.panel-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem; font-weight: 600; color: var(--ink);
    margin: 0.4rem 0 1rem 0;
    display: flex; align-items: center; gap: 10px;
}
.panel-header::before {
    content: ""; width: 4px; height: 18px; border-radius: 3px;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

/* ── Tabs ───────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: var(--surface-2) !important;
    border: 1px solid var(--line) !important; border-radius: 14px !important; padding: 4px !important;
    overflow-x: auto;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important; color: var(--muted) !important;
    font-weight: 600 !important; font-size: 0.86rem !important;
    padding: 8px 16px !important; background: transparent !important;
    transition: all .18s ease !important;
}
.stTabs [data-baseweb="tab"] p { color: inherit !important; }
.stTabs [data-baseweb="tab"]:hover { color: var(--ink) !important; }
.stTabs [aria-selected="true"] {
    background: var(--surface) !important; color: var(--accent) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12) !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Buttons ────────────────────────────────────────────────────────── */
.stButton > button {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%) !important;
    color: #FFFFFF !important; font-weight: 700 !important; font-size: 0.9rem !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.6rem 1.35rem !important;
    transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
    box-shadow: 0 6px 18px rgba(124, 92, 255, 0.30) !important;
}
.stButton > button p, .stButton > button span { color: #FFFFFF !important; }
.stButton > button::after {
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(110deg, transparent 30%, rgba(255, 255, 255, .30) 50%, transparent 70%);
    transform: translateX(-130%); transition: transform .65s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 12px 28px rgba(124, 92, 255, 0.42) !important; filter: brightness(1.06); }
.stButton > button:hover::after { transform: translateX(130%); }
.stButton > button:active { transform: translateY(0); }
.stButton > button:disabled { opacity: .42; filter: saturate(.3); box-shadow: none !important; transform: none; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 2px solid var(--amber) !important; outline-offset: 2px;
}

.stDownloadButton > button {
    background: var(--surface) !important; border: 1px solid var(--line) !important;
    color: var(--ink) !important; font-weight: 600 !important; font-size: 0.86rem !important;
    border-radius: 12px !important; padding: 0.55rem 1.15rem !important;
    transition: all .18s ease !important; box-shadow: var(--shadow) !important;
}
.stDownloadButton > button p, .stDownloadButton > button span { color: var(--ink) !important; }
.stDownloadButton > button:hover {
    border-color: var(--accent) !important; transform: translateY(-2px);
    box-shadow: var(--shadow-hover) !important;
}
.stDownloadButton > button:hover p { color: var(--accent) !important; }

/* ── Inputs ─────────────────────────────────────────────────────────── */
div[data-baseweb="input"], div[data-baseweb="base-input"] {
    background: var(--surface-2) !important;
    border-radius: 10px !important;
}
div[data-baseweb="input"] { border: 1px solid var(--line) !important; transition: border-color .15s, box-shadow .15s; }
div[data-baseweb="input"]:focus-within {
    border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
div[data-baseweb="input"] input { color: var(--ink) !important; font-weight: 600 !important; font-size: 0.9rem !important; background: transparent !important; }
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
    background: var(--surface-2) !important; color: var(--muted) !important;
}
[data-testid="stNumberInputStepUp"]:hover, [data-testid="stNumberInputStepDown"]:hover {
    background: var(--accent-soft) !important; color: var(--accent) !important;
}

/* Sliders */
div[data-testid="stSlider"] label p { color: var(--muted) !important; font-size: 0.84rem !important; font-weight: 600 !important; }
div[data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important; border: 3px solid var(--surface) !important;
    box-shadow: 0 0 0 1px var(--accent), 0 0 14px var(--accent-soft) !important;
}
div[data-baseweb="slider"] [data-testid="stSliderThumbValue"], div[data-testid="stSliderThumbValue"] { color: var(--accent) !important; font-weight: 700; }
div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] { color: var(--muted) !important; }

/* Toggle */
label[data-baseweb="checkbox"] { padding: 6px 4px; }
label[data-baseweb="checkbox"] p { color: var(--ink) !important; font-weight: 600 !important; font-size: 0.88rem !important; }

/* ── Progress ───────────────────────────────────────────────────────── */
div[data-testid="stProgress"] > div > div { background: var(--surface-2) !important; border-radius: 999px; }
div[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 55%, var(--amber) 100%) !important;
    background-size: 200% 100% !important; animation: currentFlow 2.4s linear infinite;
    border-radius: 999px;
}
div[data-testid="stProgress"] p { color: var(--muted) !important; font-size: 0.82rem !important; }

/* ── Alerts ─────────────────────────────────────────────────────────── */
div[data-testid="stAlert"] {
    background: var(--surface) !important; border: 1px solid var(--line) !important;
    border-left: 4px solid var(--accent) !important; border-radius: 14px !important;
    box-shadow: var(--shadow);
}
div[data-testid="stAlert"] * { color: var(--ink) !important; }
div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) { border-left-color: var(--ok) !important; background: var(--badge-ok-bg) !important; }
div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) { border-left-color: var(--warn) !important; background: var(--badge-warn-bg) !important; }
div[data-testid="stAlert"]:has([data-testid="stAlertContentError"])   { border-left-color: var(--bad) !important;  background: var(--badge-bad-bg) !important; }

/* ── Charts sit on glass panels ─────────────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 18px; padding: 0.6rem 0.5rem 0.2rem 0.5rem;
    box-shadow: var(--shadow); overflow: hidden;
    transition: border-color .22s ease, box-shadow .22s ease;
}
div[data-testid="stPlotlyChart"]:hover { border-color: var(--accent); box-shadow: var(--shadow-hover); }

/* ── Tables ─────────────────────────────────────────────────────────── */
.table-wrap {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 18px; overflow-x: auto; box-shadow: var(--shadow);
    margin-bottom: 1rem;
}
table.sg-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
table.sg-table thead th {
    text-align: left; font-weight: 700; font-size: 0.78rem; color: var(--muted);
    padding: 13px 16px; background: var(--surface-2); border-bottom: 1px solid var(--line);
    white-space: nowrap;
}
table.sg-table tbody td, table.sg-table tbody th {
    padding: 12px 16px; color: var(--ink); border-bottom: 1px solid var(--line);
    font-weight: 500; font-variant-numeric: tabular-nums; white-space: nowrap; text-align: left;
}
table.sg-table tbody th { font-weight: 700; }
table.sg-table tbody tr:last-child td, table.sg-table tbody tr:last-child th { border-bottom: none; }
table.sg-table tbody tr { transition: background .15s ease; }
table.sg-table tbody tr:hover { background: var(--accent-soft); }
.table-wrap.accent-last td:last-child { color: var(--accent); font-weight: 700; }

.badge { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 0.74rem; font-weight: 700; }
.badge-ok   { color: var(--ok);   background: var(--badge-ok-bg); }
.badge-warn { color: var(--warn); background: var(--badge-warn-bg); }
.badge-bad  { color: var(--bad);  background: var(--badge-bad-bg); }

/* ── Scrollbars ─────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Motion ─────────────────────────────────────────────────────────── */
@keyframes riseIn { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
@keyframes currentFlow { from { background-position: 240% 0; } to { background-position: -140% 0; } }
@keyframes ping { 0% { transform: scale(.6); opacity: .8; } 80%, 100% { transform: scale(1.9); opacity: 0; } }
@keyframes brandPulse {
    0%, 100% { box-shadow: 0 0 0 0 var(--accent-soft), 0 6px 18px rgba(124, 92, 255, 0.35); }
    50%      { box-shadow: 0 0 0 9px transparent,       0 6px 22px rgba(124, 92, 255, 0.50); }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""


def _build_css(theme: dict) -> str:
    tokens = "".join(f"--{k}: {v};" for k, v in theme.items())
    return f"<style>:root {{{tokens}}}{_STATIC_CSS}</style>"


st.markdown(_build_css(_T), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div class="brand-container">'
        '<div class="brand-icon">'
        '<svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>'
        '</div>'
        '<div><div class="brand-title">Smart Grid</div>'
        '<div class="brand-subtitle">Load optimization</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard Overview",
            "Grid Configuration",
            "PSO Optimization",
            "Before vs After",
            "Analytics",
        ],
        format_func=lambda p: {
            "Dashboard Overview": "📊  Dashboard Overview",
            "Grid Configuration": "⚙️  Grid Configuration",
            "PSO Optimization": "⚡  PSO Optimization",
            "Before vs After": "🔄  Before vs After",
            "Analytics": "📈  Analytics",
        }[p],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-note'>Appearance</div>", unsafe_allow_html=True)
    st.toggle("🌙  Dark mode", key="dark_mode")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def render_kpi_card(title: str, value: str, subtext: str, icon: str = "⚡", color: str = C_INK):
    rail = "var(--accent)" if color == C_INK else color
    return (
        f'<div class="kpi-card" style="--rail: {rail};">'
        f'<div class="kpi-label-row"><span class="kpi-label">{title}</span>'
        f'<span class="kpi-icon">{icon}</span></div>'
        f'<div class="kpi-value" style="color: {color};">{value}</div>'
        f'<div class="kpi-subtext">{subtext}</div>'
        f'</div>'
    )


def page_header(title: str, subtitle: str, right_html: str = ""):
    st.markdown(
        f'<div class="page-header"><div>'
        f'<h1 class="page-title">{title}</h1>'
        f'<p class="page-subtitle">{subtitle}</p>'
        f'</div><div>{right_html}</div></div>',
        unsafe_allow_html=True,
    )


def panel_header(text: str):
    st.markdown(f"<div class='panel-header'>{text}</div>", unsafe_allow_html=True)


def spacer(px: int):
    st.markdown(f"<div style='height: {px}px;'></div>", unsafe_allow_html=True)


def show_chart(fig):
    # theme=None -> Streamlit must not override the theme-aware Plotly styling
    st.plotly_chart(fig, use_container_width=True, theme=None)


def show_table(df: pd.DataFrame, index: bool = False, accent_last: bool = False, escape: bool = True,
               float_format=None):
    html = df.to_html(
        index=index, escape=escape, border=0, classes="sg-table",
        float_format=float_format,
    )
    extra = " accent-last" if accent_last else ""
    st.markdown(f"<div class='table-wrap{extra}'>{html}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Dashboard Overview
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Dashboard Overview":
    config = _get_config()
    results = st.session_state.opt_results

    if results is not None:
        allocation = results.optimized_allocation
        line_stats = results.optimized_line_stats
        sys_status_html = '<span class="status-pill"><span class="status-dot status-dot-green"></span> PSO Optimized</span>'
    else:
        allocation = create_initial_allocation(config)
        line_stats = compute_line_stats(allocation, config)
        sys_status_html = '<span class="status-pill"><span class="status-dot status-dot-amber"></span> Baseline Dispatch</span>'

    total_load = float(allocation.sum())
    avg_util = float(np.mean([s["utilization_pct"] for s in line_stats]))
    max_util = float(max(s["utilization_pct"] for s in line_stats))
    overloaded_count = sum(1 for s in line_stats if s["overloaded"])

    page_header("Smart Grid Overview", "Intelligent electricity distribution and optimization platform.", sys_status_html)

    # KPI Cards (Row 1)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(render_kpi_card("Total Demand", fmt_mw(config.total_demand), "Target load requested", "⚡"), unsafe_allow_html=True)
    with k2:
        st.markdown(render_kpi_card("Allocated Load", fmt_mw(total_load), f"Demand match: {(total_load / max(config.total_demand, 1e-6) * 100):.1f}%", "🔌"), unsafe_allow_html=True)
    with k3:
        st.markdown(render_kpi_card("Distribution Lines", str(config.num_lines), f"Total Cap: {fmt_mw(config.total_capacity)}", "🌐"), unsafe_allow_html=True)
    with k4:
        st.markdown(render_kpi_card("Consuming Areas", str(config.num_areas), "Active load centers", "🏙️"), unsafe_allow_html=True)

    spacer(14)

    # KPI Cards (Row 2)
    k5, k6, k7, k8 = st.columns(4)
    with k5:
        util_color = C_OK if avg_util <= 85 else C_WARN
        st.markdown(render_kpi_card("Average Utilization", fmt_pct(avg_util), "Target benchmark: 50-80%", "📊", util_color), unsafe_allow_html=True)
    with k6:
        max_color = C_BAD if max_util > 100 else (C_WARN if max_util > 85 else C_OK)
        st.markdown(render_kpi_card("Maximum Utilization", fmt_pct(max_util), "Peak line loading", "📈", max_color), unsafe_allow_html=True)
    with k7:
        ol_color = C_OK if overloaded_count == 0 else C_BAD
        ol_status = "Safe operation" if overloaded_count == 0 else f"{overloaded_count} critical line(s)"
        st.markdown(render_kpi_card("Overloaded Lines", str(overloaded_count), ol_status, "⚠️", ol_color), unsafe_allow_html=True)
    with k8:
        opt_text = "PSO Active" if results is not None else "Pending Optimization"
        opt_color = C_OK if results is not None else C_MUTED
        st.markdown(render_kpi_card("Optimization Status", opt_text, "Heuristic convergence", "🧬", opt_color), unsafe_allow_html=True)

    spacer(24)

    # Line Status Table & Utilization Visualizer
    col_table, col_chart = st.columns([1.1, 1.2])

    with col_table:
        panel_header("⚡ Distribution Line Status")

        status_rows = []
        for s in line_stats:
            if s["overloaded"]:
                badge = '<span class="badge badge-bad">OVERLOADED</span>'
            elif s["utilization_pct"] > 85:
                badge = '<span class="badge badge-warn">HIGH LOAD</span>'
            else:
                badge = '<span class="badge badge-ok">NORMAL</span>'
            status_rows.append({
                "Line": s["name"],
                "Capacity": f"{s['capacity_mw']:.1f} MW",
                "Allocated": f"{s['load_mw']:.1f} MW",
                "Loading": f"{s['utilization_pct']:.1f}%",
                "Headroom": f"{s['remaining_mw']:.1f} MW",
                "Status": badge,
            })
        df_status = pd.DataFrame(status_rows)
        show_table(df_status, escape=False)

    with col_chart:
        panel_header("📊 Real-Time Line Utilization")
        fig_util = plot_line_utilization(line_stats)
        show_chart(fig_util)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Grid Configuration
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Grid Configuration":
    config = _get_config()

    page_header("Grid Configuration", "Configure consuming area demands, distribution line ratings, and benchmark scenarios.")

    # Scenario Selection Bar
    panel_header("📦 Load Benchmark Scenarios")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        if st.button("🟢 Normal Demand", use_container_width=True):
            st.session_state.grid_config = get_normal_scenario()
            st.session_state.opt_results = None
            st.rerun()
    with sc2:
        if st.button("🔴 Peak Demand", use_container_width=True):
            st.session_state.grid_config = get_peak_demand_scenario()
            st.session_state.opt_results = None
            st.rerun()
    with sc3:
        if st.button("⚠️ Unbalanced Grid", use_container_width=True):
            st.session_state.grid_config = get_unbalanced_scenario()
            st.session_state.opt_results = None
            st.rerun()
    with sc4:
        if st.button("🔄 Reset Configuration", use_container_width=True):
            st.session_state.grid_config = get_normal_scenario()
            st.session_state.opt_results = None
            st.rerun()

    spacer(18)

    # Two clear sections
    col_demand, col_lines = st.columns([1.2, 1.0])

    with col_demand:
        panel_header("🏙️ Section A: Electricity Demand Configuration (10 Areas)")
        new_areas = []
        d_cols = st.columns(2)
        for i, area in enumerate(config.areas):
            with d_cols[i % 2]:
                val = st.number_input(
                    f"{area.name} (MW)",
                    min_value=0.0,
                    max_value=500.0,
                    value=float(area.demand_mw),
                    step=5.0,
                    key=f"cfg_demand_{i}",
                )
                new_areas.append(Area(name=area.name, demand_mw=val))

    with col_lines:
        panel_header("🔌 Section B: Distribution Line Configuration (5 Lines)")
        new_lines = []
        for j, ln in enumerate(config.lines):
            c_val = st.number_input(
                f"{ln.name} Capacity (MW)",
                min_value=5.0,
                max_value=1000.0,
                value=float(ln.max_capacity_mw),
                step=10.0,
                key=f"cfg_cap_{j}",
            )
            new_lines.append(DistributionLine(line_id=ln.line_id, name=ln.name, max_capacity_mw=c_val))

    spacer(18)

    # Apply Configuration Button
    if st.button("💾 Apply Configuration to Smart Grid", use_container_width=True):
        st.session_state.grid_config = GridConfig(areas=new_areas, lines=new_lines)
        st.session_state.opt_results = None
        st.success("Grid parameters applied successfully! Ready for PSO optimization.")
        st.rerun()

    spacer(20)

    # Configuration Summary & Feasibility Status
    curr_cfg = _get_config()
    feasible, feas_msg = check_feasibility(curr_cfg.total_demand, curr_cfg.total_capacity)
    margin = curr_cfg.total_capacity - curr_cfg.total_demand

    panel_header("📊 Configuration Summary & Feasibility Analysis")
    sm1, sm2, sm3, sm4 = st.columns(4)
    with sm1:
        st.markdown(render_kpi_card("Total Grid Demand", fmt_mw(curr_cfg.total_demand), "Aggregate area requirements", "⚡"), unsafe_allow_html=True)
    with sm2:
        st.markdown(render_kpi_card("Total Rated Capacity", fmt_mw(curr_cfg.total_capacity), "Aggregate 5-line capacity", "🔌"), unsafe_allow_html=True)
    with sm3:
        margin_color = C_OK if margin >= 0 else C_BAD
        st.markdown(render_kpi_card("Available Margin", fmt_mw(margin), "Reserve capacity buffer", "🛡️", margin_color), unsafe_allow_html=True)
    with sm4:
        feas_label = "Feasible Grid" if feasible else "Infeasible Grid"
        feas_color = C_OK if feasible else C_BAD
        feas_sub = "Meets demand criteria" if feasible else "Demand exceeds capacity"
        st.markdown(render_kpi_card("Feasibility Status", feas_label, feas_sub, "✅" if feasible else "⛔", feas_color), unsafe_allow_html=True)

    if not feasible:
        st.error(f"⛔ {feas_msg}")

    spacer(14)
    show_chart(plot_area_demand(curr_cfg))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PSO Optimization
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "PSO Optimization":
    config = _get_config()
    feasible, feas_msg = check_feasibility(config.total_demand, config.total_capacity)

    page_header("PSO Optimization Engine", "Configure swarm hyper-parameters, initiate metaheuristic load balancing, and export results.")

    # Swarm Hyper-parameter Control Panel
    panel_header("🎛️ Swarm Algorithm Hyper-parameters")

    p1, p2, p3 = st.columns(3)
    with p1:
        num_particles = st.number_input(
            "Swarm Population Size (Particles)",
            min_value=5, max_value=200, value=35, step=5,
            help="Number of candidate allocation matrices evaluated simultaneously in the swarm."
        )
    with p2:
        max_iterations = st.number_input(
            "Maximum Iterations (Generations)",
            min_value=10, max_value=2000, value=200, step=25,
            help="Maximum optimization generations before termination."
        )
    with p3:
        seed = st.number_input(
            "Random Number Seed",
            min_value=0, max_value=99999, value=42, step=1,
            help="Ensures deterministic and reproducible particle trajectory initialization."
        )

    p4, p5, p6 = st.columns(3)
    with p4:
        inertia = st.slider(
            "Inertia Weight (w)",
            min_value=0.1, max_value=1.5, value=0.72, step=0.02,
            help="Governs exploration vs exploitation. Higher values encourage wide parameter exploration."
        )
    with p5:
        c1 = st.slider(
            "Cognitive Coefficient (C₁)",
            min_value=0.1, max_value=4.0, value=1.5, step=0.1,
            help="Attraction toward individual particle's personal historical best position (Pbest)."
        )
    with p6:
        c2 = st.slider(
            "Social Coefficient (C₂)",
            min_value=0.1, max_value=4.0, value=1.5, step=0.1,
            help="Attraction toward the global best solution discovered by the entire swarm (Gbest)."
        )

    valid_params, param_errors = validate_pso_params(
        int(num_particles), int(max_iterations), float(inertia), float(c1), float(c2)
    )
    if not valid_params:
        for err in param_errors:
            st.error(err)

    if not feasible:
        st.error(f"⛔ Optimization cannot proceed: {feas_msg}")

    spacer(14)

    # Start Optimization Button
    opt_btn_disabled = (not valid_params) or (not feasible)
    if st.button("🚀 Start Particle Swarm Optimization", use_container_width=True, disabled=opt_btn_disabled):
        pso_cfg = PSOConfig(
            num_particles=int(num_particles),
            max_iterations=int(max_iterations),
            inertia_weight=float(inertia),
            cognitive_coeff=float(c1),
            social_coeff=float(c2),
            seed=int(seed),
        )

        progress_container = st.container()
        with progress_container:
            prog_bar = st.progress(0, text="Initializing swarm particles across 50 decision dimensions...")

        def on_step(it, total, best_fit):
            frac = it / total
            prog_bar.progress(frac, text=f"Optimizing Iteration {it}/{total}  |  Current Gbest Fitness: {best_fit:.5f}")

        results = run_optimization(
            config=config,
            pso_config=pso_cfg,
            progress_callback=on_step,
        )

        st.session_state.opt_results = results
        st.session_state.optimization_run = True
        prog_bar.progress(1.0, text="Optimization complete!")

        if results.is_feasible:
            st.success(f"✅ Feasible optimal load dispatch achieved in {results.execution_time:.2f} seconds!")
        else:
            st.warning("⚠️ Swarm converged but capacity constraint violations remain.")

    # Results Display
    results = st.session_state.opt_results
    if results is not None:
        spacer(20)
        panel_header("⚡ Post-Optimization Execution Metrics")

        res1, res2, res3, res4, res5 = st.columns(5)
        with res1:
            st.markdown(render_kpi_card("Best Fitness Score", f"{results.best_fitness:.5f}", "Multi-objective index", "🎯", C_OK), unsafe_allow_html=True)
        with res2:
            st.markdown(render_kpi_card("Iterations Completed", f"{results.iterations_completed}", f"Max: {max_iterations}", "🔄"), unsafe_allow_html=True)
        with res3:
            st.markdown(render_kpi_card("Execution Time", f"{results.execution_time:.2f}s", "CPU wall time", "⏱️"), unsafe_allow_html=True)
        with res4:
            ol_after = results.optimized_overloaded_count
            ol_col = C_OK if ol_after == 0 else C_BAD
            st.markdown(render_kpi_card("Overloads Post-PSO", f"{ol_after}", "Capacity violations", "⚠️", ol_col), unsafe_allow_html=True)
        with res5:
            conv_status = "Converged" if results.converged else "Max Iter"
            st.markdown(render_kpi_card("Final Status", conv_status, "Tolerance reached" if results.converged else "Exhausted generations", "✅", C_OK), unsafe_allow_html=True)

        spacer(20)

        # Convergence Plot Container
        fig_conv = plot_convergence(results.convergence_history)
        show_chart(fig_conv)

        spacer(14)

        # Allocation Matrix
        panel_header("📋 Optimized Power Dispatch Matrix (MW)")
        alloc_df = allocation_to_dataframe(
            results.optimized_allocation,
            config.area_names,
            config.line_names,
        )
        show_table(alloc_df, index=True, float_format=lambda x: f"{x:.2f}")

        # Export Tools
        panel_header("📥 Export Optimization Deliverables")
        exp1, exp2, exp3 = st.columns(3)
        with exp1:
            st.download_button(
                "📥 Download Allocation Matrix (CSV)",
                data=dataframe_to_csv_bytes(alloc_df),
                file_name="smart_grid_optimized_allocation.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp2:
            summary_df = results_summary_dataframe(results.to_summary_dict())
            st.download_button(
                "📥 Download Summary Metrics (CSV)",
                data=dataframe_to_csv_bytes(summary_df),
                file_name="smart_grid_pso_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp3:
            pdf_bytes = generate_pdf_report(results.to_summary_dict(), alloc_df, summary_df)
            if pdf_bytes:
                st.download_button(
                    "📥 Download Formal Engineering Report (PDF)",
                    data=pdf_bytes,
                    file_name="smart_grid_pso_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Before vs After
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Before vs After":
    results = st.session_state.opt_results

    page_header("Before vs After Optimization Analysis", "Rigorous comparison between initial baseline dispatch and PSO-optimized load balance.")

    if results is None:
        st.info("ℹ️ Optimization has not been executed yet. Navigate to 'PSO Optimization' and run the swarm algorithm to view before-vs-after analytics.")
    else:
        ini = results.initial_components
        opt = results.optimized_components

        # Helper for improvement percentage calculation
        def calc_improvement(before: float, after: float, lower_better: bool = True) -> tuple[float, str]:
            if before == 0:
                return 0.0, "0.0%"
            diff = before - after if lower_better else after - before
            pct = (diff / abs(before)) * 100
            sign = "+" if pct > 0 else ""
            return pct, f"{sign}{pct:.1f}%"

        # Six Metrics Side-by-Side Comparison
        panel_header("🎯 Core Objective Metric Improvements")

        m_data = [
            ("Objective Fitness Score", f"{ini['fitness']:.4f}", f"{opt['fitness']:.4f}", calc_improvement(ini['fitness'], opt['fitness'])[1]),
            ("Load Imbalance (Std Dev)", f"{ini['imbalance']:.4f}", f"{opt['imbalance']:.4f}", calc_improvement(ini['imbalance'], opt['imbalance'])[1]),
            ("Overloaded Distribution Lines", str(results.initial_overloaded_count), str(results.optimized_overloaded_count), f"{results.initial_overloaded_count - results.optimized_overloaded_count} eliminated"),
            ("Distribution Loss Index", f"{ini['loss']:.4f}", f"{opt['loss']:.4f}", calc_improvement(ini['loss'], opt['loss'])[1]),
            ("Demand Satisfaction Penalty", f"{ini['demand_penalty']:.4f}", f"{opt['demand_penalty']:.4f}", "Exact satisfaction"),
            ("Mean Grid Utilization", f"{ini['mean_utilisation']*100:.1f}%", f"{opt['mean_utilisation']*100:.1f}%", "Preserved"),
        ]

        comp_df = pd.DataFrame(m_data, columns=["Performance Metric", "Baseline Allocation", "PSO Optimized Allocation", "Improvement Delta"])
        show_table(comp_df, accent_last=True)

        spacer(20)

        # Comparative Visualization Grids
        col_comp_1, col_comp_2 = st.columns(2)
        with col_comp_1:
            fig_util_comp = plot_utilization_comparison(results.initial_line_stats, results.optimized_line_stats)
            show_chart(fig_util_comp)

        with col_comp_2:
            fig_loads_comp = plot_line_loads_comparison(results.initial_line_stats, results.optimized_line_stats)
            show_chart(fig_loads_comp)

        spacer(16)
        fig_radar = plot_before_after_radar(ini, opt)
        show_chart(fig_radar)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Analytics
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Analytics":
    config = _get_config()
    results = st.session_state.opt_results

    page_header("Grid Analytics Workspace", "Deep telemetry, spatial power allocation, loss modeling, and convergence trajectory.")

    if results is None:
        st.info("ℹ️ Running PSO Optimization unlocks the full suite of telemetry, heatmaps, and loss analyses. Displaying base demand profile below.")
        show_chart(plot_area_demand(config))
    else:
        tab_heat, tab_util, tab_dist, tab_loss, tab_conv, tab_dem = st.tabs([
            "🗺️ Load Allocation Heatmaps",
            "📊 Line Utilization & Headroom",
            "🌐 Power Dispatch Breakdown",
            "⚡ Distribution Loss Analysis",
            "📉 Convergence Trajectory",
            "🏙️ Area Demand Profile",
        ])

        with tab_heat:
            spacer(10)
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                panel_header("Baseline Load Heatmap")
                fig_h1 = plot_allocation_heatmap(results.initial_allocation, config.area_names, config.line_names, "Initial Load Dispatches (MW)")
                show_chart(fig_h1)
            with h_col2:
                panel_header("PSO Optimized Load Heatmap")
                fig_h2 = plot_allocation_heatmap(results.optimized_allocation, config.area_names, config.line_names, "Optimized Load Dispatches (MW)")
                show_chart(fig_h2)

        with tab_util:
            spacer(10)
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                show_chart(plot_line_utilization(results.optimized_line_stats))
            with col_u2:
                show_chart(plot_capacity_vs_load(results.optimized_line_stats))

        with tab_dist:
            spacer(10)
            show_chart(plot_distribution_breakdown(results.optimized_allocation, config.area_names, config.line_names))

        with tab_loss:
            spacer(10)
            fig_loss = plot_loss_analysis(results.initial_line_stats, results.optimized_line_stats)
            show_chart(fig_loss)

        with tab_conv:
            spacer(10)
            fig_c = plot_convergence(results.convergence_history)
            show_chart(fig_c)

        with tab_dem:
            spacer(10)
            show_chart(plot_area_demand(config))