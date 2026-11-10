"""
visualization.py - Theme-aware interactive Plotly charts for the Smart Grid dashboard.

Palette concept: "Substation" (dark) / "Daylight Schematic" (light)
  arc violet + filament amber + signal coral on ink / lavender paper.

Call set_theme("dark" | "light") before building figures (app.py does this on every run).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from grid_model import GridConfig


# ═══════════════════════════════════════════════════════════════════════════
# Theme definitions
# ═══════════════════════════════════════════════════════════════════════════

_THEMES = {
    "dark": {
        "colors": {
            "bg":            "#090B16",
            "card_bg":       "#131830",
            "card_elevated": "#1A2040",
            "border":        "#2A3361",
            "grid":          "rgba(143, 151, 196, 0.13)",
            "text":          "#ECEEFF",
            "text_muted":    "#8F97C4",
            "primary":       "#8B7BFF",   # arc violet
            "secondary":     "#B06CFF",
            "warning":       "#FFB020",   # filament amber
            "danger":        "#FF5C7A",   # signal coral
            "bar_initial":   "#FF5C7A",
            "bar_optimized": "#8B7BFF",
            "accent_blue":   "#4CC9F0",
            "fill_primary":  "rgba(139, 123, 255, 0.16)",
            "fill_danger":   "rgba(255, 92, 122, 0.14)",
            "heat_text":     "#FFFFFF",
            "on_bar":        "#FFFFFF",
            "hover_bg":      "#1A2040",
        },
        "area_palette": ["#8B7BFF", "#4CC9F0", "#F472B6", "#FFB020", "#4ADE9C",
                         "#B06CFF", "#60A5FA", "#FB923C", "#2DD4BF", "#E879F9"],
        "line_palette": ["#8B7BFF", "#4CC9F0", "#F472B6", "#FFB020", "#4ADE9C"],
        "heat_scale": [[0.0, "#131830"], [0.4, "#35288F"], [0.75, "#7B5CFF"], [1.0, "#C26BFF"]],
    },
    "light": {
        "colors": {
            "bg":            "#F3F1FB",
            "card_bg":       "#FFFFFF",
            "card_elevated": "#EFEBFC",
            "border":        "#DDD8F0",
            "grid":          "rgba(91, 61, 245, 0.09)",
            "text":          "#1A1640",
            "text_muted":    "#6B6791",
            "primary":       "#5B3DF5",   # arc indigo
            "secondary":     "#8A5CF6",
            "warning":       "#D97706",
            "danger":        "#E11D48",
            "bar_initial":   "#F43F5E",
            "bar_optimized": "#5B3DF5",
            "accent_blue":   "#0891B2",
            "fill_primary":  "rgba(91, 61, 245, 0.12)",
            "fill_danger":   "rgba(225, 29, 72, 0.10)",
            "heat_text":     "#1A1640",
            "on_bar":        "#FFFFFF",
            "hover_bg":      "#FFFFFF",
        },
        "area_palette": ["#5B3DF5", "#0891B2", "#DB2777", "#D97706", "#059669",
                         "#8A5CF6", "#2563EB", "#EA580C", "#0D9488", "#C026D3"],
        "line_palette": ["#5B3DF5", "#0891B2", "#DB2777", "#D97706", "#059669"],
        "heat_scale": [[0.0, "#F7F5FE"], [0.4, "#D6CEFC"], [0.75, "#9B84F8"], [1.0, "#6D4DF2"]],
    },
}

# Mutable module state (updated in place by set_theme so every function sees the active theme)
COLORS: dict = {}
AREA_PALETTE: list = []
LINE_PALETTE: list = []
HEAT_SCALE: list = []

FONT_FAMILY = "Manrope, system-ui, -apple-system, sans-serif"


def set_theme(name: str = "dark") -> None:
    """Switch the active chart theme ('dark' or 'light')."""
    t = _THEMES.get(name, _THEMES["dark"])
    COLORS.clear()
    COLORS.update(t["colors"])
    AREA_PALETTE[:] = t["area_palette"]
    LINE_PALETTE[:] = t["line_palette"]
    HEAT_SCALE[:] = t["heat_scale"]


set_theme("dark")


# ═══════════════════════════════════════════════════════════════════════════
# Shared layout
# ═══════════════════════════════════════════════════════════════════════════

def _axis_style() -> dict:
    return dict(
        gridcolor=COLORS["grid"],
        zerolinecolor=COLORS["border"],
        tickfont=dict(color=COLORS["text_muted"], size=11),
        title_font=dict(color=COLORS["text_muted"], size=12),
        linecolor=COLORS["border"],
    )


def _base_layout(title: str = "", height: int = 420) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family=FONT_FAMILY, size=12),
        margin=dict(l=55, r=30, t=60, b=45),
        title=dict(
            text=title,
            font=dict(color=COLORS["text"], size=15, family="Space Grotesk, " + FONT_FAMILY),
            x=0.015,
            y=0.95,
        ),
        xaxis=_axis_style(),
        yaxis=_axis_style(),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(color=COLORS["text"], size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(
            bgcolor=COLORS["hover_bg"],
            bordercolor=COLORS["border"],
            font=dict(color=COLORS["text"], family=FONT_FAMILY, size=12),
        ),
        height=height,
    )


def _apply_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(**_base_layout(title=title, height=height))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 1. Initial vs Optimized Line Loads
# ═══════════════════════════════════════════════════════════════════════════

def plot_line_loads_comparison(initial_stats: list[dict],
                               optimized_stats: list[dict]) -> go.Figure:
    names = [s["name"] for s in initial_stats]
    caps = [s["capacity_mw"] for s in initial_stats]
    init_loads = [s["load_mw"] for s in initial_stats]
    opt_loads = [s["load_mw"] for s in optimized_stats]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline Load",
        x=names,
        y=init_loads,
        marker=dict(color=COLORS["bar_initial"], line=dict(width=0)),
        text=[f"{v:.1f}" for v in init_loads],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>Baseline Load: %{y:.2f} MW<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Optimized Load",
        x=names,
        y=opt_loads,
        marker=dict(color=COLORS["bar_optimized"], line=dict(width=0)),
        text=[f"{v:.1f}" for v in opt_loads],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>Optimized Load: %{y:.2f} MW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        name="Rated Thermal Capacity",
        x=names,
        y=caps,
        mode="lines+markers",
        line=dict(color=COLORS["warning"], width=2.5, dash="dash"),
        marker=dict(size=9, symbol="diamond", color=COLORS["warning"],
                    line=dict(color=COLORS["card_bg"], width=1.5)),
        hovertemplate="<b>%{x}</b><br>Rated Capacity: %{y:.1f} MW<extra></extra>",
    ))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08, yaxis_title="Power Flow (MW)")
    return _apply_layout(fig, "Line Load Distribution vs Rated Thermal Capacity")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Capacity vs Allocated Load (Stacked Bar)
# ═══════════════════════════════════════════════════════════════════════════

def plot_capacity_vs_load(line_stats: list[dict]) -> go.Figure:
    names = [s["name"] for s in line_stats]
    loads = [s["load_mw"] for s in line_stats]
    remaining = [s["remaining_mw"] for s in line_stats]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Dispatched Power",
        x=names,
        y=loads,
        marker=dict(color=COLORS["primary"], line=dict(width=0)),
        text=[f"{v:.1f} MW" for v in loads],
        textposition="inside",
        textfont=dict(color=COLORS["on_bar"], size=11, family=FONT_FAMILY),
        hovertemplate="<b>%{x}</b><br>Dispatched: %{y:.2f} MW<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Available Headroom",
        x=names,
        y=remaining,
        marker=dict(color=COLORS["card_elevated"], line=dict(color=COLORS["border"], width=1.2)),
        text=[f"{v:.1f} MW" if v > 5 else "" for v in remaining],
        textposition="inside",
        textfont=dict(color=COLORS["text_muted"], size=11),
        hovertemplate="<b>%{x}</b><br>Headroom: %{y:.2f} MW<extra></extra>",
    ))
    fig.update_layout(barmode="stack", bargap=0.3, yaxis_title="Line Capacity (MW)")
    return _apply_layout(fig, "Capacity Utilization & Operational Headroom")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Area-wise Demand (Horizontal Bar)
# ═══════════════════════════════════════════════════════════════════════════

def plot_area_demand(config: GridConfig) -> go.Figure:
    names = config.area_names[::-1]
    demands = config.demands[::-1]

    colors = [AREA_PALETTE[i % len(AREA_PALETTE)] for i in range(len(names))]

    fig = go.Figure(go.Bar(
        x=demands,
        y=names,
        orientation="h",
        marker=dict(color=colors, opacity=0.95, line=dict(width=0)),
        text=[f"{d:.1f} MW" for d in demands],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text"]),
        hovertemplate="<b>%{y}</b><br>Demand: %{x:.2f} MW<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Required Power Demand (MW)",
        margin=dict(l=160, r=60, t=55, b=40),
        bargap=0.28,
    )
    return _apply_layout(fig, "Area Electricity Demand Profile", height=460)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Allocation Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_allocation_heatmap(allocation: np.ndarray,
                            area_names: list[str],
                            line_names: list[str],
                            title: str = "Load Allocation Heatmap") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=allocation,
        x=line_names,
        y=area_names,
        colorscale=[list(p) for p in HEAT_SCALE],
        xgap=3,
        ygap=3,
        text=np.round(allocation, 1),
        texttemplate="%{text}",
        textfont=dict(size=11, family=FONT_FAMILY, color=COLORS["heat_text"]),
        hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>Dispatched Load: %{z:.2f} MW<extra></extra>",
        colorbar=dict(
            title=dict(text="MW", font=dict(color=COLORS["text"], size=11)),
            tickfont=dict(color=COLORS["text_muted"], size=10),
            outlinewidth=0,
            thickness=12,
        ),
    ))
    fig.update_layout(margin=dict(l=160, r=40, t=60, b=40))
    return _apply_layout(fig, title, height=480)


# ═══════════════════════════════════════════════════════════════════════════
# 5. PSO Convergence Curve
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence(history: list[float]) -> go.Figure:
    iterations = list(range(len(history)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iterations,
        y=history,
        mode="lines",
        line=dict(color=COLORS["primary"], width=2.8, shape="spline", smoothing=0.4),
        fill="tozeroy",
        fillcolor=COLORS["fill_primary"],
        name="Global Best Fitness",
        hovertemplate="Iteration: %{x}<br>Objective Fitness: %{y:.6f}<extra></extra>",
    ))
    if len(history) > 0:
        fig.add_trace(go.Scatter(
            x=[iterations[-1]],
            y=[history[-1]],
            mode="markers",
            marker=dict(size=11, color=COLORS["warning"],
                        line=dict(color=COLORS["card_bg"], width=2)),
            name="Final Gbest",
            hovertemplate="Final Gbest: %{y:.6f}<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title="Iteration Step",
        yaxis_title="Objective Fitness Value (Minimization)",
    )
    return _apply_layout(fig, "PSO Swarm Fitness Convergence Trajectory")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Line Utilization Percentage
# ═══════════════════════════════════════════════════════════════════════════

def plot_line_utilization(line_stats: list[dict]) -> go.Figure:
    names = [s["name"] for s in line_stats]
    utils = [s["utilization_pct"] for s in line_stats]

    colors = []
    for u in utils:
        if u > 100:
            colors.append(COLORS["danger"])
        elif u > 85:
            colors.append(COLORS["warning"])
        else:
            colors.append(COLORS["primary"])

    fig = go.Figure(go.Bar(
        x=names,
        y=utils,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{u:.1f}%" for u in utils],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>Loading Ratio: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(
        y=100, line_dash="dash", line_color=COLORS["danger"],
        annotation_text="100% Thermal Rating Limit", annotation_position="top right",
        annotation_font=dict(color=COLORS["danger"], size=10)
    )
    fig.add_hline(
        y=85, line_dash="dot", line_color=COLORS["warning"],
        annotation_text="85% Dispatch Advisory", annotation_position="bottom right",
        annotation_font=dict(color=COLORS["warning"], size=10)
    )
    max_y = max(max(utils) * 1.15, 115)
    fig.update_layout(yaxis=dict(title="Line Loading (%)", range=[0, max_y]), bargap=0.35)
    return _apply_layout(fig, "Real-Time Distribution Line Utilization")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Electricity Distribution Breakdown (Sunburst)
# ═══════════════════════════════════════════════════════════════════════════

def plot_distribution_breakdown(allocation: np.ndarray,
                                area_names: list[str],
                                line_names: list[str]) -> go.Figure:
    labels, parents, values = ["Smart Grid"], [""], [0]

    for j, ln in enumerate(line_names):
        line_total = float(allocation[:, j].sum())
        labels.append(ln)
        parents.append("Smart Grid")
        values.append(line_total)

        for i, area in enumerate(area_names):
            val = float(allocation[i, j])
            if val > 0.05:
                labels.append(f"{area} ({ln})")
                parents.append(ln)
                values.append(val)

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=[COLORS["card_bg"]] + LINE_PALETTE + [AREA_PALETTE[i % len(AREA_PALETTE)]
                                                         for i in range(len(labels) - 1 - len(line_names))],
            line=dict(color=COLORS["card_bg"], width=1.5)
        ),
        hovertemplate="<b>%{label}</b><br>Dispatched: %{value:.2f} MW<extra></extra>",
        textinfo="label+value",
        textfont=dict(size=10, family=FONT_FAMILY, color="#FFFFFF"),
    ))
    return _apply_layout(fig, "Power Dispatch Distribution Breakdown", height=500)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Before / After Performance Radar Chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_before_after_radar(initial_comps: dict, optimized_comps: dict) -> go.Figure:
    categories = ["Load Balance", "Overload Avoidance", "Loss Minimization",
                   "Demand Satisfaction", "Grid Efficiency"]

    def _norm(val, worst):
        return max(0.0, 1.0 - min(val / worst, 1.0)) if worst > 0 else 1.0

    worst_imb = max(initial_comps["imbalance"], 0.5)
    worst_ol = max(initial_comps["overload"], 1.0)
    worst_loss = max(initial_comps["loss"], 0.1)
    worst_dem = max(initial_comps["demand_penalty"], 1.0)

    initial_vals = [
        _norm(initial_comps["imbalance"], worst_imb),
        _norm(initial_comps["overload"], worst_ol),
        _norm(initial_comps["loss"], worst_loss),
        _norm(initial_comps["demand_penalty"], worst_dem),
        min(1.0, initial_comps["mean_utilisation"]),
    ]
    optimized_vals = [
        _norm(optimized_comps["imbalance"], worst_imb),
        _norm(optimized_comps["overload"], worst_ol),
        _norm(optimized_comps["loss"], worst_loss),
        _norm(optimized_comps["demand_penalty"], worst_dem),
        min(1.0, optimized_comps["mean_utilisation"]),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=initial_vals + [initial_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Baseline State",
        line=dict(color=COLORS["danger"], width=2),
        fillcolor=COLORS["fill_danger"],
    ))
    fig.add_trace(go.Scatterpolar(
        r=optimized_vals + [optimized_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="PSO Optimized State",
        line=dict(color=COLORS["primary"], width=2.8),
        fillcolor=COLORS["fill_primary"],
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor=COLORS["grid"],
                linecolor=COLORS["border"],
                tickfont=dict(color=COLORS["text_muted"], size=9),
            ),
            angularaxis=dict(
                gridcolor=COLORS["grid"],
                linecolor=COLORS["border"],
                tickfont=dict(color=COLORS["text"], size=11, family=FONT_FAMILY),
            ),
        ),
    )
    return _apply_layout(fig, "Multi-Criteria Operational Performance Index", height=450)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Utilization Comparison (Grouped Bar)
# ═══════════════════════════════════════════════════════════════════════════

def plot_utilization_comparison(initial_stats: list[dict],
                                optimized_stats: list[dict]) -> go.Figure:
    names = [s["name"] for s in initial_stats]
    init_u = [s["utilization_pct"] for s in initial_stats]
    opt_u = [s["utilization_pct"] for s in optimized_stats]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline Loading",
        x=names,
        y=init_u,
        marker=dict(color=COLORS["bar_initial"], line=dict(width=0)),
        text=[f"{v:.1f}%" for v in init_u],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>Baseline Loading: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Optimized Loading",
        x=names,
        y=opt_u,
        marker=dict(color=COLORS["bar_optimized"], line=dict(width=0)),
        text=[f"{v:.1f}%" for v in opt_u],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>Optimized Loading: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(
        y=100, line_dash="dash", line_color=COLORS["warning"],
        annotation_text="Overload Threshold (100%)", annotation_position="top right",
        annotation_font=dict(color=COLORS["warning"], size=10)
    )
    max_val = max(max(init_u + opt_u) * 1.15, 115)
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08,
                      yaxis=dict(title="Line Loading (%)", range=[0, max_val]))
    return _apply_layout(fig, "Line Loading Ratio: Baseline vs Optimized")


# ═══════════════════════════════════════════════════════════════════════════
# 10. Distribution Loss Analysis Chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_loss_analysis(initial_stats: list[dict],
                       optimized_stats: list[dict],
                       k: float = 0.02) -> go.Figure:
    names = [s["name"] for s in initial_stats]
    init_loss = [k * ((s["load_mw"] / max(s["capacity_mw"], 1e-6)) ** 2) for s in initial_stats]
    opt_loss = [k * ((s["load_mw"] / max(s["capacity_mw"], 1e-6)) ** 2) for s in optimized_stats]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline Loss Index",
        x=names,
        y=init_loss,
        marker=dict(color=COLORS["bar_initial"], line=dict(width=0)),
        text=[f"{v:.4f}" for v in init_loss],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>Initial Loss Index: %{y:.5f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Optimized Loss Index",
        x=names,
        y=opt_loss,
        marker=dict(color=COLORS["primary"], line=dict(width=0)),
        text=[f"{v:.4f}" for v in opt_loss],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>Optimized Loss Index: %{y:.5f}<extra></extra>",
    ))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08,
                      yaxis_title="Quadratic Loss Index (Approx)")
    return _apply_layout(fig, "Estimated Technical Distribution Loss Index per Line")