"""
utils.py - Validation, formatting and helper functions for Smart Grid PSO.
"""

import numpy as np
import pandas as pd
import io
import csv
from datetime import datetime


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_demand(demand: float, area_name: str = "Area") -> tuple[bool, str]:
    """Validate that a demand value is a non-negative finite number."""
    if not isinstance(demand, (int, float)):
        return False, f"{area_name}: Demand must be a number."
    if np.isnan(demand) or np.isinf(demand):
        return False, f"{area_name}: Demand must be finite."
    if demand < 0:
        return False, f"{area_name}: Demand cannot be negative."
    return True, ""


def validate_capacity(capacity: float, line_name: str = "Line") -> tuple[bool, str]:
    """Validate that a capacity value is a positive finite number."""
    if not isinstance(capacity, (int, float)):
        return False, f"{line_name}: Capacity must be a number."
    if np.isnan(capacity) or np.isinf(capacity):
        return False, f"{line_name}: Capacity must be finite."
    if capacity <= 0:
        return False, f"{line_name}: Capacity must be positive."
    return True, ""


def validate_pso_params(num_particles: int, max_iterations: int,
                        inertia: float, c1: float, c2: float) -> tuple[bool, list[str]]:
    """Validate PSO hyper-parameters, returning (valid, list_of_errors)."""
    errors: list[str] = []
    if not isinstance(num_particles, int) or num_particles < 2:
        errors.append("Number of particles must be an integer ≥ 2.")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        errors.append("Maximum iterations must be an integer ≥ 1.")
    if not isinstance(inertia, (int, float)) or inertia < 0 or inertia > 2:
        errors.append("Inertia weight must be between 0 and 2.")
    if not isinstance(c1, (int, float)) or c1 < 0 or c1 > 4:
        errors.append("Cognitive coefficient (C1) must be between 0 and 4.")
    if not isinstance(c2, (int, float)) or c2 < 0 or c2 > 4:
        errors.append("Social coefficient (C2) must be between 0 and 4.")
    return (len(errors) == 0), errors


def check_feasibility(total_demand: float, total_capacity: float) -> tuple[bool, str]:
    """Check whether the grid has enough total capacity for total demand."""
    if total_capacity <= 0:
        return False, "Total grid capacity is zero or negative."
    if total_demand > total_capacity:
        shortfall = total_demand - total_capacity
        return False, (
            f"Total demand ({total_demand:.2f} MW) exceeds total capacity "
            f"({total_capacity:.2f} MW) by {shortfall:.2f} MW. "
            "No feasible allocation exists."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_mw(value: float) -> str:
    """Format a value as MW with 2 decimal places."""
    return f"{value:.2f} MW"


def fmt_pct(value: float) -> str:
    """Format a value as a percentage with 1 decimal place."""
    return f"{value:.1f}%"


def status_emoji(utilization_pct: float) -> str:
    """Return a status emoji based on line utilization percentage."""
    if utilization_pct > 100:
        return "🔴"
    elif utilization_pct > 85:
        return "🟡"
    else:
        return "🟢"


def status_text(utilization_pct: float) -> str:
    """Return a status label based on line utilization percentage."""
    if utilization_pct > 100:
        return "Overloaded"
    elif utilization_pct > 85:
        return "High Load"
    elif utilization_pct > 50:
        return "Normal"
    else:
        return "Low Load"


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def allocation_to_dataframe(allocation: np.ndarray,
                            area_names: list[str],
                            line_names: list[str]) -> pd.DataFrame:
    """Convert an allocation matrix to a labelled DataFrame."""
    df = pd.DataFrame(allocation, index=area_names, columns=line_names)
    df["Total Allocated (MW)"] = df.sum(axis=1)
    return df


def results_summary_dataframe(results: dict) -> pd.DataFrame:
    """Build a summary DataFrame from optimization results."""
    rows = [
        ("Total Demand (MW)", f"{results.get('total_demand', 0):.2f}"),
        ("Total Allocated (MW)", f"{results.get('total_allocated', 0):.2f}"),
        ("Best Fitness", f"{results.get('best_fitness', 0):.6f}"),
        ("Iterations", str(results.get('iterations_completed', 0))),
        ("Execution Time (s)", f"{results.get('execution_time', 0):.2f}"),
        ("Converged", str(results.get('converged', False))),
        ("Feasible", str(results.get('feasible', False))),
        ("Overloaded Lines (Before)", str(results.get('overloaded_before', 0))),
        ("Overloaded Lines (After)", str(results.get('overloaded_after', 0))),
        ("Load Imbalance (Before)", f"{results.get('imbalance_before', 0):.4f}"),
        ("Load Imbalance (After)", f"{results.get('imbalance_after', 0):.4f}"),
        ("Distribution Loss (Before)", f"{results.get('loss_before', 0):.4f}"),
        ("Distribution Loss (After)", f"{results.get('loss_after', 0):.4f}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes for download."""
    return df.to_csv(index=True).encode("utf-8")


# ---------------------------------------------------------------------------
# PDF report (best-effort using fpdf2)
# ---------------------------------------------------------------------------

def generate_pdf_report(results: dict,
                        allocation_df: pd.DataFrame,
                        summary_df: pd.DataFrame) -> bytes:
    """Generate a simple PDF report of the optimization results."""
    try:
        from fpdf import FPDF
    except ImportError:
        return b""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Smart Grid Load Balancing - PSO Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # Summary table
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Optimization Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    for _, row in summary_df.iterrows():
        pdf.cell(90, 7, str(row["Metric"]), border=1)
        pdf.cell(60, 7, str(row["Value"]), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Allocation table
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Optimized Allocation Matrix (MW)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 7)

    cols = ["Area"] + list(allocation_df.columns)
    col_w = min(28, (pdf.w - 20) / len(cols))

    # Header
    pdf.set_font("Helvetica", "B", 7)
    for c in cols:
        label = str(c)[:16]
        pdf.cell(col_w, 6, label, border=1)
    pdf.ln()

    # Rows
    pdf.set_font("Helvetica", "", 7)
    for idx, row_data in allocation_df.iterrows():
        label = str(idx)[:16]
        pdf.cell(col_w, 6, label, border=1)
        for val in row_data:
            pdf.cell(col_w, 6, f"{val:.2f}", border=1)
        pdf.ln()

    pdf.ln(6)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, (
        "Disclaimer: This is a simplified simulation-based smart-grid load "
        "allocation project using Particle Swarm Optimization. It is not a "
        "certified electrical power-flow or grid-control system."
    ))

    return bytes(pdf.output())

