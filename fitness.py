"""
fitness.py - Multi-objective fitness function for smart grid load balancing.

Mathematical Formulation
========================

The fitness function combines five components into a single minimisation
objective using a weighted sum:

    F(x) = w1·F_imbalance + w2·F_overload + w3·F_loss + w4·F_demand + w5·F_efficiency

where x is the allocation matrix (num_areas × num_lines).

Component definitions
---------------------

1. Load Imbalance (F_imbalance)
   Measures how unevenly the load is spread across distribution lines.
       utilisation_j = load_j / capacity_j   for each line j
       F_imbalance = std(utilisation_1, …, utilisation_L)

2. Overload Penalty (F_overload)
   Quadratic penalty for every line whose load exceeds its capacity.
       F_overload = Σ_j  max(0, load_j − capacity_j)²

3. Distribution Loss (F_loss)  [simplified quadratic model]
   A simplified approximation where losses grow quadratically with loading
   ratio.  This is NOT an AC power-flow calculation.
       F_loss = Σ_j  k · (load_j / capacity_j)²
   where k is a loss coefficient (default 0.02).

4. Demand Satisfaction Penalty (F_demand)
   Penalises any mismatch between allocated power and demanded power.
       F_demand = Σ_i  |allocated_i − demand_i|

5. Operational Efficiency (F_efficiency)
   Penalises under-utilisation of the grid.
       mean_util = mean(utilisation_1, …, utilisation_L)
       F_efficiency = max(0, target_util − mean_util)
   where target_util is a desired minimum average utilisation (default 0.5).

Penalty scaling
---------------
F_overload and F_demand use large penalty multipliers (default 1000) to
strongly discourage infeasible solutions.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class FitnessWeights:
    """Configurable weights for each fitness component."""
    w_imbalance: float = 10.0       # weight for load imbalance
    w_overload: float = 1000.0      # weight for overload penalty
    w_loss: float = 5.0             # weight for distribution loss
    w_demand: float = 1000.0        # weight for demand satisfaction
    w_efficiency: float = 2.0       # weight for operational efficiency
    loss_coefficient: float = 0.02  # k in simplified loss model
    target_utilisation: float = 0.5 # desired minimum average utilisation


def compute_fitness_components(allocation: np.ndarray,
                               demands: np.ndarray,
                               capacities: np.ndarray,
                               weights: FitnessWeights | None = None,
                               ) -> dict:
    """
    Evaluate all fitness components for a given allocation.

    Parameters
    ----------
    allocation : ndarray of shape (num_areas, num_lines)
        How much electricity each area draws from each line (MW).
    demands : ndarray of shape (num_areas,)
        Electricity demand per area (MW).
    capacities : ndarray of shape (num_lines,)
        Maximum capacity per distribution line (MW).
    weights : FitnessWeights, optional
        Component weights; defaults are used when *None*.

    Returns
    -------
    dict with keys: fitness, imbalance, overload, loss, demand_penalty,
         efficiency, line_loads, line_utils, allocated_demands
    """
    if weights is None:
        weights = FitnessWeights()

    num_areas, num_lines = allocation.shape

    # ── Per-line loads and utilisation ────────────────────────────────────
    line_loads = allocation.sum(axis=0)                        # (num_lines,)
    safe_caps = np.where(capacities > 0, capacities, 1.0)     # avoid /0
    line_utils = line_loads / safe_caps                        # fraction

    # ── Per-area allocated demand ────────────────────────────────────────
    allocated_demands = allocation.sum(axis=1)                 # (num_areas,)

    # ── 1. Load imbalance ────────────────────────────────────────────────
    imbalance = float(np.std(line_utils))

    # ── 2. Overload penalty ──────────────────────────────────────────────
    overloads = np.maximum(0.0, line_loads - capacities)
    overload = float(np.sum(overloads ** 2))

    # ── 3. Distribution loss (simplified quadratic model) ────────────────
    loss = float(np.sum(weights.loss_coefficient * (line_utils ** 2)))

    # ── 4. Demand satisfaction penalty ───────────────────────────────────
    demand_penalty = float(np.sum(np.abs(allocated_demands - demands)))

    # ── 5. Operational efficiency ────────────────────────────────────────
    mean_util = float(np.mean(line_utils))
    efficiency = max(0.0, weights.target_utilisation - mean_util)

    # ── Weighted total ───────────────────────────────────────────────────
    fitness = (
        weights.w_imbalance  * imbalance
        + weights.w_overload * overload
        + weights.w_loss     * loss
        + weights.w_demand   * demand_penalty
        + weights.w_efficiency * efficiency
    )

    return {
        "fitness": fitness,
        "imbalance": imbalance,
        "overload": overload,
        "loss": loss,
        "demand_penalty": demand_penalty,
        "efficiency": efficiency,
        "line_loads": line_loads,
        "line_utils": line_utils,
        "allocated_demands": allocated_demands,
        "mean_utilisation": mean_util,
    }


def evaluate_fitness(allocation: np.ndarray,
                     demands: np.ndarray,
                     capacities: np.ndarray,
                     weights: FitnessWeights | None = None) -> float:
    """Return only the scalar fitness value (convenience wrapper)."""
    return compute_fitness_components(allocation, demands, capacities, weights)["fitness"]

