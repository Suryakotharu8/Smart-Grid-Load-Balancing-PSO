"""
grid_model.py - Smart grid configuration, area/line definitions, and allocation logic.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Area:
    """An electricity-consuming area in the smart grid."""
    name: str
    demand_mw: float  # electricity demand in megawatts


@dataclass
class DistributionLine:
    """A distribution line in the smart grid."""
    line_id: int
    name: str
    max_capacity_mw: float  # maximum capacity in megawatts


@dataclass
class GridConfig:
    """Complete grid configuration: areas + distribution lines."""
    areas: list[Area]
    lines: list[DistributionLine]

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def num_areas(self) -> int:
        return len(self.areas)

    @property
    def num_lines(self) -> int:
        return len(self.lines)

    @property
    def area_names(self) -> list[str]:
        return [a.name for a in self.areas]

    @property
    def line_names(self) -> list[str]:
        return [ln.name for ln in self.lines]

    @property
    def demands(self) -> np.ndarray:
        """1-D array of area demands (MW)."""
        return np.array([a.demand_mw for a in self.areas], dtype=np.float64)

    @property
    def capacities(self) -> np.ndarray:
        """1-D array of line capacities (MW)."""
        return np.array([ln.max_capacity_mw for ln in self.lines], dtype=np.float64)

    @property
    def total_demand(self) -> float:
        return float(np.sum(self.demands))

    @property
    def total_capacity(self) -> float:
        return float(np.sum(self.capacities))


# ═══════════════════════════════════════════════════════════════════════════
# Line-status helper
# ═══════════════════════════════════════════════════════════════════════════

def compute_line_stats(allocation: np.ndarray,
                       config: GridConfig) -> list[dict]:
    """
    Given an allocation matrix (num_areas × num_lines) and grid config,
    compute per-line load statistics.
    """
    line_loads = allocation.sum(axis=0)  # total load on each line
    stats = []
    for j, ln in enumerate(config.lines):
        load = float(line_loads[j])
        cap = ln.max_capacity_mw
        util = (load / cap * 100.0) if cap > 0 else 0.0
        remaining = max(0.0, cap - load)
        overloaded = load > cap
        stats.append({
            "line_id": ln.line_id,
            "name": ln.name,
            "capacity_mw": cap,
            "load_mw": load,
            "utilization_pct": util,
            "remaining_mw": remaining,
            "overloaded": overloaded,
        })
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Initial (naïve / unbalanced) allocation
# ═══════════════════════════════════════════════════════════════════════════

def create_initial_allocation(config: GridConfig,
                              seed: int = 42) -> np.ndarray:
    """
    Create an intentionally *unbalanced* initial allocation to demonstrate
    the need for PSO optimisation.

    Strategy: assign each area's demand primarily to one or two lines,
    chosen by cycling through lines with some randomness. This typically
    overloads some lines while leaving others under-utilised.
    """
    rng = np.random.RandomState(seed)
    n_areas = config.num_areas
    n_lines = config.num_lines
    demands = config.demands

    allocation = np.zeros((n_areas, n_lines), dtype=np.float64)

    for i in range(n_areas):
        # Pick a "primary" line — deterministic cycle biased to first lines
        primary = i % n_lines
        # Sometimes also use a secondary line
        if rng.random() < 0.3 and n_lines > 1:
            secondary = (primary + 1) % n_lines
            split = rng.uniform(0.6, 0.9)
            allocation[i, primary] = demands[i] * split
            allocation[i, secondary] = demands[i] * (1 - split)
        else:
            allocation[i, primary] = demands[i]

    return allocation


# ═══════════════════════════════════════════════════════════════════════════
# Predefined grid scenarios
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_AREA_NAMES = [
    "Residential Zone",
    "Industrial Zone",
    "Commercial Zone",
    "IT Park",
    "Hospital Zone",
    "Educational Zone",
    "Shopping District",
    "Manufacturing Zone",
    "Railway Station",
    "Public Infrastructure",
]


def _make_config(demands: list[float], capacities: list[float]) -> GridConfig:
    """Convenience builder."""
    areas = [Area(name=DEFAULT_AREA_NAMES[i], demand_mw=d) for i, d in enumerate(demands)]
    lines = [
        DistributionLine(line_id=j + 1, name=f"Line {j + 1}", max_capacity_mw=c)
        for j, c in enumerate(capacities)
    ]
    return GridConfig(areas=areas, lines=lines)


def get_normal_scenario() -> GridConfig:
    """Normal / moderate-demand scenario."""
    demands = [45, 80, 60, 55, 35, 30, 50, 75, 40, 25]
    capacities = [120, 130, 110, 100, 115]
    return _make_config(demands, capacities)


def get_peak_demand_scenario() -> GridConfig:
    """Peak-demand scenario — higher demands, tighter margins."""
    demands = [70, 120, 95, 85, 55, 50, 80, 110, 65, 45]
    capacities = [180, 190, 170, 160, 175]
    return _make_config(demands, capacities)


def get_unbalanced_scenario() -> GridConfig:
    """Unbalanced scenario — some areas have very high demand."""
    demands = [30, 150, 40, 25, 20, 15, 130, 100, 10, 5]
    capacities = [140, 120, 100, 110, 130]
    return _make_config(demands, capacities)


def get_default_config() -> GridConfig:
    """Return the default (normal) configuration."""
    return get_normal_scenario()

