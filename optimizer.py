"""
optimizer.py - Optimization orchestration and result processing.

Ties together grid_model, fitness, and pso modules to run the full
optimisation pipeline and produce structured results.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from grid_model import GridConfig, compute_line_stats, create_initial_allocation
from fitness import FitnessWeights, compute_fitness_components, evaluate_fitness
from pso import PSOConfig, PSOSolver, PSOResult, repair_allocation


# ═══════════════════════════════════════════════════════════════════════════
# Optimisation results
# ═══════════════════════════════════════════════════════════════════════════

class OptimizationResults:
    """Structured container for all optimisation results and metrics."""

    def __init__(self,
                 config: GridConfig,
                 initial_allocation: np.ndarray,
                 pso_result: PSOResult,
                 weights: FitnessWeights):
        self.config = config
        self.initial_allocation = initial_allocation
        self.optimized_allocation = pso_result.best_position
        self.pso_result = pso_result
        self.weights = weights

        # Pre-compute analytics
        demands = config.demands
        capacities = config.capacities

        self.initial_components = compute_fitness_components(
            initial_allocation, demands, capacities, weights
        )
        self.optimized_components = compute_fitness_components(
            self.optimized_allocation, demands, capacities, weights
        )

        self.initial_line_stats = compute_line_stats(initial_allocation, config)
        self.optimized_line_stats = compute_line_stats(self.optimized_allocation, config)

    # convenience properties -----------------------------------------------

    @property
    def total_demand(self) -> float:
        return self.config.total_demand

    @property
    def total_capacity(self) -> float:
        return self.config.total_capacity

    @property
    def best_fitness(self) -> float:
        return self.pso_result.best_fitness

    @property
    def iterations_completed(self) -> int:
        return self.pso_result.iterations_completed

    @property
    def converged(self) -> bool:
        return self.pso_result.converged

    @property
    def execution_time(self) -> float:
        return self.pso_result.execution_time

    @property
    def convergence_history(self) -> list[float]:
        return self.pso_result.convergence_history

    @property
    def initial_overloaded_count(self) -> int:
        return sum(1 for s in self.initial_line_stats if s["overloaded"])

    @property
    def optimized_overloaded_count(self) -> int:
        return sum(1 for s in self.optimized_line_stats if s["overloaded"])

    @property
    def is_feasible(self) -> bool:
        """True if optimized solution has no overloaded lines and demand is met."""
        demand_ok = self.optimized_components["demand_penalty"] < 0.01
        no_overload = self.optimized_overloaded_count == 0
        return demand_ok and no_overload

    def to_summary_dict(self) -> dict:
        """Flat dictionary suitable for summary tables and CSV export."""
        return {
            "total_demand": self.total_demand,
            "total_allocated": float(np.sum(self.optimized_allocation)),
            "best_fitness": self.best_fitness,
            "iterations_completed": self.iterations_completed,
            "execution_time": self.execution_time,
            "converged": self.converged,
            "feasible": self.is_feasible,
            "overloaded_before": self.initial_overloaded_count,
            "overloaded_after": self.optimized_overloaded_count,
            "imbalance_before": self.initial_components["imbalance"],
            "imbalance_after": self.optimized_components["imbalance"],
            "loss_before": self.initial_components["loss"],
            "loss_after": self.optimized_components["loss"],
            "fitness_before": self.initial_components["fitness"],
            "fitness_after": self.optimized_components["fitness"],
            "mean_util_before": self.initial_components["mean_utilisation"],
            "mean_util_after": self.optimized_components["mean_utilisation"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_optimization(
    config: GridConfig,
    pso_config: PSOConfig | None = None,
    fitness_weights: FitnessWeights | None = None,
    initial_allocation: np.ndarray | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> OptimizationResults:
    """
    Run the full PSO optimisation pipeline.

    Parameters
    ----------
    config : GridConfig
        Smart grid configuration.
    pso_config : PSOConfig, optional
        PSO hyper-parameters.
    fitness_weights : FitnessWeights, optional
        Fitness function weights.
    initial_allocation : ndarray, optional
        Custom initial allocation; built automatically if None.
    progress_callback : callable, optional
        Progress reporting during PSO iterations.

    Returns
    -------
    OptimizationResults
    """
    if pso_config is None:
        pso_config = PSOConfig()
    if fitness_weights is None:
        fitness_weights = FitnessWeights()

    demands = config.demands
    capacities = config.capacities

    # Build initial allocation if not provided
    if initial_allocation is None:
        initial_allocation = create_initial_allocation(config, seed=pso_config.seed)
    else:
        initial_allocation = initial_allocation.copy()

    # Fitness callable for the PSO solver
    def fitness_fn(allocation: np.ndarray) -> float:
        return evaluate_fitness(allocation, demands, capacities, fitness_weights)

    # Create and run solver
    solver = PSOSolver(
        num_areas=config.num_areas,
        num_lines=config.num_lines,
        demands=demands,
        capacities=capacities,
        fitness_fn=fitness_fn,
        config=pso_config,
    )

    pso_result = solver.solve(progress_callback=progress_callback)

    return OptimizationResults(
        config=config,
        initial_allocation=initial_allocation,
        pso_result=pso_result,
        weights=fitness_weights,
    )

