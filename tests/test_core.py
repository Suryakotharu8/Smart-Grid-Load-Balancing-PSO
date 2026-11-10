"""
tests/test_core.py - Unit tests for the Smart Grid PSO project.

Run with:
    pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from grid_model import (
    GridConfig, Area, DistributionLine,
    compute_line_stats, create_initial_allocation,
    get_normal_scenario, get_peak_demand_scenario, get_unbalanced_scenario,
)
from fitness import FitnessWeights, compute_fitness_components, evaluate_fitness
from pso import PSOConfig, PSOSolver, repair_allocation, PSOResult
from optimizer import run_optimization


# ═══════════════════════════════════════════════════════════════════════════
# Grid Model Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGridModel:

    def test_normal_scenario_shapes(self):
        config = get_normal_scenario()
        assert config.num_areas == 10
        assert config.num_lines == 5
        assert len(config.demands) == 10
        assert len(config.capacities) == 5

    def test_total_demand_positive(self):
        config = get_normal_scenario()
        assert config.total_demand > 0

    def test_total_capacity_positive(self):
        config = get_normal_scenario()
        assert config.total_capacity > 0

    def test_initial_allocation_demand_conservation(self):
        """Each area's allocated load must equal its demand."""
        config = get_normal_scenario()
        alloc = create_initial_allocation(config)
        row_sums = alloc.sum(axis=1)
        np.testing.assert_allclose(row_sums, config.demands, atol=1e-10)

    def test_initial_allocation_non_negative(self):
        config = get_normal_scenario()
        alloc = create_initial_allocation(config)
        assert np.all(alloc >= 0)

    def test_line_stats_shape(self):
        config = get_normal_scenario()
        alloc = create_initial_allocation(config)
        stats = compute_line_stats(alloc, config)
        assert len(stats) == config.num_lines

    def test_line_stats_load_sum(self):
        """Total load across all lines must equal total demand."""
        config = get_normal_scenario()
        alloc = create_initial_allocation(config)
        stats = compute_line_stats(alloc, config)
        total_load = sum(s["load_mw"] for s in stats)
        assert abs(total_load - config.total_demand) < 1e-10

    def test_peak_scenario(self):
        config = get_peak_demand_scenario()
        assert config.total_demand > get_normal_scenario().total_demand

    def test_unbalanced_scenario(self):
        config = get_unbalanced_scenario()
        assert config.num_areas == 10


# ═══════════════════════════════════════════════════════════════════════════
# Fitness Function Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFitness:

    def _make_simple_config(self):
        areas = [Area("A", 50), Area("B", 50)]
        lines = [DistributionLine(1, "L1", 60), DistributionLine(2, "L2", 60)]
        return GridConfig(areas=areas, lines=lines)

    def test_perfect_allocation_low_fitness(self):
        """A balanced, feasible allocation should have low fitness."""
        config = self._make_simple_config()
        # Perfectly balanced: each area splits evenly across 2 lines
        alloc = np.array([[25, 25], [25, 25]], dtype=np.float64)
        comps = compute_fitness_components(alloc, config.demands, config.capacities)
        assert comps["demand_penalty"] < 1e-10
        assert comps["overload"] < 1e-10
        assert comps["imbalance"] < 1e-10

    def test_overload_penalty(self):
        """Overloaded allocation should have nonzero overload penalty."""
        config = self._make_simple_config()
        # All demand on line 1 → 100 MW on 60 MW line
        alloc = np.array([[50, 0], [50, 0]], dtype=np.float64)
        comps = compute_fitness_components(alloc, config.demands, config.capacities)
        assert comps["overload"] > 0

    def test_demand_penalty(self):
        """Mismatched demand should penalise."""
        config = self._make_simple_config()
        alloc = np.array([[30, 0], [50, 0]], dtype=np.float64)  # A gets only 30
        comps = compute_fitness_components(alloc, config.demands, config.capacities)
        assert comps["demand_penalty"] > 0

    def test_fitness_is_float(self):
        config = get_normal_scenario()
        alloc = create_initial_allocation(config)
        f = evaluate_fitness(alloc, config.demands, config.capacities)
        assert isinstance(f, float)
        assert np.isfinite(f)

    def test_zero_demand_no_error(self):
        """Zero demand area should not cause division by zero."""
        areas = [Area("A", 0), Area("B", 50)]
        lines = [DistributionLine(1, "L1", 60)]
        config = GridConfig(areas=areas, lines=lines)
        alloc = np.array([[0], [50]], dtype=np.float64)
        comps = compute_fitness_components(alloc, config.demands, config.capacities)
        assert np.isfinite(comps["fitness"])


# ═══════════════════════════════════════════════════════════════════════════
# Repair / Constraint Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRepair:

    def test_repair_demand_conservation(self):
        demands = np.array([30.0, 70.0])
        raw = np.array([[10, 5, 20], [-3, 80, 10]], dtype=np.float64)
        repaired = repair_allocation(raw, demands)
        np.testing.assert_allclose(repaired.sum(axis=1), demands, atol=1e-10)

    def test_repair_non_negative(self):
        demands = np.array([50.0])
        raw = np.array([[-10, -20, 100]], dtype=np.float64)
        repaired = repair_allocation(raw, demands)
        assert np.all(repaired >= 0)

    def test_repair_zero_demand(self):
        demands = np.array([0.0, 40.0])
        raw = np.array([[5, 5], [20, 20]], dtype=np.float64)
        repaired = repair_allocation(raw, demands)
        np.testing.assert_allclose(repaired[0], [0, 0])
        np.testing.assert_allclose(repaired[1].sum(), 40.0, atol=1e-10)

    def test_repair_all_zero_row(self):
        demands = np.array([30.0])
        raw = np.array([[0, 0, 0]], dtype=np.float64)
        repaired = repair_allocation(raw, demands)
        assert abs(repaired.sum() - 30.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# PSO Solver Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPSO:

    def test_pso_runs(self):
        config = get_normal_scenario()
        pso_cfg = PSOConfig(num_particles=10, max_iterations=20, seed=42)
        result = run_optimization(config, pso_config=pso_cfg)
        assert isinstance(result.pso_result, PSOResult)
        assert result.best_fitness < np.inf

    def test_pso_demand_conservation(self):
        config = get_normal_scenario()
        pso_cfg = PSOConfig(num_particles=10, max_iterations=50, seed=42)
        result = run_optimization(config, pso_config=pso_cfg)
        row_sums = result.optimized_allocation.sum(axis=1)
        np.testing.assert_allclose(row_sums, config.demands, atol=1e-8)

    def test_pso_non_negative(self):
        config = get_normal_scenario()
        pso_cfg = PSOConfig(num_particles=10, max_iterations=50, seed=42)
        result = run_optimization(config, pso_config=pso_cfg)
        assert np.all(result.optimized_allocation >= -1e-10)

    def test_pso_improves_fitness(self):
        """PSO result should be at least as good as initial allocation."""
        config = get_normal_scenario()
        pso_cfg = PSOConfig(num_particles=20, max_iterations=100, seed=42)
        result = run_optimization(config, pso_config=pso_cfg)
        assert result.optimized_components["fitness"] <= result.initial_components["fitness"] + 1e-6

    def test_pso_reproducible(self):
        """Same seed should produce same results."""
        config = get_normal_scenario()
        pso_cfg = PSOConfig(num_particles=10, max_iterations=30, seed=123)
        r1 = run_optimization(config, pso_config=pso_cfg)
        r2 = run_optimization(config, pso_config=pso_cfg)
        np.testing.assert_allclose(r1.optimized_allocation, r2.optimized_allocation, atol=1e-12)

    def test_pso_convergence_history(self):
        config = get_normal_scenario()
        pso_cfg = PSOConfig(num_particles=10, max_iterations=30, seed=42)
        result = run_optimization(config, pso_config=pso_cfg)
        history = result.convergence_history
        assert len(history) > 1
        # History should be non-increasing (gbest can only improve or stay)
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-12

    def test_pso_feasible_normal_scenario(self):
        """Normal scenario should yield a feasible solution with enough iterations."""
        config = get_normal_scenario()
        pso_cfg = PSOConfig(num_particles=30, max_iterations=200, seed=42)
        result = run_optimization(config, pso_config=pso_cfg)
        assert result.is_feasible

    def test_capacity_respected(self):
        """In a feasible normal scenario, no line should be overloaded."""
        config = get_normal_scenario()
        pso_cfg = PSOConfig(num_particles=30, max_iterations=200, seed=42)
        result = run_optimization(config, pso_config=pso_cfg)
        line_loads = result.optimized_allocation.sum(axis=0)
        for j, ln in enumerate(config.lines):
            assert line_loads[j] <= ln.max_capacity_mw + 1e-6, \
                f"Line {ln.name} overloaded: {line_loads[j]:.2f} > {ln.max_capacity_mw}"


# ═══════════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_single_area_single_line(self):
        areas = [Area("A", 50)]
        lines = [DistributionLine(1, "L1", 100)]
        config = GridConfig(areas=areas, lines=lines)
        pso_cfg = PSOConfig(num_particles=5, max_iterations=20, seed=42)
        result = run_optimization(config, pso_config=pso_cfg)
        np.testing.assert_allclose(result.optimized_allocation, [[50]], atol=1e-6)

    def test_zero_demand_area(self):
        areas = [Area("A", 0), Area("B", 40)]
        lines = [DistributionLine(1, "L1", 50), DistributionLine(2, "L2", 50)]
        config = GridConfig(areas=areas, lines=lines)
        pso_cfg = PSOConfig(num_particles=10, max_iterations=30, seed=42)
        result = run_optimization(config, pso_config=pso_cfg)
        np.testing.assert_allclose(result.optimized_allocation[0].sum(), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.optimized_allocation[1].sum(), 40.0, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

