"""
benchmark.py - Automated Sensitivity Analysis & Benchmarking for Smart Grid PSO.

This script executes empirical parameter sensitivity tests across:
  - Swarm sizes: N in {10, 20, 35, 50}
  - Inertia weights: w in {0.4, 0.72, 0.9}
  - Predefined scenarios: Normal, Peak Demand, Unbalanced Grid

Outputs a summary table to stdout and exports results to results/benchmark_results.csv.

Usage:
  python benchmark.py
"""

from __future__ import annotations

import time
import os
import pandas as pd
import numpy as np

from grid_model import get_normal_scenario, get_peak_demand_scenario, get_unbalanced_scenario
from optimizer import run_optimization
from pso import PSOConfig


def run_benchmarks():
    print("=" * 80)
    print("  SMART GRID PSO — EMPIRICAL BENCHMARK & SENSITIVITY EXPERIMENT")
    print("=" * 80)

    scenarios = [
        ("Normal", get_normal_scenario()),
        ("Peak Demand", get_peak_demand_scenario()),
        ("Unbalanced", get_unbalanced_scenario()),
    ]

    swarm_sizes = [15, 30, 50]
    inertias = [0.4, 0.72, 0.9]
    max_iter = 150
    seed = 42

    records = []
    total_runs = len(scenarios) * len(swarm_sizes) * len(inertias)
    curr_run = 0

    t_start_all = time.time()

    for sc_name, config in scenarios:
        for n_particles in swarm_sizes:
            for w in inertias:
                curr_run += 1
                pso_cfg = PSOConfig(
                    num_particles=n_particles,
                    max_iterations=max_iter,
                    inertia_weight=w,
                    cognitive_coeff=1.5,
                    social_coeff=1.5,
                    seed=seed,
                )

                t0 = time.time()
                res = run_optimization(config, pso_config=pso_cfg)
                elapsed = time.time() - t0

                overloads_before = res.initial_overloaded_count
                overloads_after = res.optimized_overloaded_count
                imbalance_before = res.initial_components["imbalance"]
                imbalance_after = res.optimized_components["imbalance"]
                loss_before = res.initial_components["loss"]
                loss_after = res.optimized_components["loss"]

                records.append({
                    "Scenario": sc_name,
                    "Particles": n_particles,
                    "Inertia (w)": w,
                    "Best Fitness": round(res.best_fitness, 5),
                    "Iterations": res.iterations_completed,
                    "Time (s)": round(elapsed, 2),
                    "Converged": "Yes" if res.converged else "No",
                    "Feasible": "Yes" if res.is_feasible else "No",
                    "Overloads (Init -> Opt)": f"{overloads_before} -> {overloads_after}",
                    "Imbalance (Init -> Opt)": f"{imbalance_before:.3f} -> {imbalance_after:.3f}",
                    "Loss Index": round(loss_after, 4),
                })

                print(f"[{curr_run:02d}/{total_runs}] Scenario: {sc_name:12s} | N={n_particles:2d} | w={w:.2f} | "
                      f"Fitness={res.best_fitness:9.4f} | Time={elapsed:.2f}s | Feasible={'✅' if res.is_feasible else '⚠️'}")

    df_results = pd.DataFrame(records)

    # Ensure output directory exists
    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", "benchmark_results.csv")
    df_results.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)
    print(f"Total experiment time: {time.time() - t_start_all:.2f}s")
    print(f"Results saved to: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmarks()

