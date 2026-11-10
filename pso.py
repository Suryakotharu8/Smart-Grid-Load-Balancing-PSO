"""
pso.py - Particle Swarm Optimization implemented from scratch.

This module provides a complete, configurable PSO solver for the smart-grid
load-balancing problem.  The implementation follows the canonical PSO
equations with inertia weight:

    v_i(t+1) = w · v_i(t)
              + c1 · r1 · (pbest_i − x_i(t))
              + c2 · r2 · (gbest   − x_i(t))

    x_i(t+1) = x_i(t) + v_i(t+1)

After each position update the particle is *repaired* so that:
  • Every element is non-negative.
  • Each row (area) sums exactly to that area's electricity demand.

This guarantees demand-satisfaction feasibility at every iteration.
Capacity constraints are handled softly via the penalty-based fitness
function defined in fitness.py.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
import time


# ═══════════════════════════════════════════════════════════════════════════
# PSO configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PSOConfig:
    """Hyper-parameters for the PSO algorithm."""
    num_particles: int = 30
    max_iterations: int = 200
    inertia_weight: float = 0.7
    cognitive_coeff: float = 1.5   # c1
    social_coeff: float = 1.5      # c2
    seed: int = 42
    convergence_tol: float = 1e-8  # stop if gbest doesn't improve by this
    convergence_patience: int = 30  # consecutive stale iterations to declare convergence
    velocity_clamp_fraction: float = 0.2  # max |v| as fraction of demand spread


# ═══════════════════════════════════════════════════════════════════════════
# Particle
# ═══════════════════════════════════════════════════════════════════════════

class Particle:
    """A single particle in the swarm."""

    def __init__(self,
                 position: np.ndarray,
                 velocity: np.ndarray):
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.pbest_position = position.copy()
        self.pbest_fitness: float = np.inf

    def update_pbest(self, fitness: float) -> None:
        if fitness < self.pbest_fitness:
            self.pbest_fitness = fitness
            self.pbest_position = self.position.copy()


# ═══════════════════════════════════════════════════════════════════════════
# Repair / constraint handling
# ═══════════════════════════════════════════════════════════════════════════

def repair_allocation(position: np.ndarray,
                      demands: np.ndarray) -> np.ndarray:
    """
    Repair a position matrix so that:
      1. All entries are ≥ 0.
      2. Each row sums to the corresponding area demand.

    Uses clipping + proportional re-scaling.
    """
    pos = np.maximum(position, 0.0)          # clip negatives
    num_areas, num_lines = pos.shape

    for i in range(num_areas):
        row_sum = pos[i].sum()
        demand_i = demands[i]

        if demand_i == 0.0:
            pos[i] = 0.0
            continue

        if row_sum > 0:
            pos[i] = pos[i] * (demand_i / row_sum)  # scale to exact demand
        else:
            # If entire row is zero, distribute demand equally
            pos[i] = demand_i / num_lines

    return pos


# ═══════════════════════════════════════════════════════════════════════════
# PSO solver
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PSOResult:
    """Container for PSO results."""
    best_position: np.ndarray         # best allocation matrix found
    best_fitness: float               # best fitness value
    convergence_history: list[float]  # gbest fitness per iteration
    iterations_completed: int
    converged: bool                   # True if early-stopped by convergence
    execution_time: float             # seconds


class PSOSolver:
    """
    Particle Swarm Optimization solver for smart-grid load balancing.

    Parameters
    ----------
    num_areas, num_lines : int
        Dimensions of the allocation matrix.
    demands : ndarray (num_areas,)
        Area electricity demands.
    capacities : ndarray (num_lines,)
        Line maximum capacities.
    fitness_fn : callable
        Function(allocation) → float  (lower is better).
    config : PSOConfig
        Hyper-parameters.
    """

    def __init__(self,
                 num_areas: int,
                 num_lines: int,
                 demands: np.ndarray,
                 capacities: np.ndarray,
                 fitness_fn: Callable[[np.ndarray], float],
                 config: PSOConfig | None = None):

        self.num_areas = num_areas
        self.num_lines = num_lines
        self.demands = demands.copy()
        self.capacities = capacities.copy()
        self.fitness_fn = fitness_fn
        self.config = config or PSOConfig()
        self.rng = np.random.RandomState(self.config.seed)

        # Velocity clamping bound
        demand_spread = max(float(demands.max() - demands.min()), 1.0)
        self.v_max = self.config.velocity_clamp_fraction * demand_spread

        # Swarm state
        self.swarm: list[Particle] = []
        self.gbest_position: np.ndarray | None = None
        self.gbest_fitness: float = np.inf

    # ──────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────

    def _initialise_swarm(self) -> None:
        """Create the initial population of particles."""
        self.swarm = []
        for _ in range(self.config.num_particles):
            # Random allocation: for each area, draw random weights across
            # lines and scale to match demand.
            raw = self.rng.random((self.num_areas, self.num_lines))
            position = repair_allocation(raw, self.demands)

            # Small random initial velocities
            velocity = self.rng.uniform(
                -self.v_max * 0.1, self.v_max * 0.1,
                size=(self.num_areas, self.num_lines),
            )

            p = Particle(position, velocity)
            fitness = self.fitness_fn(p.position)
            p.update_pbest(fitness)

            if fitness < self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = p.position.copy()

            self.swarm.append(p)

    # ──────────────────────────────────────────────────────────────────────
    # Core PSO loop
    # ──────────────────────────────────────────────────────────────────────

    def solve(self, progress_callback: Callable[[int, int, float], None] | None = None
              ) -> PSOResult:
        """
        Run the PSO optimisation.

        Parameters
        ----------
        progress_callback : callable, optional
            Called as callback(iteration, max_iterations, best_fitness).

        Returns
        -------
        PSOResult
        """
        t_start = time.time()
        cfg = self.config

        # Initialise swarm
        self._initialise_swarm()

        convergence_history: list[float] = [self.gbest_fitness]
        stale_count = 0
        converged = False
        iterations_done = 0

        for t in range(cfg.max_iterations):
            iterations_done = t + 1

            for particle in self.swarm:
                r1 = self.rng.random(particle.position.shape)
                r2 = self.rng.random(particle.position.shape)

                # ── Velocity update ──────────────────────────────────────
                inertia_term = cfg.inertia_weight * particle.velocity
                cognitive_term = cfg.cognitive_coeff * r1 * (
                    particle.pbest_position - particle.position
                )
                social_term = cfg.social_coeff * r2 * (
                    self.gbest_position - particle.position
                )
                new_velocity = inertia_term + cognitive_term + social_term

                # Velocity clamping
                new_velocity = np.clip(new_velocity, -self.v_max, self.v_max)
                particle.velocity = new_velocity

                # ── Position update ──────────────────────────────────────
                new_position = particle.position + new_velocity

                # Repair to satisfy constraints
                new_position = repair_allocation(new_position, self.demands)
                particle.position = new_position

                # ── Evaluate fitness ─────────────────────────────────────
                fitness = self.fitness_fn(particle.position)
                particle.update_pbest(fitness)

                # ── Update global best ───────────────────────────────────
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = particle.position.copy()

            convergence_history.append(self.gbest_fitness)

            # ── Convergence check ────────────────────────────────────────
            if len(convergence_history) >= 2:
                improvement = abs(convergence_history[-2] - convergence_history[-1])
                if improvement < cfg.convergence_tol:
                    stale_count += 1
                else:
                    stale_count = 0

                if stale_count >= cfg.convergence_patience:
                    converged = True
                    if progress_callback:
                        progress_callback(iterations_done, cfg.max_iterations,
                                          self.gbest_fitness)
                    break

            if progress_callback:
                progress_callback(iterations_done, cfg.max_iterations,
                                  self.gbest_fitness)

        t_end = time.time()

        return PSOResult(
            best_position=self.gbest_position.copy(),
            best_fitness=self.gbest_fitness,
            convergence_history=convergence_history,
            iterations_completed=iterations_done,
            converged=converged,
            execution_time=t_end - t_start,
        )

