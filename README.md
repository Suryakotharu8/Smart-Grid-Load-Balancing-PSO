# ⚡ Smart Grid Load Balancing Using Particle Swarm Optimization (PSO)

> An Evolutionary Computing academic project that optimises electricity load distribution across multiple distribution lines using a from-scratch PSO implementation, presented through an interactive Streamlit dashboard.

---

## 📋 Table of Contents

- [Abstract](#abstract)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Existing System vs Proposed System](#existing-system-vs-proposed-system)
- [Evolutionary Computing Concepts](#evolutionary-computing-concepts)
- [PSO Mathematical Equations](#pso-mathematical-equations)
- [System Architecture](#system-architecture)
- [Algorithm Workflow](#algorithm-workflow)
- [Fitness Function](#fitness-function)
- [Installation & Execution](#installation--execution)
- [Sample Results](#sample-results)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Disclaimer](#disclaimer)

---

## Abstract

This project presents a simulation-based smart grid system that optimises electricity load distribution across **5 distribution lines** serving **10 consuming areas** using **Particle Swarm Optimization (PSO)**. The system models a constrained allocation problem where area demands must be fully satisfied without exceeding line capacities, while minimising load imbalance and distribution losses. A multi-objective fitness function penalises overloads, demand mismatches, and inefficiency. The entire PSO algorithm is implemented from scratch in Python/NumPy, and results are presented through a polished, interactive Streamlit dashboard with before/after analysis, convergence visualisation, and exportable reports.

---

## Problem Statement

In modern power distribution networks, electricity must be efficiently allocated from multiple distribution lines to numerous consuming areas. Poor allocation leads to:

- **Overloaded lines** — risking equipment failure and power outages.
- **Under-utilised lines** — wasting infrastructure investment.
- **Higher distribution losses** — losses increase non-linearly with loading.
- **Unmet demand** — some areas may not receive their required electricity.

Traditional fixed or round-robin allocation strategies cannot adapt to varying demand patterns. An intelligent, population-based optimisation approach can explore the large solution space effectively and find near-optimal allocations.

---

## Objectives

1. Model a configurable smart grid with 10 areas and 5 distribution lines.
2. Define a multi-objective fitness function covering balance, overloads, losses, demand satisfaction, and efficiency.
3. Implement PSO from scratch (no library imports for the optimizer).
4. Demonstrate measurable improvement over a naïve/unbalanced initial allocation.
5. Provide an interactive dashboard for configuration, optimisation, analysis, and export.
6. Clearly explain all Evolutionary Computing concepts used.

---

## Existing System vs Proposed System

| Aspect | Existing System | Proposed System |
|--------|----------------|-----------------|
| Allocation | Fixed / manual | PSO-optimised |
| Adaptability | Static | Adapts to any demand scenario |
| Overload handling | Manual monitoring | Automatic penalty-based avoidance |
| Loss awareness | None | Simplified quadratic loss model |
| Balance | Often uneven | Optimised for even utilisation |
| Visualisation | Spreadsheets | Interactive Plotly dashboard |

---

## Evolutionary Computing Concepts

### Swarm Intelligence
PSO belongs to the **Swarm Intelligence** family — algorithms inspired by the collective behaviour of decentralised, self-organised agents (bird flocks, fish schools). Individual particles follow simple rules, but the swarm collectively converges toward good solutions.

### Particle Swarm Optimization (PSO)
Introduced by Kennedy & Eberhart (1995), PSO simulates a population of particles moving through a search space. Each particle remembers its own best position (**Pbest**) and is attracted to the swarm's best position (**Gbest**).

Key components:
- **Particle** — a candidate solution (allocation matrix).
- **Velocity** — direction and magnitude of search movement.
- **Pbest** — the best position a particle has found so far.
- **Gbest** — the best position found by any particle.
- **Inertia weight (w)** — controls momentum (exploration vs exploitation).
- **Cognitive coefficient (c₁)** — attraction to Pbest.
- **Social coefficient (c₂)** — attraction to Gbest.

---

## PSO Mathematical Equations

### Velocity Update

```
v_i(t+1) = w · v_i(t) + c₁ · r₁ · (Pbest_i − x_i(t)) + c₂ · r₂ · (Gbest − x_i(t))
```

### Position Update

```
x_i(t+1) = x_i(t) + v_i(t+1)
```

### Constraint Repair

After each position update:
1. Clip negative values to zero.
2. Normalise each row so it sums exactly to the area's demand.

This ensures **demand conservation** at every iteration.

### Convergence Detection

Optimisation terminates when:
- Maximum iterations are reached, **or**
- The global best fitness has not improved by more than ε = 10⁻⁸ for 30 consecutive iterations.

---

## System Architecture

```
smart_grid_pso/
├── app.py              # Streamlit dashboard and navigation
├── pso.py              # Core PSO implementation (from scratch)
├── fitness.py          # Fitness function and constraint penalties
├── grid_model.py       # Grid configuration and allocation logic
├── optimizer.py        # Optimization orchestration
├── visualization.py    # Plotly interactive charts
├── utils.py            # Validation, formatting, export helpers
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── results/            # Output directory
└── tests/
    ├── __init__.py
    └── test_core.py    # Unit tests
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `app.py` | Streamlit UI: 7 navigation pages, styling, session state |
| `pso.py` | Particle, velocity/position update, repair, convergence |
| `fitness.py` | 5-component weighted fitness function |
| `grid_model.py` | Area/Line dataclasses, scenarios, initial allocation |
| `optimizer.py` | Pipeline orchestration, result aggregation |
| `visualization.py` | 9 Plotly chart functions |
| `utils.py` | Validation, CSV/PDF export, formatting |

---

## Algorithm Workflow

```
┌─────────────────────────────────────┐
│   1. Configure Grid                 │
│      (areas, demands, capacities)   │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│   2. Create Initial Allocation      │
│      (intentionally unbalanced)     │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│   3. Initialise PSO Swarm          │
│      (random feasible particles)   │
└──────────────┬──────────────────────┘
               ▼
┌──────────────────────────────────────────────────┐
│   4. For each iteration t = 1 … T:               │
│      a. Update velocities (w, c₁, c₂, r₁, r₂)   │
│      b. Update positions                          │
│      c. Repair → clip negatives, normalise rows  │
│      d. Evaluate fitness (5-component function)   │
│      e. Update Pbest, Gbest                       │
│      f. Check convergence                         │
└──────────────┬───────────────────────────────────┘
               ▼
┌─────────────────────────────────────┐
│   5. Return Gbest allocation        │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│   6. Analyse & visualise results    │
└─────────────────────────────────────┘
```

---

## Fitness Function

The fitness function combines five components into a single **minimisation** objective:

```
F(x) = w₁·F_imbalance + w₂·F_overload + w₃·F_loss + w₄·F_demand + w₅·F_efficiency
```

### Component Details

| # | Component | Formula | Default Weight |
|---|-----------|---------|---------------|
| 1 | Load Imbalance | std(utilisation₁, …, utilisationₗ) | 10 |
| 2 | Overload Penalty | Σⱼ max(0, loadⱼ − capⱼ)² | 1000 |
| 3 | Distribution Loss | Σⱼ k·(loadⱼ/capⱼ)² | 5 |
| 4 | Demand Penalty | Σᵢ \|allocatedᵢ − demandᵢ\| | 1000 |
| 5 | Efficiency | max(0, target_util − mean_util) | 2 |

- The **loss model** uses a simplified quadratic approximation with coefficient k = 0.02. It is **not** an AC power-flow calculation.
- Overload and demand penalties use large weights (1000) to strongly discourage infeasible solutions.

---

## Installation & Execution

### Prerequisites

- Python 3.10 or higher

### Setup

```bash
# Clone or download the project
cd smart_grid_pso

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

### Running Tests

```bash
pytest tests/ -v
```

---

## Sample Results

### Normal Demand Scenario

| Metric | Before PSO | After PSO |
|--------|-----------|-----------|
| Overloaded Lines | 2 | 0 |
| Load Imbalance (std) | 0.32 | 0.02 |
| Distribution Loss | 0.068 | 0.042 |
| Fitness Score | ~650 | ~0.85 |
| Convergence | — | ~80 iterations |

*Actual values depend on the random seed and PSO parameters.*

### Dashboard Pages

1. **Dashboard Overview** — Executive KPIs, live distribution-line status, and real-time line loading chart
2. **Grid Configuration** — Interactive area demands (10 areas), line ratings (5 lines), scenario presets, and feasibility checks
3. **PSO Optimization** — Swarm parameter controls, live generation progress, dispatch matrix, and CSV/PDF report exports
4. **Before vs After** — Baseline vs PSO comparative metrics table, loading comparison bars, and 5-criteria performance radar
5. **Analytics Workspace** — Multi-tab telemetry (load heatmaps, utilization & headroom, dispatch sunburst, loss analysis, convergence trajectory, area demand profile)

---

## Limitations

1. **Simplified loss model** — Quadratic loading approximation; does not model real AC power flow, impedance, voltage, or reactive power.
2. **Static demand** — Demand is fixed during optimisation; time-varying profiles are not simulated.
3. **Full connectivity** — Any area can draw from any line; real grids have topological constraints.
4. **Metaheuristic** — PSO finds near-optimal solutions but does not guarantee global optimality.
5. **No real-time control** — This is a planning/simulation tool, not a real-time grid controller.

---

## Future Scope

- Time-series demand simulation (24-hour profiles).
- Multi-objective optimisation with Pareto front (NSGA-II).
- Renewable energy integration (solar/wind intermittency).
- Real transmission-line impedance and power-flow models (e.g., DC-OPF).
- Hybrid PSO with local search (memetic algorithm).
- Battery energy storage scheduling.
- SCADA / IoT integration for live monitoring.
- Demand-response optimisation.

---

## Disclaimer

> **This is a simplified simulation-based academic project for demonstrating Evolutionary Computing concepts. It is not a certified electrical power-flow analysis, grid-control system, or production-ready energy management platform. The distribution loss model is a simplified mathematical approximation and does not represent actual grid losses.**

---

## License

This project is developed for academic purposes as part of an Evolutionary Computing course.

---

*Built with Python, Streamlit, NumPy, Plotly, and Pandas.*

