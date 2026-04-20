# discrete_states — LLM Context & Handoff Document

## What This Project Is

A research implementation of a **Continuous-Time Markov Chain (CTMC) framework** for studying entropy production in far-from-equilibrium thermodynamic systems. It implements the theory from:

> "Large Interconnected Thermodynamic Systems Nearly Minimize Entropy Production"
> K. J. Ray and A. B. Boyd (2025), arXiv:2507.10476

The central question: how do nonequilibrium systems self-organize, and under what conditions do they approach states of minimum entropy production?

---

## Repository Structure

```
discrete_states/
├── ctmc.py                  # Main module (~1,680 lines). All core logic lives here.
├── ctmc_parallel.py         # Legacy MPI wrapper (replaced by run_sweep_v2.py)
├── run_sweep_v2.py          # Data generation for main figure (replaces ctmc_parallel.py)
├── run_sweep_all.sh         # Run full sweep (small S batched, large S serial)
├── run_sweep_large_s.sh     # Accumulate large-S trials incrementally
├── verify_sweep_data.py     # Verify output format and flag anomalies
├── test_refactored.py       # Comprehensive test suite (~800 lines)
├── FORBIDDEN_TRANSITIONS_README.md  # Docs on the forbidden transitions feature
├── CONTEXT.md               # This file
├── sample_notebooks/
│   ├── ctmc_tutorial.ipynb           # Getting-started guide
│   ├── ctmc_plots.ipynb              # Rate matrix and distribution visualization
│   ├── ctmc_testbed.ipynb            # Experimental scratch notebook
│   ├── time_symmetry.ipynb           # Involution / time-reversal demos
│   ├── sparsification_demo.ipynb     # Sparse CTMC construction
│   ├── sparsity_and_time_symmetry.ipynb
│   ├── meps_method_comparison.ipynb  # Euler vs. JAX L-BFGS comparison
│   ├── pump_scaling_demo.ipynb       # Large-scale Arrhenius pump studies
│   └── plot_sweeps.ipynb             # Parameter sweep visualization
├── extra_plots/
│   ├── plot_common.py               # Shared constants and utilities for plotting
│   ├── plot_appendix.py             # 10 appendix figures (Arrhenius, cyclic, decay)
│   ├── plot_fixed_degree.py         # 4-panel fixed-degree sweep figure
│   ├── run_appendix.py              # Data generation: Arrhenius ER/SW experiments
│   ├── run_cyclic_appendix.py       # Data generation: cyclic ring experiments
│   ├── run_fixed_degree_v2.py       # Data generation: fixed-degree sweep
│   ├── run_fixed_degree_v2_dense.py # Data generation: dense baseline
│   ├── data_*/                      # Generated data directories (gitignored)
│   └── final_plots/                 # Generated figures (gitignored)
└── *.npz                    # Cached numerical results (not tracked in git)
```

---

## Core Concepts (Physics)

You need to understand these to work on the code:

- **Rate matrix R**: An S x S matrix where `R[i,j]` (i != j) is the rate of transitioning from state i to state j. Diagonal entries are negative and satisfy `R[i,i] = -sum(R[i,j] for j != i)` (rows sum to zero). This is standard CTMC convention.

- **NESS (Nonequilibrium Steady State)**: The stationary distribution pi satisfying `R^T @ pi = 0`. For irreducible chains, it is unique. Computed via analytic inversion, eigenvalue decomposition, or numeric integration (method chosen by heuristic based on system size).

- **EPR (Entropy Production Rate)**: Measures thermodynamic irreversibility. Defined as a sum over all pairs of states involving `R[i,j] * pi[i] * log(R[i,j] / R[j,i])`. Non-negative for any distribution; zero only at detailed balance (equilibrium).

- **MEPS (Minimum Entropy Production State)**: The probability distribution that minimizes EPR. A key prediction of the paper is that large, interconnected systems have MEPS close to NESS. Found via gradient descent (Euler method in numpy, or L-BFGS in JAX).

- **Detailed balance**: When `R[i,j] * pi[i] = R[j,i] * pi[j]` for all i,j. Systems in detailed balance have zero EPR and represent thermodynamic equilibrium.

- **Forbidden transitions**: Pairs (i,j) where `R[i,j] = R[j,i] = 0`. The code enforces this symmetry automatically. These are structurally absent edges in the state-transition graph.

- **Time-reversal symmetry / Involutions**: An involution sigma is a permutation where sigma(sigma(i)) = i. It defines how states map under time-reversal. `time_even_states=True` means identity involution (every state maps to itself). `time_even_states=False` allows arbitrary involutions where states can swap with a partner. The time-reversed rate matrix uses `rev_R[i,j] = R[sigma(j), sigma(i)]`.

---

## Key Class: `ContinuousTimeMarkovChain`

### Construction

```python
# From a generator function:
ctmc = ContinuousTimeMarkovChain(
    generator=arrhenius_pump_generator,
    S=50, N=100,            # 50 states, batch of 100 systems
    n_pumps=500, pump_strength=5
)

# From an explicit rate matrix:
ctmc = ContinuousTimeMarkovChain(R=my_rate_matrix)

# With time-reversal involution:
ctmc = ContinuousTimeMarkovChain(R=R, time_even_states=False)

# time_even_states is now a constructor parameter (default True).
# Previously it was hardcoded and had to be set after construction.
```

### Important Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `R` | `(S,S)` or `(N,S,S)` | Rate matrix (validated, forbidden transitions zeroed) |
| `rev_R` | same | Time-reversed rate matrix |
| `S` | int | Number of states |
| `batch` | bool | Whether operating in batch mode (N systems) |
| `forbidden_mask` | `(S,S)` or `(N,S,S)` bool | True where transitions are forbidden |
| `statewise_Q` | `(S,S)` or `(N,S,S)` | Precomputed per-state entropy flow contributions |
| `min_rate` | float | Threshold below which rates become forbidden (default 1e-12) |
| `time_even_states` | bool | Whether involution is identity |
| `involution_indices` | `(S,)` int array | Permutation defining the involution |

### Key Methods

**Steady states:**
- `get_ness()` → pi: Nonequilibrium steady state
- `get_meps(n_iter, lr, ...)` → p: Minimum EPR state (Euler method)
- `get_meps_jax(n_restarts, ...)` → p: MEPS via JAX L-BFGS (preferred when JAX available)

**Thermodynamic quantities (all accept a state vector):**
- `get_epr(state)` → scalar or (N,): Total entropy production rate
- `get_statewise_epr(state)` → (S,) or (N,S): Per-state EPR contributions
- `get_activity(state)` → scalar or (N,): Total transition activity
- `get_prob_current(state)` → scalar or (N,): Total probability current

**Dynamics:**
- `get_time_deriv(state)` → dp/dt under the master equation
- `evolve_state(state, dt)` → Forward Euler step with positivity clipping

**State constructors:**
- `get_uniform()`, `get_random_state()`, `get_local_state(mu, sigma)`

### Rate Matrix Generators

All return an `(S,S)` or `(N,S,S)` rate matrix:

| Generator | Use Case |
|-----------|----------|
| `uniform_generator(S, N)` | Random rates, uniformly distributed |
| `normal_generator(S, N, mu, sigma)` | Gaussian-distributed rates |
| `gamma_generator(S, N, mu, sigma)` | Gamma-distributed rates |
| `spiral_staircase_generator(S, N, ...)` | Chain with catalytic shortcut (legacy) |
| `cyclic_generator(S, N, ..., decay_alpha)` | True ring topology with modular distance |
| `detailed_balance_generator(S, N, energy, beta)` | Equilibrium systems (zero EPR) |
| `arrhenius_pump_generator(S, N, energy, barrier, n_pumps, pump_strength)` | Nonequilibrium with catalytic pumps |
| `exponential_generator(S, N, scale)` | Exponentially distributed rates |

### Sparsification

```python
from ctmc import sparsify, small_world_sparsify

# Erdos-Renyi sparsification: random edge removal
R_sparse = sparsify(R, avg_degree=10, ensure_connected=True, seed=42)

# Watts-Strogatz small-world sparsification: ring lattice + rewiring
R_sw = small_world_sparsify(R, k=6, beta=0.1, ensure_connected=True, seed=42)
```

`sparsify()` removes edges via Erdos-Renyi sampling while maintaining strong connectivity (irreducibility). `small_world_sparsify()` starts with a ring lattice of degree k and rewires each edge with probability beta, producing networks with high clustering and short path lengths. Both automatically repair disconnected components and respect forbidden transition symmetry.

---

## extra_plots/ — Numerical Experiments & Figures

The `extra_plots/` directory contains the numerical experiments supporting the paper's appendix. The workflow is: `run_*.py` scripts generate `.npz` data files, then `plot_*.py` scripts produce publication figures.

**Data generation scripts** (`run_*.py`): Each script sweeps system size S with many random trials per size, computing NESS EPR, MEPS EPR, uniform EPR, and D_KL(NESS || MEPS) for each trial. Experiments cover Arrhenius pumps on Erdos-Renyi and small-world topologies, cyclic ring generators with and without distance decay, and a fixed-degree sweep holding average degree constant while growing S.

**Plotting scripts** (`plot_*.py`): Both `plot_appendix.py` and `plot_fixed_degree.py` import shared infrastructure from `plot_common.py`. They accept a CLI argument (`mean` or `median`) to control the center line statistic, and always show IQR (25th/75th percentile) bands. Usage: `python plot_appendix.py mean` or `python plot_fixed_degree.py median`.

**Equilibrium filter**: All plotting scripts discard trials where NESS EPR < 1e-10 (the `EQ_THRESHOLD` constant). These are effectively equilibrium systems produced when random generators happen to create near-canceling cycle forces, leaving EPR at the float-noise floor (~1e-16). A clear bimodal gap of many orders of magnitude separates these from the weakest genuinely driven systems (~1e-8), so the exact threshold is not critical. The filter drops only ~0.4% of trials, concentrated at small system sizes (S=10, S=25) in sparse configurations.

**Key metric**: The primary quantity of interest is the excess EPR ratio `sigma_NESS / sigma_MEPS - 1`, which measures how far the NESS entropy production exceeds the theoretical minimum. Values near zero indicate that NESS is close to MEPS.

---

## Main Sweep Data Generation (run_sweep_v2.py)

The primary figure in the paper (Fig. 6) shows how the scaled excess EPR ratio converges toward zero as system size grows, across different pump strengths and pump fractions. The data for this figure is generated by `run_sweep_v2.py`, which replaces the original MPI-based `ctmc_parallel.py`.

### Data format

Each `.npz` file stores per-system-size arrays with shape `(5, N_trials)`:

| Row | Contents |
|-----|----------|
| 0 | MEPS EPR |
| 1 | NESS EPR |
| 2 | Uniform EPR |
| 3 | D_KL(NESS \|\| MEPS) |
| 4 | D_KL(MEPS \|\| NESS) |

Rows 0-2 are backward-compatible with the old 3-row format. The notebook `plot_sweeps.ipynb` only indexes rows 0-2, so it works unchanged.

Directory layout (matches what `plot_sweeps.ipynb` expects):
```
sample_notebooks/plot_data_v2/
├── 0005/              # pump_strength = 5
│   ├── ratio_005.npz  # pump_ratio = 5%
│   ├── ratio_020.npz  # pump_ratio = 20%
│   └── ratio_080.npz  # pump_ratio = 80%
├── 0025/              # pump_strength = 25
│   └── ...
├── 0100/              # pump_strength = 100
├── 0300/              # pump_strength = 300
├── 0400/              # pump_strength = 400
└── 0500/              # pump_strength = 500
```

### Parameter grid

- **Pump strengths**: 5, 25, 100, 300, 400, 500 (percentage of energy scale)
- **Pump ratios**: 5%, 20%, 80% (fraction of transitions that are pumped)
- **System sizes**: 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000
- **Trials per size**: 10000, 5000, 2000, 1000, 1000, 500, 250, 125, 25, 15, 6, 4, 2
- **Energy distribution**: `gen_args=[0, 1]` → E_s ~ U(0,1), barriers ~ U(0,1), pump forces F ~ U(0, α)
- **Generator**: `arrhenius_pump_generator` (dense, fully-connected graphs — no sparsification)

### How to run the experiment

The sweep is 6 strengths × 3 ratios = 18 independent jobs. System sizes ≤ 1000 are fast enough to batch; sizes ≥ 2000 are compute-intensive and should run one-at-a-time.

**Full sweep (all sizes, all parameters):**
```bash
bash run_sweep_all.sh
```
This runs small S (5–1000) in batch, then large S (2000–16000) one-at-a-time for fault tolerance. Safe to interrupt and re-run — it appends to existing data files.

**Single (strength, ratio) pair:**
```bash
python run_sweep_v2.py 100 20                       # full sweep for strength=100, ratio=20%
python run_sweep_v2.py 100 20 --s-values 5 10 25    # specific S values only
python run_sweep_v2.py 100 20 --use-jax              # use JAX MEPS optimizer (faster on GPU)
```

**Accumulating large-S trials incrementally:**
```bash
# Each invocation adds trials to the existing file
bash run_sweep_large_s.sh 16000      # 1 trial of S=16000 for all 18 (strength, ratio) pairs
bash run_sweep_large_s.sh 16000      # run again to accumulate
bash run_sweep_large_s.sh 8000 3     # 3 trials of S=8000
```

**Parallelization strategy for a cluster:**
- S ≤ 1000: Run all 18 (strength, ratio) jobs in parallel. Each takes minutes.
- S = 2000–4000: Run each (strength, ratio, S) triple as a separate job. Each takes ~10 min.
- S ≥ 8000: Run one trial at a time. Each S=16000 trial takes significant time and memory (~4 GB per rate matrix). Submit many single-trial jobs and let them accumulate via incremental append.
- If JAX + GPU available, use `--use-jax` for significantly faster MEPS optimization at large S.

**Verify output:**
```bash
python verify_sweep_data.py sample_notebooks/plot_data_v2
```
Checks format, flags MEPS > NESS violations, reports trial counts vs targets.

### Important notes on energy distribution

The `arrhenius_pump_generator` uses `gen_args` for both state energies and pump forces. With `gen_args=[0, 1]`:
- State energies: E_s ~ U(0, 1)
- Energy barriers: E_{s↔s'} ~ U(0, 1) (then abs applied, which is a no-op)
- Pump forces: F_{s→s'} ~ U(0, α) where α = pump_strength

This means pumps only slow down transitions in one direction (F ≥ 0), creating asymmetry by making the reverse direction relatively faster. The manuscript should specify F ∈ [0, α], not F ∈ [−α, α]. Symmetric pumping (F ∈ [−α, α]) causes numerical overflow at high pump strengths due to rates spanning > 10^87.

---

## Architectural Patterns & Conventions

1. **Batch dimension**: Single system uses shape `(S,S)` for R and `(S,)` for states. Batch mode uses `(N,S,S)` and `(N,S)`. The `self.batch` flag controls which path is taken. All methods handle both cases.

2. **Lazy invalidation**: Calling `set_rate_matrix()` deletes cached attributes (`ness`, `meps`, `rev_R`, etc.) via `delattr`, forcing recomputation on next access.

3. **Forbidden transition safety**: Any computation involving `log(R[i,j])` or `log(R[i,j]/R[j,i])` must mask out forbidden transitions (where both rates are zero) to avoid `log(0)`. The `statewise_Q` precomputation handles this.

4. **Involution symmetry for forbidden transitions**: When `time_even_states=False`, if transition s->s' is forbidden, then sigma(s')->sigma(s) must also be forbidden (where sigma is the involution).

5. **NESS solver strategy**: Tries analytic matrix inversion first (fast, works for small/well-conditioned systems), falls back to eigenvalue method, then numeric integration. Selection is threshold-based on S.

6. **MEPS optimization**: Euler method uses forward Euler on the MEPS gradient flow with adaptive learning rate decay and positivity clipping. JAX method uses softmax parameterization (unconstrained optimization) with L-BFGS. JAX is generally more reliable.

7. **No external state**: Everything lives on the `ContinuousTimeMarkovChain` instance. No global state, no singletons.

---

## Running Tests

```bash
python test_refactored.py
```

Tests cover: rate matrix validation, probability conservation, EPR properties (MEPS <= NESS), detailed balance (zero EPR), forbidden transitions, batch mode, graph connectivity, involutions, and MEPS convergence.

---

## Optional Dependencies

- **numpy** (required): Core numerical computation
- **scipy** (required): Eigenvalue solvers, optimization fallback
- **JAX + jaxopt** (optional): L-BFGS MEPS solver. Significantly better convergence than Euler method. Code gracefully falls back to scipy if unavailable.
- **mpi4py** (optional): For `ctmc_parallel.py` cluster computation.
- **matplotlib** (optional): Notebooks use it for plotting.

---

## Development History & Current State

### Stable and complete:
- Core CTMC class with all thermodynamic quantities
- All rate matrix generators
- NESS computation (three solver strategies)
- MEPS optimization (Euler and JAX L-BFGS)
- Forbidden transitions with symmetric enforcement
- Sparsification with connectivity guarantees
- Graph connectivity utilities (irreducibility check, SCC detection, repair)
- Batch mode for all operations
- Comprehensive test suite

### Recently added (working but may evolve):
- **Time-reversal involutions** (`time_even_states=False`, `involution_indices`, `involution_builder()`): Allows arbitrary state-swap involutions beyond the identity. The time-reversed rate matrix and forbidden transition symmetry both respect the involution structure. `time_even_states` is now a constructor parameter.
- **JAX L-BFGS solver** (`get_meps_jax()`): Uses softmax parameterization for unconstrained optimization. Better convergence than Euler, especially for large state spaces.
- **Small-world sparsification** (`small_world_sparsify()`): Watts-Strogatz rewiring of ring lattices, preserving rate values from the original dense matrix.
- **Appendix experiment suite** (`extra_plots/`): Full pipeline of data generation and plotting scripts for the paper's numerical appendix, with equilibrium filtering, mean/median center lines, and IQR bands.

### Known rough edges:
- MEPS Euler method can struggle near simplex boundaries (probabilities approaching zero)
- Batch mode is memory-intensive for large N or S (everything is dense numpy arrays)
- The `min_rate` default was changed from `1e-32` to `1e-12` at some point; the FORBIDDEN_TRANSITIONS_README.md references the old default

---

## Common Tasks an LLM Might Be Asked To Do

- **Add a new rate matrix generator**: Follow the pattern of existing generators in `ctmc.py`. Takes `(S, N, **kwargs)`, returns `(N, S, S)` rate matrix with rows summing to zero and non-negative off-diagonals.

- **Add a new thermodynamic quantity**: Add a method to `ContinuousTimeMarkovChain`. Handle both single and batch cases. If it involves log-ratios of rates, mask out forbidden transitions.

- **Improve MEPS convergence**: The Euler solver in `get_meps()` and the JAX solver in `get_meps_jax()` are the two current approaches. Any new solver should return a normalized probability distribution and handle batch mode.

- **Modify notebooks**: The sample_notebooks/ directory contains Jupyter notebooks for demos and research. These are the primary user-facing output of the project.

- **Performance optimization**: Batch operations use numpy broadcasting extensively. JAX is used optionally for MEPS. Any new computation-heavy code should work with the batch dimension pattern.
