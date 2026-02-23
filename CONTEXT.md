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
├── ctmc.py                  # Main module (~1,470 lines). All core logic lives here.
├── ctmc_parallel.py         # MPI wrapper for cluster-scale batched computation
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
| `cyclic_generator(S, N, ...)` | Ring topology with directional bias |
| `detailed_balance_generator(S, N, energy, beta)` | Equilibrium systems (zero EPR) |
| `arrhenius_pump_generator(S, N, energy, barrier, n_pumps, pump_strength)` | Nonequilibrium with catalytic pumps |
| `exponential_generator(S, N, scale)` | Exponentially distributed rates |

### Sparsification

```python
from ctmc import sparsify

R_sparse = sparsify(R, avg_degree=10, ensure_connected=True, seed=42)
```

Removes edges via Erdos-Renyi sampling while maintaining strong connectivity (irreducibility). Automatically repairs disconnected components. Respects forbidden transition symmetry.

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
- **Time-reversal involutions** (`time_even_states=False`, `involution_indices`, `involution_builder()`): Allows arbitrary state-swap involutions beyond the identity. The time-reversed rate matrix and forbidden transition symmetry both respect the involution structure.
- **JAX L-BFGS solver** (`get_meps_jax()`): Uses softmax parameterization for unconstrained optimization. Better convergence than Euler, especially for large state spaces.

### Known rough edges:
- MEPS Euler method can struggle near simplex boundaries (probabilities approaching zero)
- Batch mode is memory-intensive for large N or S (everything is dense numpy arrays)
- Some notebooks are exploratory scratch work, not polished documentation
- The `min_rate` default was changed from `1e-32` to `1e-12` at some point; the FORBIDDEN_TRANSITIONS_README.md references the old default

---

## Common Tasks an LLM Might Be Asked To Do

- **Add a new rate matrix generator**: Follow the pattern of existing generators in `ctmc.py`. Takes `(S, N, **kwargs)`, returns `(N, S, S)` rate matrix with rows summing to zero and non-negative off-diagonals.

- **Add a new thermodynamic quantity**: Add a method to `ContinuousTimeMarkovChain`. Handle both single and batch cases. If it involves log-ratios of rates, mask out forbidden transitions.

- **Improve MEPS convergence**: The Euler solver in `get_meps()` and the JAX solver in `get_meps_jax()` are the two current approaches. Any new solver should return a normalized probability distribution and handle batch mode.

- **Modify notebooks**: The sample_notebooks/ directory contains Jupyter notebooks for demos and research. These are the primary user-facing output of the project.

- **Performance optimization**: Batch operations use numpy broadcasting extensively. JAX is used optionally for MEPS. Any new computation-heavy code should work with the batch dimension pattern.
