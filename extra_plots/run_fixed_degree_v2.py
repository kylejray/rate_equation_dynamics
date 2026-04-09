"""Fixed-degree sparsity sweep with repair edge tracking (v2).

This script extends the fixed-degree sweep to track how many edges were added
during connectivity repair. This reveals whether connectivity becomes a limiting
factor for sparse networks.

Key Insight:
    By measuring the fraction of edges that must be added to maintain
    irreducibility, we can identify when the sparsification process itself
    becomes problematic for maintaining connected components.

Data Generated:
    - MEPS_EPR, NESS_EPR, UNIF_EPR: EPR across three distributions
    - repair_fracs: fraction of edges added by repair step per trial

Output:
    Saves .npz file to data_fixed_degree_v2/degXXX.npz with keys:
    - s_values: array of system sizes tested
    - s_trials: array of trial counts per system size
    - 'SSSSS': shape (3, N_trials) = [MEPS_EPR, NESS_EPR, UNIF_EPR]
    - 'SSSSS_repair': shape (N_trials,) = fraction of repaired edges

Usage:
    python run_fixed_degree_v2.py <degree>

    degree: integer avg_degree (e.g., 4, 8, 16, 32, 64)

Example:
    python run_fixed_degree_v2.py 8
    # Generates data_fixed_degree_v2/deg008.npz
"""
import sys
import os
import time
import gc

sys.setrecursionlimit(10000)
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ctmc import (
    ContinuousTimeMarkovChain as MC,
    arrhenius_pump_generator,
    is_irreducible,
)
# We need to do sparsification manually to capture repair counts
from ctmc import _repair_connectivity

# ---- Markov chain generator parameters ----
PUMP_STRENGTH = 2  # Strength of drive in Arrhenius pump
PUMP_RATIO = 0.20  # Fraction of total possible edges that are pumped
GEN_ARGS = [0, 1]  # Additional arguments for arrhenius_pump_generator

# ---- Reproducibility ----
SEED_BASE = 54321  # Base seed for RNG (per-trial seed = SEED_BASE + S)

# ---- System size sweep parameters ----
S_VALS   = [5,   10,  25,  50, 100, 250, 500, 1000]
S_TRIALS = [500, 300, 200, 200, 100,  50,  30,   15]


def sparsify_with_stats(R, avg_degree, seed=None):
    """Sparsify matrix with connectivity repair and track repair statistics.

    Performs sparsification by randomly removing off-diagonal edges and then
    repairs connectivity if the result is disconnected. Tracks how many edges
    were added during repair.

    Parameters
    ----------
    R : ndarray, shape (..., S, S)
        Generator matrices. Can be 2D (single) or 3D (batch).
    avg_degree : float
        Target average degree for sparsification.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    R_sparse : ndarray
        Sparsified generator(s).
    repair_fracs : ndarray
        Fraction of edges that were added by repair, per sample.
    """
    R = np.asarray(R, dtype=float)
    single = (R.ndim == 2)
    if single:
        R = R[np.newaxis, ...].copy()
    else:
        R = R.copy()

    N, S, _ = R.shape
    p = avg_degree / (S - 1)
    p = min(p, 1.0)

    rng = np.random.default_rng(seed)
    off_diag = ~np.eye(S, dtype=bool)

    repair_fracs = np.zeros(N)

    for n in range(N):
        # Create random symmetric sparsity mask
        upper = rng.random((S, S)) < p
        mask = np.triu(upper, k=1)
        mask = mask | mask.T
        mask |= np.eye(S, dtype=bool)
        R[n] *= mask

        # Count edges before repair
        edges_before = ((np.abs(R[n]) > 1e-12) & off_diag).sum()

        # Repair connectivity if disconnected
        if not is_irreducible(R[n]):
            edges_added = _repair_connectivity(R[n], rng)
        else:
            edges_added = 0

        # Track repair as fraction of final edges
        edges_after = ((np.abs(R[n]) > 1e-12) & off_diag).sum()
        repair_fracs[n] = edges_added / max(edges_after, 1)

    if single:
        return R[0], repair_fracs[0]
    return R, repair_fracs


def run_batch(S, N, avg_degree):
    """Run N trials with fixed degree and return EPR data and repair statistics.

    Parameters
    ----------
    S : int
        System size (number of states).
    N : int
        Number of independent trials.
    avg_degree : int
        Target average degree for sparsification.

    Returns
    -------
    data : ndarray, shape (3, N)
        [MEPS_EPR, NESS_EPR, UNIF_EPR] for each trial.
    repair_fracs : ndarray, shape (N,)
        Fraction of edges added by repair per trial.
    """
    n_pumps = max(1, int(PUMP_RATIO * (S**2 - S)))
    R_dense = arrhenius_pump_generator(S=S, N=N, n_pumps=n_pumps,
                                        pump_strength=PUMP_STRENGTH,
                                        gen_args=list(GEN_ARGS))

    k = min(avg_degree, S - 1)
    if k < S - 1:
        # Sparsify and track repair statistics
        R_sparse, repair_fracs = sparsify_with_stats(R_dense, avg_degree=k, seed=None)
        machine = MC(R=R_sparse)
    else:
        # Graph is already effectively dense; no sparsification
        machine = MC(R=R_dense)
        repair_fracs = np.zeros(N)

    machine.verbose = False

    # Compute steady states and distributions
    ness = machine.get_ness()
    meps = machine.get_meps()
    unif = machine.get_uniform()

    # Compute EPR for each distribution
    data = np.array([
        np.atleast_1d(machine.get_epr(meps)),
        np.atleast_1d(machine.get_epr(ness)),
        np.atleast_1d(machine.get_epr(unif)),
    ])
    return data, repair_fracs


def main():
    """Execute the fixed-degree v2 sweep with repair tracking and save results."""
    avg_degree = int(sys.argv[1])
    print(f"Fixed avg_degree = {avg_degree}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data_fixed_degree_v2')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'deg{avg_degree:03d}.npz')

    results = {}
    for S, N in zip(S_VALS, S_TRIALS):
        np.random.seed(SEED_BASE + S)
        t0 = time.time()
        data, repair_fracs = run_batch(S, N, avg_degree)
        dt = time.time() - t0
        key = f'{S:05d}'
        results[key] = data
        results[f'{key}_repair'] = repair_fracs

        density = min(avg_degree, S-1) / (S-1) * 100 if S > 1 else 100
        med_excess = np.median(data[1] / data[0] - 1)
        med_meps = np.median(data[0])
        mean_repair = repair_fracs.mean() * 100
        disconn = (repair_fracs > 0).sum()
        print(f"  S={S:5d}  N={N:3d}  density={density:5.1f}%  "
              f"med_excess={med_excess:.4f}  "
              f"med_meps_epr={med_meps:.2e}  "
              f"repair={mean_repair:.1f}% of edges  "
              f"disconnected={disconn}/{N}  "
              f"t={dt:.1f}s")
        gc.collect()

    np.savez(out_path,
             s_values=np.array(S_VALS),
             s_trials=np.array(S_TRIALS),
             **results)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
