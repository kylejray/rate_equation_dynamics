"""Dense (100% connectivity) baseline for fixed-degree v2 comparison.

This script generates the fully-connected (complete graph) baseline data in
the v2 format (with repair edge tracking). Provides a reference point showing
EPR behavior when all possible state transitions are allowed.

Since fully-connected networks require no repair, the repair_fracs are all
zeros by definition.

Data Generated:
    - MEPS_EPR, NESS_EPR, UNIF_EPR: EPR across three distributions
    - repair_fracs: always 0.0 (no repair needed for dense networks)

Output:
    Saves .npz file to data_fixed_degree_v2/dense.npz with keys:
    - s_values: array of system sizes tested
    - s_trials: array of trial counts per system size
    - 'SSSSS': shape (3, N_trials) = [MEPS_EPR, NESS_EPR, UNIF_EPR]
    - 'SSSSS_repair': shape (N_trials,) = zeros (no repair in dense case)

Usage:
    python run_fixed_degree_v2_dense.py
"""
import sys
import os
import time
import gc

sys.setrecursionlimit(10000)
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ctmc import ContinuousTimeMarkovChain as MC, arrhenius_pump_generator

# ---- System size sweep parameters ----
S_VALS   = [5,   10,  25,  50, 100, 250, 500, 1000]
S_TRIALS = [500, 300, 200, 200, 100,  50,  30,   15]

# ---- Markov chain generator parameters ----
PUMP_STRENGTH = 2  # Strength of drive in Arrhenius pump
PUMP_RATIO = 0.20  # Fraction of total possible edges that are pumped
GEN_ARGS = [0, 1]  # Additional arguments for arrhenius_pump_generator

# ---- Reproducibility ----
SEED_BASE = 54321  # Base seed for RNG (per-trial seed = SEED_BASE + S)


def main():
    """Execute the dense v2 baseline sweep and save results."""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_fixed_degree_v2')
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for S, N in zip(S_VALS, S_TRIALS):
        np.random.seed(SEED_BASE + S)
        n_pumps = max(1, int(PUMP_RATIO * (S**2 - S)))
        t0 = time.time()
        machine = MC(generator=arrhenius_pump_generator, S=S, N=N,
                     n_pumps=n_pumps, pump_strength=PUMP_STRENGTH, gen_args=list(GEN_ARGS))
        machine.verbose = False
        ness = machine.get_ness()
        meps = machine.get_meps()
        unif = machine.get_uniform()
        data = np.array([
            np.atleast_1d(machine.get_epr(meps)),
            np.atleast_1d(machine.get_epr(ness)),
            np.atleast_1d(machine.get_epr(unif)),
        ])
        dt = time.time() - t0
        key = f'{S:05d}'
        results[key] = data
        # Repair fractions are zero for dense networks (no repair needed)
        results[f'{key}_repair'] = np.zeros(N)
        print(f"  S={S:5d}  N={N:3d}  med_excess={np.median(data[1]/data[0] - 1):.4f}  t={dt:.1f}s")
        gc.collect()

    np.savez(os.path.join(out_dir, 'dense.npz'),
             s_values=np.array(S_VALS), s_trials=np.array(S_TRIALS), **results)
    print("Saved dense.npz")


if __name__ == '__main__':
    main()
