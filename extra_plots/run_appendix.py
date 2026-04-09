"""
Appendix figure experiments: ER and SW sparsity sweeps with D_KL divergence.

This script performs sparsity sweep experiments on continuous-time Markov chains
(CTMCs) with either Erdős-Rényi (ER) or Small-World (SW) network topologies,
computing entropy production rates (EPR) and Kullback-Leibler divergences across
multiple system sizes and trials.

OUTPUTS:
    Saves .npz files to data_appendix_{topology}_{pump_sign}/ with keys:
    - s_values: list of system sizes used
    - s_trials: list of trial counts per system size
    - {SSSSS}: system-size keys (5-digit zero-padded) containing arrays of shape (5, N_trials):
      [MEPS_EPR, NESS_EPR, UNIF_EPR, DKL_NESS_MEPS, DKL_MEPS_NESS]
      where MEPS=MaxEnt pseudosteady-state, NESS=non-equilibrium steady-state, UNIF=uniform

USAGE:
    python run_appendix.py <topology> <pump_sign> <edge_frac>

    topology:   'er' (Erdős-Rényi) or 'sw' (Small-World)
    pump_sign:  'pos' (positive pump) or 'neg' (negative pump)
    edge_frac:  0.10, 0.25, 0.50, or 1.00 (fraction of possible edges)

EXAMPLES:
    python run_appendix.py er pos 0.10
    python run_appendix.py sw neg 0.50
"""
import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ctmc import (
    ContinuousTimeMarkovChain as MC,
    arrhenius_pump_generator,
    sparsify,
    small_world_sparsify,
)

# ── Parameters ──
# System sizes to test and corresponding trial counts (decreasing due to computational cost)
S_VALS   = [5,   10,  25,  50, 100, 250, 500, 1000, 2000]
S_TRIALS = [500, 300, 200, 200, 100,  50,  30,   15,    5]

# Pump strength: positive (+2 = +200%) and negative (-2 = -200%) shifts
PUMP_STRENGTH_POS = 2
PUMP_STRENGTH_NEG = -2

# Fraction of transitions to pump (20% of possible S*(S-1) transitions)
PUMP_RATIO = 0.20

# Arrhenius generator arguments: mean=0, std_dev=1
GEN_ARGS = [0, 1]

# Small-World rewiring probability (Watts-Strogatz parameter)
SW_BETA = 0.1

# Memory limit per batch: chunk trials if needed to stay under this threshold
MAX_BATCH_BYTES = 200_000_000

# Seed base for reproducibility (modified by trial index and parameters)
SEED_BASE = 12345


def run_batch(S, N, pump_strength, sparsity_frac=None, sw_frac=None, sw_beta=None):
    """
    Run N trials of CTMC experiments and compute EPR and D_KL metrics.

    Creates a continuous-time Markov chain with an Arrhenius generator and
    optional sparsification (ER or SW topology), then computes:
    - Entropy production rates (EPR) for MEPS, NESS, and uniform distributions
    - Kullback-Leibler divergences between NESS and MEPS (both directions)

    Args:
        S: System size (number of states)
        N: Number of trials
        pump_strength: Strength of the pump (positive or negative)
        sparsity_frac: Fraction of edges for ER sparsification (0 < frac <= 1.0).
                       If None or 1.0, no ER sparsification applied.
        sw_frac: Fraction of edges for SW sparsification. If None, no SW applied.
        sw_beta: Rewiring probability for SW (Watts-Strogatz). Required if sw_frac given.

    Returns:
        np.ndarray of shape (5, N) containing:
        [MEPS_EPR, NESS_EPR, UNIF_EPR, DKL_NESS_MEPS, DKL_MEPS_NESS]
    """
    # Calculate number of pumped transitions (20% of possible transitions)
    n_pumps = max(1, int(PUMP_RATIO * (S**2 - S)))

    # Create base CTMC with Arrhenius generator and pump
    machine = MC(
        generator=arrhenius_pump_generator,
        S=S, N=N,
        n_pumps=n_pumps,
        pump_strength=pump_strength,
        gen_args=list(GEN_ARGS),
    )
    machine.verbose = False

    # Apply ER (Erdős-Rényi) sparsification if requested
    if sparsity_frac is not None and sparsity_frac < 1.0:
        avg_deg = max(2, sparsity_frac * (S - 1))
        R_sparse = sparsify(machine.R, avg_degree=avg_deg)
        machine = MC(R=R_sparse)
        machine.verbose = False

    # Apply SW (Small-World) sparsification if requested
    # Uses matched edge density to ER: k=int(sw_frac*(S-1)), adjusted to be even
    if sw_frac is not None and sw_beta is not None:
        k_eff = max(2, int(sw_frac * (S - 1)))
        # Watts-Strogatz requires even k for undirected graphs
        if k_eff % 2 != 0:
            k_eff -= 1
        k_eff = max(2, min(k_eff, S - 1))
        R_sw = small_world_sparsify(machine.R, k=k_eff, beta=sw_beta)
        machine = MC(R=R_sw)
        machine.verbose = False

    # Compute three reference steady-state distributions
    ness = machine.get_ness()  # Non-equilibrium steady-state
    meps = machine.get_meps()  # MaxEnt pseudosteady-state
    unif = machine.get_uniform()  # Uniform distribution

    # Compute entropy production rates for each distribution
    meps_epr = np.atleast_1d(machine.get_epr(meps))
    ness_epr = np.atleast_1d(machine.get_epr(ness))
    unif_epr = np.atleast_1d(machine.get_epr(unif))

    # Compute Kullback-Leibler divergences between NESS and MEPS
    # Reshape to (N, S) if single trial returns 1D array
    if ness.ndim == 1:
        ness = ness[np.newaxis, :]
        meps = meps[np.newaxis, :]
    dkl_nm = np.array([machine.dkl(ness[i], meps[i]) for i in range(N)])
    dkl_mn = np.array([machine.dkl(meps[i], ness[i]) for i in range(N)])

    return np.array([meps_epr, ness_epr, unif_epr, dkl_nm, dkl_mn])


def run_config(S, N, pump_strength, sparsity_frac=None, sw_frac=None, sw_beta=None, seed=None):
    """
    Run N trials with automatic memory-safe batching.

    For large system sizes, N trials may exceed memory limits. This function
    automatically chunks trials into batches that fit within MAX_BATCH_BYTES,
    processes each batch with run_batch(), and concatenates results.

    Args:
        S: System size (number of states)
        N: Total number of trials to run
        pump_strength: Strength of the pump
        sparsity_frac: Fraction of edges for ER sparsification
        sw_frac: Fraction of edges for SW sparsification
        sw_beta: Rewiring probability for SW
        seed: Random seed for reproducibility

    Returns:
        np.ndarray of shape (5, N) with concatenated results from all batches
    """
    if seed is not None:
        np.random.seed(seed)

    # Estimate memory per trial: each CTMC has rate matrix of size S x S (8 bytes per float64)
    bytes_per_trial = S * S * 8

    # Calculate maximum trials per batch that fit in memory limit
    batch_size = max(1, int(MAX_BATCH_BYTES / bytes_per_trial))

    # If all trials fit in one batch, run directly
    if batch_size >= N:
        return run_batch(S, N, pump_strength, sparsity_frac, sw_frac, sw_beta)

    # Otherwise, process in chunks and concatenate
    all_results = []
    remaining = N
    while remaining > 0:
        trials_in_batch = min(batch_size, remaining)
        batch_result = run_batch(S, trials_in_batch, pump_strength, sparsity_frac, sw_frac, sw_beta)
        all_results.append(batch_result)
        remaining -= trials_in_batch
        gc.collect()  # Free memory after each batch

    return np.concatenate(all_results, axis=1)


def main():
    """
    Main entry point: parse arguments, run sparsity sweep, and save results.

    Processes command-line arguments and runs the sparsity sweep experiments
    for either ER or SW topology with the specified pump direction and edge
    fraction. Results are saved to an .npz file with system sizes, trial counts,
    and per-size metric arrays.
    """
    if len(sys.argv) != 4:
        print("Usage: python run_appendix.py <er|sw> <pos|neg> <edge_frac>")
        sys.exit(1)

    topology = sys.argv[1]    # 'er' or 'sw'
    pump_sign = sys.argv[2]   # 'pos' or 'neg'
    edge_frac = float(sys.argv[3])

    pump_strength = PUMP_STRENGTH_POS if pump_sign == "pos" else PUMP_STRENGTH_NEG

    # Create output directory: data_appendix_{topology}_{pump_sign}/
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f'data_appendix_{topology}_{pump_sign}')
    os.makedirs(out_dir, exist_ok=True)

    # Filename label: frac{edge_frac*100:03d} (e.g., frac025 for 0.25)
    frac_label = f"frac{int(edge_frac*100):03d}"

    print(f"\n{'='*60}")
    print(f"  {topology.upper()} | pump={pump_sign} (strength={pump_strength}) | edge_frac={edge_frac}")
    print(f"{'='*60}")

    # Computational cost limits: MEPS optimizer becomes slow at large S with dense graphs
    # Empirical thresholds based on observed runtime
    if edge_frac >= 0.50:
        max_s = 1000  # Dense graphs slow down significantly
    else:
        max_s = 2000  # Sparse graphs can handle larger sizes

    output = {}
    for i, (s, n) in enumerate(zip(S_VALS, S_TRIALS)):
        if s > max_s:
            print(f"  S={s:>5} skipped (too slow at edge_frac={edge_frac})")
            continue
        t0 = time.time()

        if topology == "er":
            # Erdős-Rényi: average degree is proportional to edge_frac*(S-1)
            actual_frac = edge_frac if edge_frac < 1.0 else 1.0
            avg_deg = max(2, actual_frac * (s - 1)) if actual_frac < 1.0 else s - 1
            ed = avg_deg / (s - 1) if s > 1 else 1.0
            print(f"  S={s:>5}, N={n:>4}, avg_deg={avg_deg:.1f} (ed={ed:.3f}) ...",
                  end=" ", flush=True)
            results = run_config(
                S=s, N=n, pump_strength=pump_strength,
                sparsity_frac=edge_frac if edge_frac < 1.0 else None,
                seed=SEED_BASE + i + int(edge_frac * 1000) + (0 if pump_sign == "pos" else 5000),
            )
        elif topology == "sw":
            # Small-World: k is the number of neighbors in the ring, adjusted to be even
            k_eff = max(2, int(edge_frac * (s - 1)))
            if k_eff % 2 != 0:
                k_eff -= 1
            k_eff = max(2, min(k_eff, s - 1))
            ed = k_eff / (s - 1) if s > 1 else 1.0
            print(f"  S={s:>5}, N={n:>4}, k={k_eff} (ed={ed:.3f}, beta={SW_BETA}) ...",
                  end=" ", flush=True)
            results = run_config(
                S=s, N=n, pump_strength=pump_strength,
                sw_frac=edge_frac, sw_beta=SW_BETA,
                seed=SEED_BASE + i + int(edge_frac * 1000) + (0 if pump_sign == "pos" else 5000),
            )

        # Store results with system size as 5-digit zero-padded key
        output[f"{s:05}"] = results
        dt = time.time() - t0
        print(f"done ({dt:.1f}s)")

    # Collect system sizes and trial counts that were actually computed
    s_used = [s for s in S_VALS if f"{s:05}" in output]
    t_used = [S_TRIALS[S_VALS.index(s)] for s in s_used]

    # Save results to .npz file
    fname = os.path.join(out_dir, f"{frac_label}.npz")
    np.savez(fname,
             s_values=np.array(s_used),
             s_trials=np.array(t_used),
             **output)
    print(f"  Saved → {fname}")


if __name__ == "__main__":
    main()
