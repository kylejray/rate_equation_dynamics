"""
Cyclic ring experiments for appendix figures.

This script generates synthetic CTMC (Continuous-Time Markov Chain) data on ring
topologies under three experimental regimes to demonstrate when MEPS (minimum EPR
stationary distribution) minimization succeeds or fails.

EXPERIMENTS
───────────
1. flat_ring: Forward/backward asymmetry on ring with NO distance decay.
   - Rates ~ exp(noise), reverse ~ fwd * U(0, MAX_REVERSE_RATE)
   - Tests invariance under random reversibility scaling
   - Pattern should HOLD across system sizes

2. flat_ring_sw: Small-world rewiring applied to flat ring (preserves edge count).
   - Applies small-world shortcuts with rewiring probability beta
   - Tests robustness to topology changes at matched edge density
   - Pattern should still HOLD

3. decay_sweep: Parameterized distance-dependent decay on full ring.
   - Rates ~ exp(-alpha * dist + noise), sweeps alpha from 0 (flat) to ~2 (strong decay)
   - Tests breakdown of the MEPS-minimization hypothesis
   - Shows where the pattern BREAKS

OUTPUT DATA
───────────
Each run saves shape (5, N) arrays:
  [MEPS_EPR, NESS_EPR, UNIF_EPR, DKL(NESS||MEPS), DKL(MEPS||NESS)]
  where N is the number of trials for that system size.

USAGE
─────
    python run_cyclic_appendix.py flat_ring <edge_frac>
    python run_cyclic_appendix.py flat_ring_sw <edge_frac>
    python run_cyclic_appendix.py decay_sweep <alpha>

where:
  <edge_frac>  : fraction of maximum possible edges in ring (0, 1]
  <alpha>      : distance decay parameter in rate scaling exp(-alpha*dist)
"""
import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ctmc import ContinuousTimeMarkovChain as MC, cyclic_generator, small_world_rewire

# ──────────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────

# System sizes to simulate and number of trials per size
S_VALS   = [5,   10,  25,  50, 100, 250, 500, 1000, 2000]
S_TRIALS = [500, 300, 200, 200, 100,  50,  30,   15,    5]

# Rate generation parameters
MU = 0                        # Mean of Gaussian noise added to log rates
SIGMA = 1                     # Std dev of Gaussian noise
MAX_REVERSE_RATE = 1.5        # Maximum multiplier for reverse rates (U[0, MAX_REVERSE_RATE])

# Small-world rewiring parameter
SW_BETA = 0.1                 # Rewiring probability for small-world topology

# Memory management: chunk batches to stay under this byte limit
# Prevents OOM on large system sizes
MAX_BATCH_BYTES = 200_000_000

# Random seed base; will be offset per experiment and size
SEED_BASE = 9999


def run_batch(S, N, max_jump, decay_alpha=0.0, sw_beta=None):
    """Generate rate matrices, compute stationary distributions, and compute metrics.

    Parameters
    ──────────
    S : int
        System size (number of states).
    N : int
        Number of trials.
    max_jump : int
        Maximum distance connectivity on ring.
    decay_alpha : float
        Distance decay parameter for rate generation.
    sw_beta : float or None
        If not None and > 0, apply small-world rewiring with this probability.

    Returns
    ───────
    metrics : ndarray of shape (5, N)
        Stacked metrics for each trial:
          [MEPS_EPR, NESS_EPR, UNIF_EPR, DKL(NESS||MEPS), DKL(MEPS||NESS)]
    """
    # Generate random rate matrices
    R = cyclic_generator(S=S, N=N, mu=MU, sigma=SIGMA,
                         max_jump=max_jump,
                         max_reverse_rate=MAX_REVERSE_RATE,
                         decay_alpha=decay_alpha)

    # Optionally apply small-world rewiring (preserves edge density)
    if sw_beta is not None and sw_beta > 0:
        R = small_world_rewire(R, beta=sw_beta)

    # Initialize CTMC and compute candidate distributions
    machine = MC(R=R)
    machine.verbose = False

    ness = machine.get_ness()  # Non-equilibrium steady state
    meps = machine.get_meps()  # Minimum EPR stationary distribution
    unif = machine.get_uniform()  # Uniform distribution

    # Compute EPR (entropy production rate) for each candidate
    meps_epr = np.atleast_1d(machine.get_epr(meps))
    ness_epr = np.atleast_1d(machine.get_epr(ness))
    unif_epr = np.atleast_1d(machine.get_epr(unif))

    # Ensure distributions are 2D: (N, S) for batch-wise KL divergence
    if ness.ndim == 1:
        ness = ness[np.newaxis, :]
        meps = meps[np.newaxis, :]

    # Compute divergences: KL(NESS || MEPS) and KL(MEPS || NESS)
    dkl_nm = np.array([machine.dkl(ness[i], meps[i]) for i in range(N)])
    dkl_mn = np.array([machine.dkl(meps[i], ness[i]) for i in range(N)])

    return np.array([meps_epr, ness_epr, unif_epr, dkl_nm, dkl_mn])


def run_config(S, N, max_jump, decay_alpha=0.0, sw_beta=None, seed=None):
    """Run N trials in chunks to respect memory limits; return (5, N) metrics.

    Parameters
    ──────────
    S : int
        System size.
    N : int
        Total number of trials.
    max_jump : int
        Maximum distance connectivity.
    decay_alpha : float
        Distance decay parameter.
    sw_beta : float or None
        Small-world rewiring probability.
    seed : int or None
        Random seed for reproducibility.

    Returns
    ───────
    metrics : ndarray of shape (5, N)
        Concatenated metrics from all trials.

    Notes
    ─────
    Automatically chunks trials to stay under MAX_BATCH_BYTES memory limit.
    This is necessary for large S where O(S^2) matrices become expensive.
    """
    if seed is not None:
        np.random.seed(seed)

    # Compute how many trials fit in memory: each rate matrix is S x S float64
    bytes_per_matrix = S * S * 8
    trials_per_chunk = max(1, int(MAX_BATCH_BYTES / bytes_per_matrix))

    # If all trials fit in one batch, run directly
    if trials_per_chunk >= N:
        return run_batch(S, N, max_jump, decay_alpha, sw_beta)

    # Otherwise chunk and accumulate results
    all_results = []
    remaining = N
    while remaining > 0:
        # Compute chunk size for this iteration
        k = min(trials_per_chunk, remaining)
        r = run_batch(S, k, max_jump, decay_alpha, sw_beta)
        all_results.append(r)
        remaining -= k
        gc.collect()  # Force garbage collection to free rate matrices

    return np.concatenate(all_results, axis=1)


def main():
    """Parse command-line arguments and dispatch to the requested experiment."""
    if len(sys.argv) < 3:
        print("Usage: python run_cyclic_appendix.py <experiment> <param>")
        print()
        print("Experiments:")
        print("  flat_ring <edge_frac>    - Flat ring at given edge density")
        print("  flat_ring_sw <edge_frac> - Flat ring with small-world rewiring")
        print("  decay_sweep <alpha>      - Full ring with distance decay parameter")
        sys.exit(1)

    experiment = sys.argv[1]
    param = sys.argv[2]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if experiment == "flat_ring":
        """
        Flat ring with no distance decay.

        Varies max_jump to match the specified edge density across system sizes,
        but keeps decay_alpha=0 (all distances equally likely).
        Tests MEPS minimization on rings without distance bias.
        """
        edge_frac = float(param)
        out_dir = os.path.join(base_dir, 'data_cyclic_flat')
        os.makedirs(out_dir, exist_ok=True)
        label = f"frac{int(edge_frac*100):03d}"

        print(f"\n{'='*60}")
        print(f"  Flat ring: edge_frac={edge_frac}, decay_alpha=0")
        print(f"{'='*60}")

        output = {}
        for i, (s, n) in enumerate(zip(S_VALS, S_TRIALS)):
            # Scale max_jump to maintain edge density across system sizes
            mj = max(1, int(edge_frac * (s - 1) / 2))
            ed = 2 * mj / (s - 1) if s > 1 else 1.0  # Effective edge density

            t0 = time.time()
            print(f"  S={s:>5}, N={n:>4}, max_jump={mj:>4} (ed={ed:.3f}) ...", end=" ", flush=True)
            results = run_config(s, n, mj, decay_alpha=0.0,
                                  seed=SEED_BASE + i + int(edge_frac * 1000))
            output[f"{s:05}"] = results
            print(f"done ({time.time()-t0:.1f}s)")

        # Save results: collect system sizes and trial counts that were computed
        s_used = [s for s in S_VALS if f"{s:05}" in output]
        t_used = [S_TRIALS[S_VALS.index(s)] for s in s_used]
        np.savez(os.path.join(out_dir, f"{label}.npz"),
                 s_values=np.array(s_used), s_trials=np.array(t_used), **output)
        print(f"  Saved → {out_dir}/{label}.npz")

    elif experiment == "flat_ring_sw":
        """
        Flat ring with small-world rewiring.

        Generates flat rings at the specified edge density, then applies
        small-world rewiring (adds shortcuts with probability SW_BETA).
        Tests robustness of MEPS minimization to topological changes.
        """
        edge_frac = float(param)
        out_dir = os.path.join(base_dir, 'data_cyclic_flat_sw')
        os.makedirs(out_dir, exist_ok=True)
        label = f"frac{int(edge_frac*100):03d}"

        # Skip large systems for dense SW: eigendecomposition and MEPS optimization
        # become prohibitively slow on dense matrices
        max_s_sw = 1000 if edge_frac >= 0.50 else 2000

        print(f"\n{'='*60}")
        print(f"  Flat ring + SW(beta={SW_BETA}): edge_frac={edge_frac}, decay_alpha=0")
        print(f"{'='*60}")

        output = {}
        for i, (s, n) in enumerate(zip(S_VALS, S_TRIALS)):
            if s > max_s_sw:
                print(f"  S={s:>5} skipped (too slow)")
                continue

            # Scale max_jump to maintain edge density
            mj = max(1, int(edge_frac * (s - 1) / 2))
            ed = 2 * mj / (s - 1) if s > 1 else 1.0

            t0 = time.time()
            print(f"  S={s:>5}, N={n:>4}, max_jump={mj:>4} (ed={ed:.3f}, beta={SW_BETA}) ...",
                  end=" ", flush=True)
            results = run_config(s, n, mj, decay_alpha=0.0, sw_beta=SW_BETA,
                                  seed=SEED_BASE + i + int(edge_frac * 1000) + 500)
            output[f"{s:05}"] = results
            print(f"done ({time.time()-t0:.1f}s)")

        # Save results
        s_used = [s for s in S_VALS if f"{s:05}" in output]
        t_used = [S_TRIALS[S_VALS.index(s)] for s in s_used]
        np.savez(os.path.join(out_dir, f"{label}.npz"),
                 s_values=np.array(s_used), s_trials=np.array(t_used), **output)
        print(f"  Saved → {out_dir}/{label}.npz")

    elif experiment == "decay_sweep":
        """
        Full ring with distance-dependent decay parameter sweep.

        Generates complete ring graphs (max_jump = S//2) where transition rates
        decay exponentially with distance: rate ~ exp(-alpha * dist).
        Sweeps alpha to show the transition from MEPS minimization holding to
        breaking down.
        """
        alpha = float(param)
        out_dir = os.path.join(base_dir, 'data_cyclic_decay')
        os.makedirs(out_dir, exist_ok=True)
        label = f"alpha{alpha:.2f}".replace('.', 'p')

        # Skip large systems for non-zero decay: eigendecomposition becomes slow
        # on full (dense) matrices. Alpha=0 (flat) is slightly faster.
        max_s = 2000 if alpha == 0 else 1000

        print(f"\n{'='*60}")
        print(f"  Decay sweep: alpha={alpha}, full ring (max_jump=S//2)")
        print(f"{'='*60}")

        output = {}
        for i, (s, n) in enumerate(zip(S_VALS, S_TRIALS)):
            if s > max_s:
                print(f"  S={s:>5} skipped (dense matrix)")
                continue

            # Full ring connectivity
            mj = s // 2
            t0 = time.time()
            print(f"  S={s:>5}, N={n:>4}, max_jump={mj:>4}, alpha={alpha} ...", end=" ", flush=True)
            results = run_config(s, n, mj, decay_alpha=alpha,
                                  seed=SEED_BASE + i + int(alpha * 1000) + 2000)
            output[f"{s:05}"] = results
            print(f"done ({time.time()-t0:.1f}s)")

        # Save results
        s_used = [s for s in S_VALS if f"{s:05}" in output]
        t_used = [S_TRIALS[S_VALS.index(s)] for s in s_used]
        np.savez(os.path.join(out_dir, f"{label}.npz"),
                 s_values=np.array(s_used), s_trials=np.array(t_used), **output)
        print(f"  Saved → {out_dir}/{label}.npz")


if __name__ == "__main__":
    main()
