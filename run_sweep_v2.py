"""Regenerate the main sweep data with D_KL included (v2).

This replaces ctmc_parallel.py. Instead of MPI, each (strength, ratio)
pair is one independent job, suitable for SLURM array submission or
manual invocation.

Data format (backward-compatible + extended):
    Row 0: MEPS_EPR
    Row 1: NESS_EPR
    Row 2: UNIF_EPR
    Row 3: D_KL(NESS || MEPS)
    Row 4: D_KL(MEPS || NESS)

Rows 0-2 are identical to the old 3-row format. The notebook's
process_data_dictionary() only indexes d[0], d[1], d[2], so the extra
rows are invisible to existing code.

Directory layout matches what plot_sweeps.ipynb expects:
    <out_dir>/<strength:04d>/ratio_<ratio_pct:03d>.npz

Incremental append:
    If the output file already exists, new trials are concatenated along
    the trial axis for each S value. This lets you accumulate large-S
    results one trial at a time:

        python run_sweep_v2.py 100 20 --s-values 16000 --n-trials 1
        python run_sweep_v2.py 100 20 --s-values 16000 --n-trials 1
        # ... repeat until you have enough trials

Usage:
    # Run full sweep for one (strength, ratio) pair:
    python run_sweep_v2.py <pump_strength> <pump_ratio_pct>

    # Run specific S values only:
    python run_sweep_v2.py 100 20 --s-values 500 1000 2000

    # Override trial count:
    python run_sweep_v2.py 100 20 --s-values 16000 --n-trials 1

    # Custom output directory:
    python run_sweep_v2.py 100 20 --out-dir plot_data_v2

    # Use JAX MEPS optimizer (if available):
    python run_sweep_v2.py 100 20 --use-jax

Energy distribution note:
    gen_args=[0, 1] gives E_s ~ U(0, 1) and pump forces F ~ U(0, α).
    This matches the original ctmc_parallel.py and all existing data.
    The manuscript should specify F ∈ [0, α] accordingly.
"""
import sys
import os
import argparse
import time
import gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctmc import (
    ContinuousTimeMarkovChain as MC,
    arrhenius_pump_generator,
    HAS_JAX,
)

# ── Default sweep parameters (matching the existing data) ──────────
DEFAULT_S_VALS   = [5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000]
DEFAULT_S_TRIALS = [10000, 5000, 2000, 1000, 1000, 500, 250, 125, 25, 15, 6, 4, 2]

# Map S -> default N trials
S_TRIAL_MAP = dict(zip(DEFAULT_S_VALS, DEFAULT_S_TRIALS))

N_ROWS = 5  # MEPS_EPR, NESS_EPR, UNIF_EPR, DKL_NM, DKL_MN


def run_batch(S, N, pump_strength, pump_ratio, gen_args, use_jax=False,
              jax_restarts=5, euler_iter=5000, euler_lr=0.01, seed=None):
    """Run N trials at system size S and return 5-row EPR + D_KL data.

    Parameters
    ----------
    S : int
        Number of states.
    N : int
        Number of independent trials (batch size).
    pump_strength : float
        Pump strength parameter α.
    pump_ratio : float
        Fraction of transitions to pump (0 to 1).
    gen_args : list
        Arguments for np.random.uniform (energy distribution bounds).
    use_jax : bool
        Use JAX MEPS optimizer if available.
    jax_restarts : int
        Number of restarts for JAX optimizer.
    euler_iter : int
        Number of Euler iterations for MEPS.
    euler_lr : float
        Learning rate for Euler MEPS.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    data : ndarray, shape (5, N)
        [MEPS_EPR, NESS_EPR, UNIF_EPR, DKL(NESS||MEPS), DKL(MEPS||NESS)]
    """
    if seed is not None:
        np.random.seed(seed)

    n_pumps = max(1, int(pump_ratio * (S**2 - S)))

    R = arrhenius_pump_generator(
        S=S, N=N,
        n_pumps=n_pumps,
        pump_strength=pump_strength,
        gen_args=list(gen_args),
    )

    machine = MC(R=R)
    machine.verbose = False

    # Compute distributions
    ness = machine.get_ness()

    if use_jax and HAS_JAX:
        meps = machine.get_meps_jax(num_restarts=jax_restarts)
    else:
        meps = machine.get_meps(n_iter=euler_iter, lr=euler_lr)

    unif = machine.get_uniform()

    # Compute EPR
    meps_epr = np.atleast_1d(machine.get_epr(meps))
    ness_epr = np.atleast_1d(machine.get_epr(ness))
    unif_epr = np.atleast_1d(machine.get_epr(unif))

    # Compute D_KL
    dkl_nm = np.atleast_1d(machine.dkl(ness, meps))   # D_KL(NESS || MEPS)
    dkl_mn = np.atleast_1d(machine.dkl(meps, ness))   # D_KL(MEPS || NESS)

    data = np.array([meps_epr, ness_epr, unif_epr, dkl_nm, dkl_mn])
    assert data.shape == (N_ROWS, N), f"Expected ({N_ROWS}, {N}), got {data.shape}"
    return data


def load_existing(path):
    """Load existing .npz file and return its contents as a mutable dict.

    Returns
    -------
    contents : dict or None
        Keys include 's_values', 's_trials', and per-S data arrays.
        None if file doesn't exist.
    """
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


def merge_results(existing, s_values, new_data):
    """Merge new trial data into existing results, appending along trial axis.

    Parameters
    ----------
    existing : dict or None
        Existing data from load_existing(), or None.
    s_values : list of int
        System sizes in this run.
    new_data : dict
        Maps '{s:05d}' -> ndarray of shape (5, N_new).

    Returns
    -------
    merged : dict
        Ready to pass to np.savez().
    """
    if existing is None:
        # Fresh start
        all_s = sorted(s_values)
        trials = []
        result = {'s_values': np.array(all_s)}
        for s in all_s:
            key = f'{s:05d}'
            result[key] = new_data[key]
            trials.append(new_data[key].shape[1])
        result['s_trials'] = np.array(trials)
        return result

    # Merge into existing
    old_s = set(existing['s_values'].tolist())
    new_s = set(s_values)
    all_s = sorted(old_s | new_s)

    result = {'s_values': np.array(all_s)}
    trials = []

    for s in all_s:
        key = f'{s:05d}'
        old_arr = existing.get(key)
        new_arr = new_data.get(key)

        if old_arr is not None and new_arr is not None:
            # Ensure compatible row count
            if old_arr.shape[0] == 3 and new_arr.shape[0] == 5:
                # Upgrade old 3-row to 5-row with NaN padding
                pad = np.full((2, old_arr.shape[1]), np.nan)
                old_arr = np.vstack([old_arr, pad])
            elif old_arr.shape[0] != new_arr.shape[0]:
                raise ValueError(
                    f"Row mismatch at S={s}: existing has {old_arr.shape[0]} rows, "
                    f"new has {new_arr.shape[0]} rows"
                )
            merged = np.hstack([old_arr, new_arr])
        elif old_arr is not None:
            merged = old_arr
        else:
            merged = new_arr

        result[key] = merged
        trials.append(merged.shape[1])

    result['s_trials'] = np.array(trials)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate sweep data with D_KL (v2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('pump_strength', type=float,
                        help='Pump strength α (e.g. 100 for 100%%)')
    parser.add_argument('pump_ratio_pct', type=float,
                        help='Pump ratio as percent (e.g. 20 for 20%%)')
    parser.add_argument('--s-values', type=int, nargs='+', default=None,
                        help='System sizes to run (default: all 13 sizes)')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Override trial count for ALL S values')
    parser.add_argument('--out-dir', type=str, default='sample_notebooks/plot_data_v2',
                        help='Output directory (default: sample_notebooks/plot_data_v2)')
    parser.add_argument('--gen-args', type=float, nargs=2, default=[0, 1],
                        help='Energy distribution bounds (default: 0 1 matching '
                             'original ctmc_parallel.py; use -1 1 to match paper)')
    parser.add_argument('--use-jax', action='store_true',
                        help='Use JAX MEPS optimizer if available')
    parser.add_argument('--jax-restarts', type=int, default=5,
                        help='Number of JAX optimizer restarts (default: 5)')
    parser.add_argument('--euler-iter', type=int, default=5000,
                        help='Euler MEPS iterations (default: 5000)')
    parser.add_argument('--euler-lr', type=float, default=0.01,
                        help='Euler MEPS learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: based on S)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without running')
    args = parser.parse_args()

    strength = args.pump_strength
    ratio_pct = args.pump_ratio_pct
    ratio = ratio_pct / 100.0

    s_values = args.s_values if args.s_values else DEFAULT_S_VALS

    # Output path
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_dir)
    strength_dir = os.path.join(base, f'{int(strength):04d}')
    os.makedirs(strength_dir, exist_ok=True)
    out_path = os.path.join(strength_dir, f'ratio_{int(ratio_pct):03d}.npz')

    # Header
    print("=" * 70)
    print(f"Sweep v2: strength={strength}, ratio={ratio_pct}%")
    print(f"Energy distribution: U({args.gen_args[0]}, {args.gen_args[1]})")
    print(f"MEPS optimizer: {'JAX' if (args.use_jax and HAS_JAX) else 'Euler'}")
    if args.use_jax and not HAS_JAX:
        print("  (JAX requested but not available, falling back to Euler)")
    print(f"Output: {out_path}")
    print(f"S values: {s_values}")
    print("=" * 70)

    # Load existing data for incremental append
    existing = load_existing(out_path)
    if existing is not None:
        print(f"Found existing file with S={existing['s_values'].tolist()}")
        for s in existing['s_values']:
            key = f'{s:05d}'
            if key in existing:
                print(f"  S={s}: {existing[key].shape[1]} existing trials "
                      f"({existing[key].shape[0]} rows)")

    if args.dry_run:
        for S in s_values:
            N = args.n_trials if args.n_trials else S_TRIAL_MAP.get(S, 10)
            print(f"  Would run: S={S}, N={N}")
        return

    # Run
    new_data = {}
    total_t0 = time.time()

    for S in s_values:
        N = args.n_trials if args.n_trials else S_TRIAL_MAP.get(S, 10)

        seed = args.seed if args.seed else (54321 + S)

        t0 = time.time()
        print(f"\n  S={S:>6d}  N={N:>5d}  ", end='', flush=True)

        try:
            data = run_batch(
                S=S, N=N,
                pump_strength=strength,
                pump_ratio=ratio,
                gen_args=args.gen_args,
                use_jax=args.use_jax,
                jax_restarts=args.jax_restarts,
                euler_iter=args.euler_iter,
                euler_lr=args.euler_lr,
                seed=seed,
            )
            key = f'{S:05d}'
            new_data[key] = data

            dt = time.time() - t0
            med_excess = np.median(data[1] / data[0] - 1)
            med_dkl = np.median(data[3])
            print(f"med_excess={med_excess:.4f}  med_DKL={med_dkl:.2e}  "
                  f"t={dt:.1f}s", flush=True)

        except Exception as e:
            dt = time.time() - t0
            print(f"FAILED after {dt:.1f}s: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

        gc.collect()

    # Merge and save
    if new_data:
        merged = merge_results(existing, s_values, new_data)
        np.savez(out_path, **merged)

        total_dt = time.time() - total_t0
        print(f"\n{'=' * 70}")
        print(f"Saved: {out_path}")
        print(f"Total time: {total_dt:.1f}s")
        print(f"S values: {merged['s_values'].tolist()}")
        print(f"Trials:   {merged['s_trials'].tolist()}")
        print(f"Rows per array: {N_ROWS}")
    else:
        print("\nNo successful runs — nothing saved.")


if __name__ == '__main__':
    main()
