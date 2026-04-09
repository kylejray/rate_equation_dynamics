"""Shared configuration and utilities for appendix plotting scripts.

Provides the equilibrium filter, minimum system size threshold, and
filtering diagnostics used by both plot_appendix.py and
plot_fixed_degree.py.
"""
import os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Equilibrium filter ────────────────────────────────────────────────
# Systems with NESS EPR below this threshold are treated as equilibrium
# artefacts and excluded from statistics.  The gap between float-noise
# (~1e-16) and the weakest genuine NESS (~1e-8) spans many orders of
# magnitude, so the exact value is not critical.
EQ_THRESHOLD = 1e-10

# Minimum system size to include in plots.  S < 10 has high equilibrium
# contamination (up to ~25% of trials) and unreliable statistics.
MIN_SYSTEM_SIZE = 10

# Minimum number of trials required to plot a data point.
MIN_TRIALS = 3

# Collect per-dataset filtering stats for the diagnostics report.
_filter_log = []


def print_filter_diagnostics(statistic_name):
    """Print a summary of how many trials were dropped by the equilibrium filter."""
    print(f"\n{'=' * 72}")
    print(f"Equilibrium filter: NESS EPR < {EQ_THRESHOLD:.0e}")
    print(f"{'=' * 72}")

    filtered_entries = [(f, s, n, nf)
                        for (f, s, n, nf) in _filter_log if nf > 0]
    if filtered_entries:
        print(f"{'File':<30s} {'S':>5s}  {'Total':>6s}  "
              f"{'Dropped':>7s}  {'%':>6s}")
        print('-' * 60)
        for f, s, n, nf in sorted(filtered_entries,
                                   key=lambda x: (-x[3] / x[2], x[1])):
            print(f"{f:<30s} {s:>5d}  {n:>6d}  "
                  f"{nf:>7d}  {100 * nf / n:>5.1f}%")
    else:
        print(f"No trials filtered at any system size "
              f"(S >= {MIN_SYSTEM_SIZE}).")

    total_trials = sum(n for _, _, n, _ in _filter_log)
    total_dropped = sum(nf for _, _, _, nf in _filter_log)
    if total_trials > 0:
        print(f"\nTotal: {total_dropped}/{total_trials} trials dropped "
              f"({100 * total_dropped / total_trials:.2f}%)")


def resolve_center_fn(statistic):
    """Return np.mean or np.median from a string argument.

    Parameters
    ----------
    statistic : str
        'mean' or 'median'.

    Returns
    -------
    callable
    """
    if statistic == 'mean':
        return np.mean
    elif statistic == 'median':
        return np.median
    else:
        raise ValueError(
            f"statistic must be 'mean' or 'median', got {statistic!r}")
