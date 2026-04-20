"""Verify sweep data format and flag anomalies.

Checks:
  1. Directory structure matches expected layout
  2. Each .npz has correct keys and array shapes
  3. Flags MEPS > NESS violations
  4. Flags NaN/Inf values
  5. Reports trial counts vs targets
  6. Summarizes D_KL statistics (if present)

Usage:
    python verify_sweep_data.py [data_dir]
    python verify_sweep_data.py sample_notebooks/plot_data_v2
"""
import sys
import os
import numpy as np
from pathlib import Path

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'sample_notebooks', 'plot_data_v2')

TARGET_S_VALS   = [5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000]
TARGET_S_TRIALS = [10000, 5000, 2000, 1000, 1000, 500, 250, 125, 25, 15, 6, 4, 2]
TARGET_MAP = dict(zip(TARGET_S_VALS, TARGET_S_TRIALS))

EQ_THRESHOLD = 1e-10


def verify_file(path, verbose=True):
    """Verify a single .npz file. Returns dict of diagnostics."""
    result = {'path': str(path), 'ok': True, 'warnings': [], 'errors': []}

    try:
        d = np.load(path)
    except Exception as e:
        result['ok'] = False
        result['errors'].append(f"Can't load: {e}")
        return result

    # Check required keys
    if 's_values' not in d or 's_trials' not in d:
        result['errors'].append("Missing s_values or s_trials")
        result['ok'] = False
        return result

    s_values = d['s_values']
    s_trials = d['s_trials']

    for s, n_expected in zip(s_values, s_trials):
        key = f'{s:05d}'
        if key not in d:
            result['errors'].append(f"Missing key '{key}' for S={s}")
            result['ok'] = False
            continue

        arr = d[key]
        n_rows, n_trials = arr.shape

        # Check shape
        if n_rows not in (3, 5):
            result['errors'].append(f"S={s}: unexpected {n_rows} rows (want 3 or 5)")
        if n_rows == 3:
            result['warnings'].append(f"S={s}: old 3-row format (no D_KL)")

        # Check trial count
        if n_trials != n_expected:
            result['warnings'].append(
                f"S={s}: {n_trials} trials (s_trials says {n_expected})")

        target = TARGET_MAP.get(s)
        if target and n_trials < target:
            result['warnings'].append(
                f"S={s}: {n_trials}/{target} trials (below target)")

        # Check for NaN/Inf
        bad = ~np.isfinite(arr)
        if bad.any():
            n_bad = bad.sum()
            result['warnings'].append(f"S={s}: {n_bad} NaN/Inf values")

        # Check MEPS > NESS violations (only for non-equilibrium)
        meps_epr = arr[0]
        ness_epr = arr[1]
        mask = ness_epr > EQ_THRESHOLD
        if mask.sum() > 0:
            violations = (meps_epr[mask] > ness_epr[mask] * 1.001).sum()
            if violations > 0:
                result['warnings'].append(
                    f"S={s}: {violations}/{mask.sum()} MEPS > NESS violations")

        # D_KL checks (if 5-row)
        if n_rows >= 5:
            dkl_nm = arr[3]
            dkl_mn = arr[4]
            neg_dkl = ((dkl_nm < -1e-10) | (dkl_mn < -1e-10)).sum()
            if neg_dkl > 0:
                result['errors'].append(f"S={s}: {neg_dkl} negative D_KL values!")

    return result


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR

    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        sys.exit(1)

    print("=" * 80)
    print(f"Verifying sweep data in: {data_dir}")
    print("=" * 80)

    dirpath = Path(data_dir)
    total_files = 0
    total_warnings = 0
    total_errors = 0

    # Summary table
    print(f"\n{'strength':>8s}  {'ratio':>6s}  {'rows':>4s}  "
          f"{'S_count':>7s}  {'min_N':>5s}  {'max_N':>5s}  "
          f"{'warns':>5s}  {'errs':>4s}  {'status':>8s}")
    print("─" * 75)

    for strength_dir in sorted(dirpath.iterdir()):
        if not strength_dir.is_dir():
            continue

        for npz_file in sorted(strength_dir.glob('*.npz')):
            total_files += 1
            result = verify_file(npz_file)

            d = np.load(npz_file)
            s_vals = d['s_values']
            trials = d['s_trials']
            n_rows = 0
            for s in s_vals:
                key = f'{s:05d}'
                if key in d:
                    n_rows = d[key].shape[0]
                    break

            strength = strength_dir.name
            ratio = npz_file.stem.replace('ratio_', '')

            n_warn = len(result['warnings'])
            n_err = len(result['errors'])
            total_warnings += n_warn
            total_errors += n_err

            status = "OK" if result['ok'] and n_warn == 0 else (
                "WARN" if result['ok'] else "ERROR")

            print(f"{strength:>8s}  {ratio:>6s}  {n_rows:>4d}  "
                  f"{len(s_vals):>7d}  {min(trials):>5d}  {max(trials):>5d}  "
                  f"{n_warn:>5d}  {n_err:>4d}  {status:>8s}")

            if n_warn > 0 or n_err > 0:
                for w in result['warnings']:
                    print(f"    WARN: {w}")
                for e in result['errors']:
                    print(f"    ERR:  {e}")

    # Trial count summary
    print(f"\n{'=' * 80}")
    print("TRIAL COUNT SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n{'S':>8s}  {'target':>6s}  ", end='')

    # Collect actual counts per S across all files
    s_counts = {}
    for strength_dir in sorted(dirpath.iterdir()):
        if not strength_dir.is_dir():
            continue
        for npz_file in sorted(strength_dir.glob('*.npz')):
            d = np.load(npz_file)
            label = f"{strength_dir.name}/{npz_file.stem}"
            for s in d['s_values']:
                key = f'{s:05d}'
                if key in d:
                    if s not in s_counts:
                        s_counts[s] = {}
                    s_counts[s][label] = d[key].shape[1]

    # Print compact per-S summary
    for s in sorted(s_counts.keys()):
        counts = list(s_counts[s].values())
        target = TARGET_MAP.get(s, '?')
        print(f"\n  S={s:>6d}  target={target:>6}  "
              f"actual: min={min(counts)}, max={max(counts)}, "
              f"across {len(counts)} files")

    print(f"\n{'=' * 80}")
    print(f"Total: {total_files} files, {total_warnings} warnings, {total_errors} errors")
    if total_errors > 0:
        print("STATUS: ERRORS FOUND")
    elif total_warnings > 0:
        print("STATUS: WARNINGS (check above)")
    else:
        print("STATUS: ALL OK")


if __name__ == '__main__':
    main()
