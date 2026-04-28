"""Combined Figure 6: excess EPR ratio (top row) + D_KL (bottom row).

Single figure with one column per pump strength. Top row is the
NESS/MEPS-1 excess EPR ratio (as in the paper); bottom row is
D_KL(NESS || MEPS). Three lines per panel, one per pump ratio (5/20/80%).

Toggles:
  --log-stats   mean +/- std of log(value) on a linear y-axis
  --drop-db     drop detailed-balance trials instead of clamping to floor

Usage:
    python fig6_combined.py [--data-dir ...] [--out ...] [--log-stats] [--drop-db]
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

ROW_MEPS, ROW_NESS = 0, 1
ROW_DKL_NM = 3  # D_KL(NESS || MEPS)
DB_THRESHOLD = 1e-15
DB_FLOOR = 1e-16
RATIO_OUTLIER_ABS = 100


def filtered_ratio(meps, ness, drop_db=False):
    db_mask = (meps < DB_THRESHOLD) | (ness < DB_THRESHOLD)
    if drop_db:
        meps = meps[~db_mask]
        ness = ness[~db_mask]
        if meps.size == 0:
            return np.empty(0)
    else:
        meps = np.where(db_mask, DB_FLOOR, meps)
        ness = np.where(db_mask, DB_FLOOR, ness)
    ratio = ness / meps - 1
    return ratio[np.abs(ratio) < RATIO_OUTLIER_ABS]


def filtered_dkl(meps, ness, dkl, drop_db=False):
    db_mask = (meps < DB_THRESHOLD) | (ness < DB_THRESHOLD)
    if drop_db:
        return dkl[~db_mask]
    return np.where(db_mask, DB_FLOOR, dkl)


def mean_std(vals):
    if vals.size == 0:
        return np.nan, np.nan
    m = np.nanmean(vals)
    s = np.nanstd(vals, ddof=1) if vals.size > 1 else 0.0
    return m, s


def mean_std_log(vals):
    v = vals[vals > 0]
    if v.size == 0:
        return np.nan, np.nan
    lv = np.log(v)
    m = lv.mean()
    s = lv.std(ddof=1) if lv.size > 1 else 0.0
    return m, s


def load(data_dir, log_stats=False, drop_db=False):
    """Return {strength: {ratio_pct: {'s': s, 'ratio': (m,s), 'dkl': (m,s)}}}."""
    stat_fn = mean_std_log if log_stats else mean_std
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, '*', 'ratio_*.npz'))):
        strength_raw = int(os.path.basename(os.path.dirname(f)))
        strength = 0.25 if strength_raw == 0 else float(strength_raw)
        m = re.search(r'ratio_(\d+)', os.path.basename(f))
        if not m:
            continue
        ratio_pct = int(m.group(1))

        d = np.load(f)
        s_values = np.asarray(d['s_values'])
        r_mean = np.full(len(s_values), np.nan)
        r_std = np.full(len(s_values), np.nan)
        k_mean = np.full(len(s_values), np.nan)
        k_std = np.full(len(s_values), np.nan)

        for i, s in enumerate(s_values):
            key = f'{s:05d}'
            if key not in d.files:
                continue
            arr = d[key]
            if arr.shape[0] < 2:
                continue
            meps, ness = arr[ROW_MEPS], arr[ROW_NESS]
            r = filtered_ratio(meps, ness, drop_db=drop_db)
            r_mean[i], r_std[i] = stat_fn(r)
            if arr.shape[0] >= 5:
                dkl = filtered_dkl(meps, ness, arr[ROW_DKL_NM], drop_db=drop_db)
                k_mean[i], k_std[i] = stat_fn(dkl)

        out.setdefault(strength, {})[ratio_pct] = {
            's': s_values,
            'ratio': (r_mean, r_std),
            'dkl': (k_mean, k_std),
        }
    return out


def plot(data, out_path, log_stats=False, title_suffix=''):
    strengths = sorted(data.keys())
    n = len(strengths)
    if n == 0:
        print('No data to plot.')
        return

    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7.2),
                             sharex='col', sharey='row')
    if n == 1:
        axes = axes.reshape(2, 1)

    ratio_colors = {5: '#e69f00', 20: '#cc3311', 80: '#663300'}

    for col, strength in enumerate(strengths):
        for ratio_pct in sorted(data[strength].keys()):
            rec = data[strength][ratio_pct]
            s_values = rec['s']
            r_mean, r_std = rec['ratio']
            k_mean, k_std = rec['dkl']
            color = ratio_colors.get(ratio_pct, None)

            v = np.isfinite(r_mean) if log_stats else (np.isfinite(r_mean) & (r_mean > 0))
            axes[0, col].errorbar(s_values[v], r_mean[v], yerr=r_std[v],
                                  fmt='o-', markersize=3, capsize=2,
                                  color=color, label=f'pumps = {ratio_pct}%')
            v = np.isfinite(k_mean) if log_stats else (np.isfinite(k_mean) & (k_mean > 0))
            axes[1, col].errorbar(s_values[v], k_mean[v], yerr=k_std[v],
                                  fmt='o-', markersize=3, capsize=2,
                                  color=color, label=f'pumps = {ratio_pct}%')

        pct = int(round(strength * 100))
        axes[0, col].set_title(f'pump strength {pct}%')
        for row in range(2):
            axes[row, col].set_xscale('log')
            if not log_stats:
                axes[row, col].set_yscale('log')
            axes[row, col].grid(True, which='both', alpha=0.25)
        axes[1, col].set_xlabel('number of states')

    if log_stats:
        axes[0, 0].set_ylabel(r'$\langle \log(\sigma_{NESS}/\sigma_{MEPS} - 1) \rangle$')
        axes[1, 0].set_ylabel(r'$\langle \log D_{KL}(\pi_{NESS} \,\Vert\, m_R) \rangle$')
    else:
        axes[0, 0].set_ylabel(r'$\sigma_{NESS}/\sigma_{MEPS} - 1$')
        axes[1, 0].set_ylabel(r'$D_{KL}(\pi_{NESS} \,\Vert\, m_R)$')
    axes[0, -1].legend(fontsize=8, loc='best')

    if title_suffix:
        fig.suptitle(title_suffix)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f'saved {out_path}')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=os.path.join(here, 'data_fig6'))
    p.add_argument('--out', default=os.path.join(here, 'plots', 'fig6_combined.png'))
    p.add_argument('--title', default='')
    p.add_argument('--log-stats', action='store_true',
                   help='Use mean +/- std of log(value) on a linear y-axis')
    p.add_argument('--drop-db', action='store_true',
                   help='Drop detailed-balance trials instead of clamping to floor')
    args = p.parse_args()

    data = load(args.data_dir, log_stats=args.log_stats, drop_db=args.drop_db)
    if not data:
        print(f'No data found under {args.data_dir}')
        return
    for strength in sorted(data):
        ratios = sorted(data[strength].keys())
        print(f'  strength={strength}: ratios={ratios}')
    plot(data, args.out, log_stats=args.log_stats, title_suffix=args.title)


if __name__ == '__main__':
    main()
