"""Figure 5 reproduction — scatter plots of scaled excess EPR or D_KL.

One panel per system size N in {3, 9, 27, 81, 243, 729}, three colored series:
  - blue  = NESS
  - orange= random Dirichlet
  - green = uniform

Two stat modes:
  default (excess): y = (sigma_state - sigma_MEPS) / sigma_MEPS
  --dkl:            y = D_KL(p_state || m_R)

Pump parameters match the paper: pump_strength = 4 (400%), pump_percent = 0.8.

Usage:
    python fig5_combined.py                  # generate data if missing, plot excess
    python fig5_combined.py --dkl            # D_KL version (uses same data)
    python fig5_combined.py --regen          # force regeneration
"""
import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SIZES = [3, 9, 27, 81, 243, 729]
TRIALS = [5000, 4000, 2500, 1500, 1000, 500]
PUMP_STRENGTH = 4
PUMP_PERCENT = 0.8

STATE_COLORS = {'ness': 'tab:blue', 'rand': 'tab:orange', 'unif': 'tab:green'}
DB_FLOOR = 1e-16
DB_THRESHOLD = 1e-15  # below this, NESS/MEPS are at the estimator floor — treat as DB


def generate(out_path, use_jax=True):
    from ctmc import ContinuousTimeMarkovChain as MC, arrhenius_pump_generator

    print(f'Generating Fig 5 data → {out_path}')
    print(f'  pump_strength={PUMP_STRENGTH}  pump_percent={PUMP_PERCENT}')
    print(f'  sizes={SIZES}  trials={TRIALS}')

    saved = {'sizes': np.asarray(SIZES), 'trials': np.asarray(TRIALS)}
    for s, n in zip(SIZES, TRIALS):
        n_pumps = max(1, int(PUMP_PERCENT * (s * s - s) / 2))
        t0 = time.perf_counter()
        machine = MC(S=s, N=n, generator=arrhenius_pump_generator,
                     pump_strength=PUMP_STRENGTH, n_pumps=n_pumps)
        ness = machine.get_ness()
        try:
            meps = machine.get_meps(method='jax' if use_jax else 'euler', state=ness)
        except Exception as e:
            print(f'  S={s}: jax MEPS failed ({e}); falling back to euler')
            meps = machine.get_meps(method='euler', state=ness)
        unif = machine.get_uniform()
        rand = machine.get_random_state()

        epr = {
            'ness': machine.get_epr(ness),
            'meps': machine.get_epr(meps),
            'unif': machine.get_epr(unif),
            'rand': machine.get_epr(rand),
        }
        dkl = {
            'ness': machine.dkl(ness, meps),
            'unif': machine.dkl(unif, meps),
            'rand': machine.dkl(rand, meps),
        }

        for k, v in epr.items():
            saved[f'{s:05d}_epr_{k}'] = np.asarray(v)
        for k, v in dkl.items():
            saved[f'{s:05d}_dkl_{k}'] = np.asarray(v)

        dt = time.perf_counter() - t0
        med = np.nanmedian(epr['ness'] / np.maximum(epr['meps'], DB_FLOOR) - 1)
        print(f'  S={s:>4d} N={n:>4d}  {dt:6.2f}s   median(NESS excess) = {med:.3g}')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **saved)
    print(f'saved {out_path}')


def plot(npz_path, out_path, keep_db=False):
    d = np.load(npz_path)
    sizes = list(d['sizes'])
    n = len(sizes)
    fig, axes = plt.subplots(2, n, figsize=(2.7 * n, 7.2),
                             sharey='row', sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, s in enumerate(sizes):
        meps_epr = d[f'{s:05d}_epr_meps']
        ness_epr = d[f'{s:05d}_epr_ness']
        # Trials where MEPS or NESS sits at the estimator floor are detailed-
        # balance machines our algorithm can't distinguish from zero. Drop them.
        db = (meps_epr < DB_THRESHOLD) | (ness_epr < DB_THRESHOLD)
        keep_trial = np.ones_like(meps_epr, dtype=bool) if keep_db else ~db

        ax_top, ax_bot = axes[0, col], axes[1, col]
        for st in ('ness', 'rand', 'unif'):
            excess = (d[f'{s:05d}_epr_{st}'] - meps_epr) / np.maximum(meps_epr, DB_FLOOR)
            dkl = d[f'{s:05d}_dkl_{st}']
            base = keep_trial & np.isfinite(meps_epr) & (meps_epr > 0)
            v = base & np.isfinite(excess) & (excess > 0)
            ax_top.scatter(meps_epr[v], excess[v], c=STATE_COLORS[st],
                           label=st, alpha=0.18, s=12, edgecolors='none')
            v = base & np.isfinite(dkl) & (dkl > 0)
            ax_bot.scatter(meps_epr[v], dkl[v], c=STATE_COLORS[st],
                           label=st, alpha=0.18, s=12, edgecolors='none')

        ax_top.set_title(f'N = {s}')
        for ax in (ax_top, ax_bot):
            ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.2)
        ax_bot.set_xlabel(r'$\sigma_{\mathrm{MEPS}} / k_B$')

    axes[0, 0].set_ylabel(r'$\sigma / \sigma_{\mathrm{MEPS}} - 1$')
    axes[1, 0].set_ylabel(r'$D_{KL}(p \,\Vert\, m_R)$')
    axes[0, 0].set_ylim(1e-2, 1e2)
    axes[1, 0].set_ylim(1e-4, 10)
    axes[0, -1].legend(markerscale=3, framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f'saved {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.path.join(HERE, 'data_fig5', 'fig5_data.npz'))
    p.add_argument('--out', default=None,
                   help='Output PNG (default plots/fig5_combined.png)')
    p.add_argument('--regen', action='store_true', help='Force regeneration of data')
    p.add_argument('--no-jax', action='store_true', help='Use euler MEPS solver')
    p.add_argument('--keep-db', action='store_true',
                   help='Keep detailed-balance trials (default drops them)')
    args = p.parse_args()

    if args.regen or not os.path.exists(args.data):
        generate(args.data, use_jax=not args.no_jax)
    else:
        print(f'Reusing existing data {args.data} (pass --regen to refresh)')

    suffix = '_keepdb' if args.keep_db else ''
    out = args.out or os.path.join(HERE, 'plots', f'fig5_combined{suffix}.png')
    plot(args.data, out, keep_db=args.keep_db)


if __name__ == '__main__':
    main()
