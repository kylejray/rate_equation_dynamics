"""Regenerate fixed-degree sweep figure with equilibrium filtering.

Produces a 4-panel figure showing how excess EPR, raw NESS/MEPS EPR,
and connectivity repair fraction vary with system size S at several
fixed average degrees k.

Equilibrium filter
------------------
Trials with NESS EPR < EQ_THRESHOLD (default 1e-10) are discarded as
effectively equilibrium systems.  See plot_common.py for details.

Figures produced
----------------
  - 1 fixed-degree sweep figure (4 panels)

Usage
-----
    python plot_fixed_degree.py mean      # mean center line + IQR bands
    python plot_fixed_degree.py median    # median center line + IQR bands
    python plot_fixed_degree.py           # defaults to mean
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from plot_common import (
    BASE, EQ_THRESHOLD, MIN_SYSTEM_SIZE, MIN_TRIALS,
    _filter_log, print_filter_diagnostics, resolve_center_fn,
)


# ═════════════════════════════════════════════════════════════════════
# Data loader
# ═════════════════════════════════════════════════════════════════════

def load_fixed_degree(path, center_fn):
    """Load fixed-degree sweep data and compute summary statistics.

    Parameters
    ----------
    path : str
        Path to .npz file with shape (3, N) arrays keyed by zero-padded
        system size.  Rows: [MEPS_EPR, NESS_EPR, UNIF_EPR].
        Optional '{S}_repair' keys give per-trial repair edge fractions.
    center_fn : callable
        Aggregation function for center line (np.mean or np.median).

    Returns
    -------
    dict
        Keys: 's', 'epr_center', 'epr_q25', 'epr_q75',
              'meps_center', 'meps_q25', 'meps_q75',
              'ness_center', 'ness_q25', 'ness_q75', 'disconn_frac'.
    """
    d = np.load(path)
    s_vals = d['s_values']
    out = {k: [] for k in [
        's',
        'epr_center', 'epr_q25', 'epr_q75',
        'meps_center', 'meps_q25', 'meps_q75',
        'ness_center', 'ness_q25', 'ness_q75',
        'disconn_frac',
    ]}
    for s in s_vals:
        if s < MIN_SYSTEM_SIZE:
            continue
        key = f'{s:05d}'
        if key not in d:
            continue
        arr = d[key]  # (3, N): [meps_epr, ness_epr, unif_epr]
        repair = d.get(f'{key}_repair', np.zeros(arr.shape[1]))
        ness_epr = arr[1]

        # Equilibrium filter
        eq_mask = ness_epr > EQ_THRESHOLD
        n_total = len(ness_epr)
        n_kept = eq_mask.sum()
        _filter_log.append((os.path.basename(path), s, n_total, n_total - n_kept))

        excess = arr[1][eq_mask] / arr[0][eq_mask] - 1
        mask = np.isfinite(excess) & (excess > 0)
        if mask.sum() < MIN_TRIALS:
            continue

        meps_vals = arr[0][eq_mask]
        ness_vals = arr[1][eq_mask]
        out['s'].append(s)
        out['epr_center'].append(center_fn(excess[mask]))
        out['epr_q25'].append(np.percentile(excess[mask], 25))
        out['epr_q75'].append(np.percentile(excess[mask], 75))
        out['meps_center'].append(center_fn(meps_vals[mask]))
        out['meps_q25'].append(np.percentile(meps_vals[mask], 25))
        out['meps_q75'].append(np.percentile(meps_vals[mask], 75))
        out['ness_center'].append(center_fn(ness_vals[mask]))
        out['ness_q25'].append(np.percentile(ness_vals[mask], 25))
        out['ness_q75'].append(np.percentile(ness_vals[mask], 75))
        out['disconn_frac'].append((repair > 0).mean() * 100)

    return {k: np.array(v) for k, v in out.items()}


# ═════════════════════════════════════════════════════════════════════
# Figure builder
# ═════════════════════════════════════════════════════════════════════

# Visual style for each degree line.
DEGREES = [4, 6, 8, 16, 64]
COLORS = {
    4: '#d62728',    # red
    6: '#ff7f0e',    # orange
    8: '#2ca02c',    # green
    16: '#1f77b4',   # blue
    64: '#9467bd',   # purple
    'dense': '#333333',
}
MARKERS = {4: 'v', 6: 'D', 8: 's', 16: 'o', 64: '^', 'dense': '*'}


def make_fixed_degree_figure(out_dir, center_fn):
    """4-panel figure: excess EPR, NESS EPR, MEPS EPR, and repair fraction vs S.

    Each line corresponds to a fixed average degree k, with a dense
    (fully connected) baseline for reference.
    """
    DATA_DIR = os.path.join(BASE, 'data_fixed_degree_v2')

    all_data = {}
    for deg in DEGREES:
        path = os.path.join(DATA_DIR, f'deg{deg:03d}.npz')
        if os.path.exists(path):
            all_data[deg] = load_fixed_degree(path, center_fn)
    dense_path = os.path.join(DATA_DIR, 'dense.npz')
    if os.path.exists(dense_path):
        all_data['dense'] = load_fixed_degree(dense_path, center_fn)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 5.5))
    plot_order = ['dense'] + DEGREES
    for deg in plot_order:
        if deg not in all_data:
            continue
        d = all_data[deg]
        label = 'Dense (100%)' if deg == 'dense' else f'$k = {deg}$'
        color = COLORS[deg]
        marker = MARKERS[deg]
        marker_size = 9 if deg == 'dense' else 7
        style = dict(marker=marker, color=color, label=label,
                     markersize=marker_size, linewidth=1.5, zorder=3)

        ax1.plot(d['s'], d['epr_center'], **style)
        ax1.fill_between(d['s'], d['epr_q25'], d['epr_q75'],
                         color=color, alpha=0.12)
        ax2.plot(d['s'], d['ness_center'], **style)
        ax2.fill_between(d['s'], d['ness_q25'], d['ness_q75'],
                         color=color, alpha=0.12)
        ax3.plot(d['s'], d['meps_center'], **style)
        ax3.fill_between(d['s'], d['meps_q25'], d['meps_q75'],
                         color=color, alpha=0.12)
        # Dense is always connected; only plot repair for sparse degrees
        if deg != 'dense':
            ax4.plot(d['s'], d['disconn_frac'], **style)

    for ax in [ax1, ax2, ax3]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of states $S$', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
    ax1.set_ylabel(
        r'$\sigma_{\mathrm{NESS}}/\sigma_{\mathrm{MEPS}} - 1$', fontsize=11)
    ax1.set_title('Excess EPR', fontsize=12)
    ax2.set_ylabel(r'$\sigma_{\mathrm{NESS}}$', fontsize=11)
    ax2.set_title('NESS Entropy Production', fontsize=12)
    ax3.set_ylabel(r'$\sigma_{\mathrm{MEPS}}$', fontsize=11)
    ax3.set_title('MEPS Entropy Production', fontsize=12)

    ax4.set_xscale('log')
    ax4.set_xlabel('Number of states $S$', fontsize=11)
    ax4.set_ylabel('% trials disconnected\nbefore repair', fontsize=11)
    ax4.set_title('Connectivity Repair Needed', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_ylim(-5, 105)  # 5% margin around [0, 100]

    fig.suptitle(
        r'Fixed Average Degree: Arrhenius pump '
        r'$E_s, E_{s \leftrightarrow s^\prime} \sim U(0,1)$, '
        r'$\alpha = 2$, pump fraction = 20%',
        fontsize=11, y=1.01, style='italic')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig_fixed_degree_v2.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved -> {out_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main(statistic='mean'):
    """Generate the fixed-degree sweep figure.

    Parameters
    ----------
    statistic : str
        'mean' or 'median' — determines center line.
    """
    center_fn = resolve_center_fn(statistic)

    out_dir = os.path.join(BASE, 'final_plots', statistic)
    os.makedirs(out_dir, exist_ok=True)

    make_fixed_degree_figure(out_dir, center_fn)

    print_filter_diagnostics(statistic)


if __name__ == "__main__":
    stat = sys.argv[1] if len(sys.argv) > 1 else 'mean'
    main(stat)
