"""Regenerate appendix plots with equilibrium filtering.

Generates 10 figures using either mean or median center lines with
interquartile range (IQR) bands, saving to final_plots/<statistic>/.

Equilibrium filter
------------------
Trials with NESS EPR < EQ_THRESHOLD (default 1e-10) are discarded as
effectively equilibrium systems.  These arise when the random generator
produces near-canceling cycle forces, leaving the system at the
floating-point noise floor (~1e-16).  A clear gap of many orders of
magnitude separates these from the weakest genuinely driven systems
(~1e-8), so the filter is insensitive to the exact threshold.

Figures produced
----------------
  - 4 Arrhenius appendix figures (ER/SW x pos/neg pump)
  - 2 flat ring figures (with/without small-world rewiring)
  - 4 distance-decay sweep figures (full/sparse x with/without SW)

Usage
-----
    python plot_appendix.py mean      # mean center line + IQR bands
    python plot_appendix.py median    # median center line + IQR bands
    python plot_appendix.py           # defaults to mean
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

def load_5row(path, center_fn):
    """Load Arrhenius / cyclic experiment data and compute summary statistics.

    Parameters
    ----------
    path : str
        Path to .npz file with shape (5, N) arrays keyed by zero-padded
        system size.  Rows: [MEPS_EPR, NESS_EPR, UNIF_EPR, DKL_NM, DKL_MN].
    center_fn : callable
        Aggregation function for center line (np.mean or np.median).

    Returns
    -------
    dict
        Keys: 's', 'epr_center', 'epr_q25', 'epr_q75',
              'unif_epr_center', 'unif_epr_q25', 'unif_epr_q75',
              'dkl_center', 'dkl_q25', 'dkl_q75'.
        All values are numpy arrays indexed by system size.
    """
    d = np.load(path)
    s_vals = d['s_values']
    result = {k: [] for k in [
        's',
        'epr_center', 'epr_q25', 'epr_q75',
        'unif_epr_center', 'unif_epr_q25', 'unif_epr_q75',
        'dkl_center', 'dkl_q25', 'dkl_q75',
    ]}
    for s in s_vals:
        if s < MIN_SYSTEM_SIZE:
            continue
        key = f'{s:05}'
        if key not in d:
            continue
        arr = d[key]
        ness_epr = arr[1]

        # Equilibrium filter: drop trials at the float-noise floor
        eq_mask = ness_epr > EQ_THRESHOLD
        n_total = len(ness_epr)
        n_kept = eq_mask.sum()
        if n_kept < MIN_TRIALS:
            continue
        _filter_log.append((os.path.basename(path), s, n_total, n_total - n_kept))

        # Compute excess EPR ratios: sigma / sigma_MEPS - 1
        excess = arr[1][eq_mask] / arr[0][eq_mask] - 1
        unif_excess = arr[2][eq_mask] / arr[0][eq_mask] - 1
        dkl = arr[3][eq_mask]

        mask = np.isfinite(excess) & np.isfinite(unif_excess) & np.isfinite(dkl)
        if mask.sum() == 0:
            continue

        result['s'].append(s)
        result['epr_center'].append(center_fn(excess[mask]))
        result['epr_q25'].append(np.percentile(excess[mask], 25))
        result['epr_q75'].append(np.percentile(excess[mask], 75))
        result['unif_epr_center'].append(center_fn(unif_excess[mask]))
        result['unif_epr_q25'].append(np.percentile(unif_excess[mask], 25))
        result['unif_epr_q75'].append(np.percentile(unif_excess[mask], 75))
        result['dkl_center'].append(center_fn(dkl[mask]))
        result['dkl_q25'].append(np.percentile(dkl[mask], 25))
        result['dkl_q75'].append(np.percentile(dkl[mask], 75))

    return {k: np.array(v) for k, v in result.items()}


# ═════════════════════════════════════════════════════════════════════
# Figure builders
# ═════════════════════════════════════════════════════════════════════

def make_appendix_figure(data_dir, fracs, frac_labels, title, out_path,
                         center_fn):
    """2 x N panel figure: excess EPR (top) and D_KL (bottom) per density.

    Parameters
    ----------
    data_dir : str
        Directory containing frac{NNN}.npz files.
    fracs : list of float
        Edge density fractions (e.g. [0.10, 0.25, 0.50, 1.00]).
    frac_labels : list of str
        Human-readable labels for each fraction.
    title : str
        Figure super-title.
    out_path : str
        Output PNG path.
    center_fn : callable
        np.mean or np.median.
    """
    n_cols = len(fracs)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 7), sharex=True)
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    colors = plt.cm.tab10(np.linspace(0, 0.4, n_cols))

    for col, (frac, flabel) in enumerate(zip(fracs, frac_labels)):
        frac_key = f'frac{int(frac*100):03d}'
        path = os.path.join(data_dir, f'{frac_key}.npz')
        if not os.path.exists(path):
            for row in range(2):
                axes[row, col].text(0.5, 0.5, 'no data', ha='center',
                                    va='center', transform=axes[row, col].transAxes)
            axes[0, col].set_title(f'Edge density = {flabel}')
            continue

        d = load_5row(path, center_fn)
        color = colors[col]

        # Top row: excess EPR (NESS and uniform)
        ax = axes[0, col]
        ax.plot(d['s'], d['epr_center'], 'o-', color=color, markersize=5,
                lw=2, label='NESS')
        ax.fill_between(d['s'], d['epr_q25'], d['epr_q75'],
                        alpha=0.2, color=color)
        ax.plot(d['s'], d['unif_epr_center'], '^--', color='gray',
                markersize=4, lw=1.5, alpha=0.7, label='Uniform')
        ax.fill_between(d['s'], d['unif_epr_q25'], d['unif_epr_q75'],
                        alpha=0.1, color='gray')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'Edge density = {flabel}', fontsize=12)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(r'$\sigma/\sigma_{\mathrm{MEPS}} - 1$', fontsize=11)
            ax.legend(fontsize=9)

        # Bottom row: D_KL(NESS || MEPS)
        ax = axes[1, col]
        ax.plot(d['s'], d['dkl_center'], 's-', color=color, markersize=5, lw=2)
        ax.fill_between(d['s'], d['dkl_q25'], d['dkl_q75'],
                        alpha=0.2, color=color)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('System size S', fontsize=11)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(
                r'$D_{\mathrm{KL}}(\mathrm{NESS} \| \mathrm{MEPS})$',
                fontsize=11)

    fig.suptitle(title, fontsize=13, y=1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved -> {out_path}")
    plt.close()


def make_cyclic_panel_figure(data_dir, keys, labels, title, out_path,
                             center_fn, colors=None):
    """2 x N panel figure for cyclic ring experiments.

    Same layout as make_appendix_figure but loads by filename key
    rather than computing frac keys from floats.
    """
    n_cols = len(keys)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 7), sharex=True)
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 0.4, n_cols))

    for col, (key, label) in enumerate(zip(keys, labels)):
        path = os.path.join(data_dir, f'{key}.npz')
        if not os.path.exists(path):
            for row in range(2):
                axes[row, col].text(0.5, 0.5, 'no data', ha='center',
                                    va='center', transform=axes[row, col].transAxes)
            axes[0, col].set_title(label, fontsize=12)
            continue

        d = load_5row(path, center_fn)
        color = colors[col]

        ax = axes[0, col]
        ax.plot(d['s'], d['epr_center'], 'o-', color=color, markersize=5,
                lw=2, label='NESS')
        ax.fill_between(d['s'], d['epr_q25'], d['epr_q75'],
                        alpha=0.2, color=color)
        ax.plot(d['s'], d['unif_epr_center'], '^--', color='gray',
                markersize=4, lw=1.5, alpha=0.7, label='Uniform')
        ax.fill_between(d['s'], d['unif_epr_q25'], d['unif_epr_q75'],
                        alpha=0.1, color='gray')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(r'$\sigma/\sigma_{\mathrm{MEPS}} - 1$', fontsize=11)
            ax.legend(fontsize=9)

        ax = axes[1, col]
        ax.plot(d['s'], d['dkl_center'], 's-', color=color, markersize=5, lw=2)
        ax.fill_between(d['s'], d['dkl_q25'], d['dkl_q75'],
                        alpha=0.2, color=color)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('System size S', fontsize=11)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(
                r'$D_{\mathrm{KL}}(\mathrm{NESS} \| \mathrm{MEPS})$',
                fontsize=11)

    fig.suptitle(title, fontsize=13, y=1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved -> {out_path}")
    plt.close()


def make_decay_figure(data_dir, gammas, out_path, center_fn,
                      edge_label="full ring"):
    """1 x 2 overlay figure: excess EPR and D_KL for each decay exponent.

    Multiple gamma values are plotted on shared axes with a shared legend.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    cmap = plt.cm.plasma
    colors = [cmap(x) for x in np.linspace(0.05, 0.95, len(gammas))]

    handles = []
    for gamma, color in zip(gammas, colors):
        # File naming: gamma=0.25 -> alpha0p25.npz
        key = f"alpha{gamma:.2f}".replace('.', 'p')
        path = os.path.join(data_dir, f'{key}.npz')
        if not os.path.exists(path):
            continue
        d = load_5row(path, center_fn)
        label = rf'$\gamma={gamma}$'

        h, = axes[0].plot(d['s'], d['epr_center'], 'o-', color=color,
                          markersize=4, lw=2, label=label)
        axes[0].fill_between(d['s'], d['epr_q25'], d['epr_q75'],
                             alpha=0.1, color=color)
        handles.append(h)

        axes[1].plot(d['s'], d['dkl_center'], 's-', color=color,
                     markersize=4, lw=2, label=label)
        axes[1].fill_between(d['s'], d['dkl_q25'], d['dkl_q75'],
                             alpha=0.1, color=color)

    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('System size S', fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(
        r'$\sigma_{\mathrm{NESS}}/\sigma_{\mathrm{MEPS}} - 1$', fontsize=11)
    axes[1].set_ylabel(
        r'$D_{\mathrm{KL}}(\mathrm{NESS} \| \mathrm{MEPS})$', fontsize=11)

    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc='lower center', ncol=min(len(handles), 5),
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    params = (
        r'$R_{s \to s^\prime} \propto '
        r'e^{-\gamma \cdot d(s,s^\prime) + \mathcal{N}(0,1)}$, '
        r'$R_{s^\prime \to s} = R_{s \to s^\prime} \cdot U(0, 1.5)$, '
        + edge_label)
    fig.suptitle('Distance Decay Breaks Entropy Minimization\n' + params,
                 fontsize=13, y=1.02)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved -> {out_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main(statistic='mean'):
    """Generate all 10 appendix figures.

    Parameters
    ----------
    statistic : str
        'mean' or 'median' — determines center line in all plots.
    """
    center_fn = resolve_center_fn(statistic)

    out_dir = os.path.join(BASE, 'final_plots', statistic)
    os.makedirs(out_dir, exist_ok=True)

    # ── Arrhenius appendix: ER/SW x pos/neg pump ─────────────────────
    er_fracs = [0.10, 0.25, 0.50, 1.00]
    er_labels = ['10%', '25%', '50%', '100%']
    sw_fracs = [0.10, 0.25, 0.50]
    sw_labels = ['10%', '25%', '50%']

    param_er = (r'$E_s, E_{s \leftrightarrow s^\prime} \sim U(0,1)$, '
                r'pump fraction = 20%')
    param_sw = (r'$E_s, E_{s \leftrightarrow s^\prime} \sim U(0,1)$, '
                r'pump fraction = 20%, $\beta_{\mathrm{SW}}=0.1$')

    configs = [
        ('data_appendix_er_pos', er_fracs, er_labels,
         r'Erdos-Renyi Sparsity -- Positive Pump ($\alpha = 2$)'
         + '\n' + param_er,
         'fig_appendix_er_pos.png'),
        ('data_appendix_er_neg', er_fracs, er_labels,
         r'Erdos-Renyi Sparsity -- Negative Pump ($\alpha = -2$)'
         + '\n' + param_er,
         'fig_appendix_er_neg.png'),
        ('data_appendix_sw_pos', sw_fracs, sw_labels,
         r'Small-World Sparsity -- Positive Pump ($\alpha = 2$)'
         + '\n' + param_sw,
         'fig_appendix_sw_pos.png'),
        ('data_appendix_sw_neg', sw_fracs, sw_labels,
         r'Small-World Sparsity -- Negative Pump ($\alpha = -2$)'
         + '\n' + param_sw,
         'fig_appendix_sw_neg.png'),
    ]
    for data_sub, fracs, labels, title, fname in configs:
        make_appendix_figure(
            os.path.join(BASE, data_sub), fracs, labels, title,
            os.path.join(out_dir, fname), center_fn)

    # ── Cyclic flat ring figures ─────────────────────────────────────
    fracs_cyc = ['frac010', 'frac025', 'frac050']
    frac_labels_cyc = ['Edge density = 10%', 'Edge density = 25%',
                       'Edge density = 50%']

    flat_params = (
        r'Ring with fwd/bwd asymmetry, no distance decay ($\gamma=0$)'
        '\n'
        r'$R_{s \to s^\prime} \propto e^{\mathcal{N}(0,1)}$, '
        r'$R_{s^\prime \to s} = R_{s \to s^\prime} \cdot U(0, 1.5)$')
    make_cyclic_panel_figure(
        os.path.join(BASE, 'data_cyclic_flat'), fracs_cyc, frac_labels_cyc,
        flat_params, os.path.join(out_dir, 'fig_cyclic_flat_ring.png'),
        center_fn)

    sw_params = (
        r'Ring + Small-World rewiring ($\beta_{\mathrm{SW}}=0.1$), '
        r'no distance decay ($\gamma=0$)'
        '\n'
        r'$R_{s \to s^\prime} \propto e^{\mathcal{N}(0,1)}$, '
        r'$R_{s^\prime \to s} = R_{s \to s^\prime} \cdot U(0, 1.5)$')
    make_cyclic_panel_figure(
        os.path.join(BASE, 'data_cyclic_flat_sw'), fracs_cyc, frac_labels_cyc,
        sw_params, os.path.join(out_dir, 'fig_cyclic_flat_ring_sw.png'),
        center_fn)

    # ── Distance decay sweep figures ─────────────────────────────────
    gammas = [0.0, 0.01, 0.02, 0.08, 0.25, 1.0]
    decay_configs = [
        ('data_cyclic_decay',
         'fig_cyclic_decay_sweep_full.png',
         r'full ring (max\_jump $= S/2$)'),
        ('data_cyclic_decay_sparse',
         'fig_cyclic_decay_sweep_sparse10.png',
         r'edge density $\approx 10\%$'),
        ('data_cyclic_decay_full_sw',
         'fig_cyclic_decay_sweep_full_sw.png',
         r'full ring + SW ($\beta_{\mathrm{SW}}=0.1$)'),
        ('data_cyclic_decay_sparse_sw',
         'fig_cyclic_decay_sweep_sparse10_sw.png',
         r'edge density $\approx 10\%$ + SW ($\beta_{\mathrm{SW}}=0.1$)'),
    ]
    for data_sub, fname, edge_label in decay_configs:
        make_decay_figure(
            os.path.join(BASE, data_sub), gammas,
            os.path.join(out_dir, fname), center_fn,
            edge_label=edge_label)

    # ── Diagnostics ──────────────────────────────────────────────────
    print_filter_diagnostics(statistic)


if __name__ == "__main__":
    stat = sys.argv[1] if len(sys.argv) > 1 else 'mean'
    main(stat)
