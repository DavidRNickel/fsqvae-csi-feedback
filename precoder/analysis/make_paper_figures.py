"""
Generates a fixed set of publication-ready figures for an IEEE conference submission
(tight page budget: single-column width, minimal whitespace, no notebook-style
titles/legends-outside-axes). Chosen to cover the performance story end to end rather
than exhaustively sweep every config:

  -- deepmimo (real, ray-traced), the `_dm_` tag --
  1. sumrate_vs_snr_dm_N4_Tu12.png  -- mildest undersampling in this set (Td/N=1.0,
                                       pilots span the full channel rank) -- every family
                                       nearly coincides; the low end of the N=4/8/32 trio
  2. sumrate_vs_snr_dm_N8_Tu12.png  -- contrast: mild undersampling (Td/N=0.75), same
                                       budget/family set -- CVQ is competitive here
  3. sumrate_vs_snr_dm_N32_Tu12.png -- headline result: severe pilot undersampling
                                       (Td/N=0.25), matched feedback budget, all families
  4. sumrate_vs_bits_dm_N8.png  -- rate/feedback-budget tradeoff at mild undersampling --
                                       contrast with #5's severe-undersampling tradeoff
  5. sumrate_vs_bits_dm_N32.png -- the rate/feedback-budget tradeoff at severe undersampling
  6. fsq_vs_fsqm_dm_N32.png     -- ablation: learned vs. DFT pilots, quantizer held
                                       fixed (same Tu) -- isolates the pilot-learning gain

  Filenames 1-3 carry a `_Tu<n>` tag (the matched feedback budget), since those three
  figures have no title and their legend only names each family, not its budget -- unlike
  #4/#5 (Tu is the x-axis) or #6 (each legend label already says "Tu=12"/"Tu=40"), nothing
  in the image itself would otherwise say which budget was plotted.

  -- gmm (synthetic Gaussian-mixture), the `_gmm_` tag --
  7. sumrate_vs_snr_gmm_N4_Tu12_K1.png  -- low end of the gmm N=4/8/32 trio, K=1
  8. sumrate_vs_snr_gmm_N8_Tu12_K{1,2}.png  -- middle of the trio, K=1
  9. sumrate_vs_snr_gmm_N32_Tu16_K{1,2}.png -- the deepmimo-N=32 comparison repeated on gmm.
                                       Deliberately NOT a "generalizes to gmm too" figure --
                                       checked first (see git history), and on gmm neither
                                       FSQM nor FSQ leads at any (K, J, budget): CVQ/KVQ win
                                       instead. Included to make an honest claim: the FSQM
                                       advantage is specific to real, ray-traced deepmimo
                                       channels (undersampled, high-rank, non-Gaussian), not a
                                       universal property of learned pilots -- on a channel a
                                       KLT already matches well, classical quantization wins.
 10. sumrate_vs_bits_gmm_N32_K1.png -- the bits tradeoff on gmm -- unlike #5's dm version,
                                       CVQ's curve rises steeply and does not plateau
 11. sumrate_vs_snr_gmm_K_sweep_N32_Tu16.png -- gmm-only axis of variation (no dm equivalent):
                                       CVQ vs. FSQ vs. FSQM (and VQ/VQM, where their K runs exist)
                                       across the mixture component count K.
                                       Two findings: CVQ's lead over FSQM *grows* with K
                                       (7.68->11.24 at SNR=10dB, 16 bits) while FSQM's *shrinks*
                                       (6.54->5.45) -- a K-means codebook fits a K-component
                                       mixture better as K grows, a fixed-capacity learned
                                       quantizer doesn't; and FSQ (fixed DFT pilots) only loses to
                                       FSQM (learned pilots) at K=1 -- at K=2/4, FSQ beats FSQM at
                                       every SNR, the opposite of deepmimo's story. Colors are
                                       swapped from every other figure here (FSQM is blue, FSQ is
                                       gray) -- see K_SWEEP_COLORS' comment.

  -- mechanism --
 12. fsqvae_latent_space_dm_Tu12.png -- FSQ-VAE latent space, unquantized vs. quantized, at
                                       SNR=0 and SNR=15 (4 panels total). `_Tu12` because the model's Tu
                                       (from LATENT_RUN-equivalent `run` in fig_latent_space)
                                       isn't shown anywhere in the image itself.

Every filename tags its channel model: `_dm_` for deepmimo, `_gmm_` for gmm.

Figures 1-11 need only pandas/matplotlib (plain `python3`). Figure 12 additionally needs
torch/einops/scikit-learn to load the trained FSQVAE checkpoint
(results_fsq_eval/, see the README) -- if it lives in a separate environment:
    python make_paper_figures.py --latent-only
"""
import argparse
import colorsys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'photos')

# ---------------------------------------------------------------------------
# Publication rcParams: small fonts/markers/linewidths sized for a figure that
# will be shrunk to IEEE single-column width (~3.45in) in the final PDF, and
# tight margins everywhere so savefig(bbox_inches='tight') has little to trim.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'markers.fillstyle': 'full',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'grid.linewidth': 0.4,
    'legend.handlelength': 1.6,
    'legend.labelspacing': 0.3,
    'legend.borderpad': 0.3,
    'legend.columnspacing': 1.0,
    'pdf.fonttype': 42,   # embed fonts as real text, not paths -- IEEE/editing-friendly
    'ps.fonttype': 42,
    'savefig.dpi': 400,
    'figure.dpi': 150,
})

SINGLE_COL = (3.45, 2.35)  # inches -- fits one IEEE column with no scaling needed

FAMILY_STYLE = {
    'FSQ':  {'color': 'tab:blue',   'linestyle': '-',  'marker': 'o'},
    'FSQM': {'color': 'tab:orange', 'linestyle': '-',  'marker': 's'},
    'VQ':   {'color': 'tab:cyan',   'linestyle': '-',  'marker': 'v'},
    'VQM':  {'color': 'tab:brown',  'linestyle': '-',  'marker': 'P'},
    'CVQ':  {'color': 'tab:red',    'linestyle': '-.', 'marker': '^'},
    'DVQ':  {'color': 'tab:purple', 'linestyle': ':',  'marker': 'x'},
    'KVQ':  {'color': 'tab:green',  'linestyle': '--', 'marker': 'D'},
}
FAMILY_ORDER = ['FSQ', 'FSQM', 'VQ', 'VQM', 'CVQ', 'DVQ', 'KVQ']
# DVQ/KVQ's reported std is aggregated differently from the multi-run mean/std of FSQ/FSQM/VQ/VQM/CVQ,
# so it isn't comparable and is never drawn as an error bar
ERRBAR_FAMILIES = ('FSQ', 'FSQM', 'VQ', 'VQM', 'CVQ')


def _shades(base_color, n):
    """n evenly-spaced lightness variants of base_color (darkest first), for sibling
    lines that share one family's color -- e.g. different K at fixed family.

    The saturation floor below only applies when base_color already has some saturation:
    colorsys reports an achromatic color's hue as 0 (red) since hue is undefined for pure
    gray, so flooring saturation unconditionally would force a *tinted red*, not a gray,
    out of e.g. 'tab:gray' -- silently, since the result is still a valid color, just not
    the one asked for. An explicitly gray base_color is kept achromatic."""
    if n <= 1:
        return [base_color]
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l_min, l_max = 0.30, 0.65
    s_out = s if s < 1e-6 else max(s, 0.4)
    return [colorsys.hls_to_rgb(h, l_min + (l_max - l_min) * i / (n - 1), s_out)
            for i in range(n)]


def _family(method, pilot_type):
    if method == 'fsqvae':
        return 'FSQM' if pilot_type == 'learned' else 'FSQ'
    if method == 'vqvae':
        return 'VQM' if pilot_type == 'learned' else 'VQ'
    return {'cvq_proj': 'CVQ', 'dft_rotated': 'DVQ', 'transform_vq': 'KVQ'}[method]


def _bits(row):
    """Total feedback bits/user, counted per family. CVQ: Tu(O) + TuM. DVQ: log2(N*O) + TuM, since
    its O is an oversampling factor, not a bit count. Everything else: O directly."""
    if row['fam'] == 'CVQ':
        return row['O'] + row['TuM']
    if row['fam'] == 'DVQ':
        return np.log2(row['N'] * row['O']) + row['TuM']
    return row['O']


def load_sumrate_table():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aggregated_results.csv'))
    df['TuM'] = df['TuM'].fillna(0)
    df['fam'] = [_family(m, p) for m, p in zip(df['method'], df['pilot_type'])]
    df['bits'] = df.apply(_bits, axis=1)
    # CVQ has two 12-bit configs (Tu=7/TuM=5 and Tu=8/TuM=4); plot only 8/4, which does better on
    # average: on gmm at 78% of matched points (mean +0.20 bits/s/Hz), on deepmimo only narrowly
    # (56% of points, mean +0.11, but its largest gaps all favor 8/4). Left in the CSV, dropped here.
    df = df[~((df['fam'] == 'CVQ') & (df['O'] == 7) & (df['TuM'] == 5))]
    return df


def _tight_save(fig, name):
    # PNG only -- a vector .pdf was dropped from here on request, so don't regenerate it
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f'{name}.png'), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'wrote {name}.png')


# ---------------------------------------------------------------------------
# Figures 1 & 2 & 5: sum rate vs. SNR at one matched feedback budget, one J,
# every family that has a config at that budget.
# ---------------------------------------------------------------------------
def _plot_sumrate_vs_snr(ax, df, name, channel_model, N, Td, J, budget, K=None, legend_loc='upper left', legend=True):
    mask = (df['channel_model'] == channel_model) & (df['N'] == N) & (df['Td'] == Td) & (df['J'] == J)
    if K is not None:
        mask &= df['K'] == K
    base = df[mask]
    if base.empty:
        raise RuntimeError(f'{name}: no rows for {channel_model} N={N} Td={Td} J={J} K={K}')

    for fam in FAMILY_ORDER:
        style = FAMILY_STYLE[fam]
        rows = base[(base['fam'] == fam) & np.isclose(base['bits'], budget)].sort_values('snr_db')
        if rows.empty:
            continue
        dupes = rows['snr_db'][rows['snr_db'].duplicated()]
        assert dupes.empty, (f'{name}/{fam}: {len(dupes)} duplicate SNR point(s) at budget={budget} -- '
                             f'an underspecified filter (missing K?) is mixing rows into one line, '
                             f'see fig_sumrate_vs_bits\' K-mixing bug for what this looks like when missed')
        ax.plot(rows['snr_db'], rows['sumrate'], label=fam, **style)
        std = rows['sumrate_std'].to_numpy()
        mask_std = np.isfinite(std) & (std > 0) & (fam in ERRBAR_FAMILIES)
        if mask_std.any():
            ax.errorbar(rows['snr_db'].to_numpy()[mask_std], rows['sumrate'].to_numpy()[mask_std],
                       yerr=std[mask_std], fmt='none', ecolor=style['color'], elinewidth=0.7,
                       capsize=1.5, alpha=0.8, zorder=1)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Sum rate (bits/s/Hz)')
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc=legend_loc, ncol=1, frameon=True, framealpha=0.9)


def fig_sumrate_vs_snr(df, name, *args, **kwargs):
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    _plot_sumrate_vs_snr(ax, df, name, *args, **kwargs)
    fig.tight_layout(pad=0.3)
    _tight_save(fig, name)


def fig_sumrate_vs_snr_mixed(df, name, panels):
    # 1xN row spanning both IEEE columns; each panel is (title, kwargs for _plot_sumrate_vs_snr),
    # and one shared legend (union of every panel's families) sits above the row
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 2.2))
    seen = {}
    for i, (ax, (title, kw)) in enumerate(zip(axes, panels)):
        _plot_sumrate_vs_snr(ax, df, name, legend=False, **kw)
        ax.set_title(f'({chr(ord("a") + i)}) {title}')
        if i > 0:
            ax.set_ylabel('')
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    labels = [f for f in FAMILY_ORDER if f in seen]
    fig.legend([seen[l] for l in labels], labels, loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), frameon=False)
    fig.tight_layout(pad=0.3)
    _tight_save(fig, name)




# ---------------------------------------------------------------------------
# Figure 3: sum rate vs. feedback bits at one SNR, one J.
# ---------------------------------------------------------------------------
def _plot_sumrate_vs_bits(ax, df, name, channel_model, N, Td, J, snr, bits_values=(8, 12, 16, 20), K=None, legend=True):
    mask = (df['channel_model'] == channel_model) & (df['N'] == N) & (df['Td'] == Td) & (df['J'] == J)
    if K is not None:
        mask &= df['K'] == K
    base = df[mask & np.isclose(df['snr_db'], snr)]
    if base.empty:
        raise RuntimeError(f'{name}: no rows for {channel_model} N={N} Td={Td} J={J} SNR={snr} K={K}')

    for fam in FAMILY_ORDER:
        style = FAMILY_STYLE[fam]
        rows = base[(base['fam'] == fam) & base['bits'].round(6).isin(bits_values)].sort_values('bits')
        if rows.empty:
            continue
        dupes = rows['bits'][rows['bits'].duplicated()]
        assert dupes.empty, (f'{name}/{fam}: {len(dupes)} duplicate bits point(s) -- an underspecified '
                             f'filter (missing K?) is mixing rows into one line')
        ax.plot(rows['bits'], rows['sumrate'], label=fam, **style)
        std = rows['sumrate_std'].to_numpy()
        mask_std = np.isfinite(std) & (std > 0) & (fam in ERRBAR_FAMILIES)
        if mask_std.any():
            ax.errorbar(rows['bits'].to_numpy()[mask_std], rows['sumrate'].to_numpy()[mask_std],
                       yerr=std[mask_std], fmt='none', ecolor=style['color'], elinewidth=0.7,
                       capsize=1.5, alpha=0.8, zorder=1)

    ax.set_xlabel('Feedback bits / user')
    ax.set_ylabel('Sum rate (bits/s/Hz)')
    ax.set_xticks(list(bits_values))
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc='upper left', ncol=2, frameon=True, framealpha=0.9)


def fig_sumrate_vs_bits(df, name, *args, **kwargs):
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    _plot_sumrate_vs_bits(ax, df, name, *args, **kwargs)
    fig.tight_layout(pad=0.3)
    _tight_save(fig, name)


def fig_sumrate_vs_bits_all(df, name, channel_model, configs, snr, bits_values=(8, 12, 16, 20)):
    # 1xN row spanning both IEEE columns; one panel per (N, Td, J) config
    fig, axes = plt.subplots(1, len(configs), figsize=(7.16, 2.2))
    for i, (ax, (N, Td, J)) in enumerate(zip(axes, configs)):
        _plot_sumrate_vs_bits(ax, df, name, channel_model, N=N, Td=Td, J=J, snr=snr, bits_values=bits_values, legend=False)
        ax.set_title(f'({chr(ord("a") + i)}) DeepMIMO\n$N={N}$, $J={J}$')
        if i > 0:
            ax.set_ylabel('')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(labels), frameon=False)
    fig.tight_layout(pad=0.3)
    _tight_save(fig, name)


# ---------------------------------------------------------------------------
# Figure 4: FSQ vs. FSQM ablation -- same Tu (quantizer resolution), only the
# pilot matrix (DFT vs. learned) differs, across the full SNR range.
# ---------------------------------------------------------------------------
def fig_fsq_vs_fsqm(df, name, N, J, tu_values=(12, 40)):
    base = df[(df['channel_model'] == 'deepmimo') & (df['method'] == 'fsqvae') & (df['N'] == N) & (df['J'] == J)]
    if base.empty:
        raise RuntimeError(f'{name}: no fsqvae rows for N={N} J={J}')

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    tu_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(tu_values)))
    for tu, color in zip(tu_values, tu_colors):
        for pilot, ls, mk in (('dft', '-', 'o'), ('learned', '--', 's')):
            rows = base[(base['O'] == tu) & (base['pilot_type'] == pilot)].sort_values('snr_db')
            if rows.empty:
                continue
            label = f"{'FSQM' if pilot == 'learned' else 'FSQ'} (Tu={tu:g})"
            ax.plot(rows['snr_db'], rows['sumrate'], color=color, linestyle=ls, marker=mk,
                   markersize=4, label=label)
            std = rows['sumrate_std'].to_numpy()
            m = np.isfinite(std) & (std > 0)
            if m.any():
                ax.errorbar(rows['snr_db'].to_numpy()[m], rows['sumrate'].to_numpy()[m], yerr=std[m],
                           fmt='none', ecolor=color, elinewidth=0.7, capsize=1.5, alpha=0.8, zorder=1)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Sum rate (bits/s/Hz)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', ncol=1, frameon=True, framealpha=0.9)
    fig.tight_layout(pad=0.3)
    _tight_save(fig, name)


# ---------------------------------------------------------------------------
# gmm-only figure: sum rate vs. SNR across the mixture component count K, for a handful of
# families, at one matched budget. gmm has no dm equivalent of this axis (K is a property of
# the synthetic mixture model only) -- see module docstring for the two findings this exposes
# (CVQ's lead over FSQM grows with K; FSQ beats FSQM once K>1). Note the
# color override (see K_SWEEP_COLORS below).
# ---------------------------------------------------------------------------
# Local to this figure only -- does NOT touch FAMILY_STYLE, which every other figure in this
# script reads from. FSQM -> blue (on request) and FSQ -> gray, since FSQ's usual blue is now
# taken; CVQ keeps its usual red. NOTE: FSQ/FSQM are therefore the opposite colors here vs.
# every other figure in this script (FSQ is blue everywhere else) -- read the legend, not the
# color, when comparing this figure to any other.
K_SWEEP_COLORS = {'CVQ': 'tab:red', 'FSQ': 'tab:gray', 'FSQM': 'tab:blue'}


def fig_gmm_k_sweep(df, name, N, Td, J, budget, families=('CVQ', 'FSQ', 'FSQM'), Ks=(1, 2, 4)):
    base = df[(df['channel_model'] == 'gmm') & (df['N'] == N) & (df['Td'] == Td) & (df['J'] == J)
             & np.isclose(df['bits'], budget)]
    if base.empty:
        raise RuntimeError(f'{name}: no rows for gmm N={N} Td={Td} J={J} budget={budget}')

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    for fam in families:
        base_color = K_SWEEP_COLORS.get(fam, FAMILY_STYLE[fam]['color'])
        shades = _shades(base_color, len(Ks))
        for K, shade in zip(Ks, shades):
            rows = base[(base['fam'] == fam) & (base['K'] == K)].sort_values('snr_db')
            if rows.empty:
                continue
            ax.plot(rows['snr_db'], rows['sumrate'], color=shade, linestyle=FAMILY_STYLE[fam]['linestyle'],
                   marker=FAMILY_STYLE[fam]['marker'], markersize=4, label=f'{fam} (K={K:g})')
            std = rows['sumrate_std'].to_numpy()
            m = np.isfinite(std) & (std > 0)
            if m.any():
                ax.errorbar(rows['snr_db'].to_numpy()[m], rows['sumrate'].to_numpy()[m], yerr=std[m],
                           fmt='none', ecolor=shade, elinewidth=0.7, capsize=1.5, alpha=0.8, zorder=1)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Sum rate (bits/s/Hz)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', ncol=2, frameon=True, framealpha=0.9, fontsize=5.5)
    fig.tight_layout(pad=0.3)
    _tight_save(fig, name)


# ---------------------------------------------------------------------------
# Figure 12: FSQ-VAE latent space (needs torch/einops/fsqvae)
# ---------------------------------------------------------------------------
def fig_latent_space(snrs_db=(0, 15), levels_expected=3):
    import sys
    from pathlib import Path
    from contextlib import contextmanager
    import argparse as _argparse
    import torch

    precoder_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    if str(precoder_dir) not in sys.path:
        sys.path.insert(0, str(precoder_dir))
    from fsqvae import FSQVAE
    from common_setup import add_shared_args, finalize_args, build_network_conf, load_train_val_test_channels

    run = {'num_bs_ant': 32, 'num_uts': 20, 'Td': 8, 'Tu': 12, 'dirnum': '_eval', 'use_dft_matrix': False}
    seed = 0

    @contextmanager
    def _in_dir(path):
        prev = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev)

    parser = _argparse.ArgumentParser()
    add_shared_args(parser)
    args = parser.parse_args(['--num-bs-ant', str(run['num_bs_ant']), '--num-uts', str(run['num_uts']),
                              '--Td', str(run['Td']), '--Tu', str(run['Tu']), '--dirnum', run['dirnum'],
                              '--use-dft-matrix', str(run['use_dft_matrix']), '--seed', str(seed)])
    finalize_args(args)
    fsqvae_config, network_conf, N, J = build_network_conf(args)

    cache = precoder_dir / f'results_fsq{args.dirnum}'
    if not args.use_dft_matrix:
        cache = cache / 'learn_M'
    model_dir = cache / args.scenario / f'N{N}' / f'N{N}_J{J}_Td{args.Td}_Tu{args.Tu}'

    model = FSQVAE(fsqvae_config, network_conf, splice_latent=True, get_quantized_splice=False)
    model.load_state_dict(torch.load(model_dir / 'best_model.pt', weights_only=True, map_location=args.device))
    model.to(args.device).eval()
    assert len(model.levels) == levels_expected, f'need a {levels_expected}D latent space, got {model.levels}'

    with _in_dir(precoder_dir):
        chans_train, _, _ = load_train_val_test_channels(args, N, J)
    H = chans_train.reshape(-1, J, N).permute(0, 2, 1)

    # one shared pilot-noise draw per SNR -- the quantized panel is then exactly the rounded
    # unquantized one, same as the notebook's Part 4 (compute_fsq_latents)
    latents = {}
    with torch.no_grad():
        for snr_db in snrs_db:
            sqrt_noise_pwr = float(np.sqrt(10 ** (-snr_db / 10)))
            noise = sqrt_noise_pwr * torch.randn(size=(H.shape[0], args.Td, J), dtype=torch.cfloat,
                                                 device=args.device)
            latents[snr_db] = {}
            for name, quantize in (('unquantized', False), ('quantized', True)):
                model.fsq.get_quantized_splice = quantize
                z, _ = model((H, noise))
                latents[snr_db][name] = z.reshape(-1, z.shape[-1]).cpu().numpy()

    panels = [(snr_db, name) for snr_db in snrs_db for name in ('unquantized', 'quantized')]
    markersize = 0.6
    fig = plt.figure(figsize=(7.16, 2.6))  # full IEEE two-column width
    for idx, (snr_db, name) in enumerate(panels):
        z = latents[snr_db][name]
        if name == 'quantized':
            z = np.unique(z, axis=0)
        ax = fig.add_subplot(1, len(panels), idx + 1, projection='3d')
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], s=markersize, rasterized=True, color='tab:blue')
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ticks = [-1, 0, 1]
        ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(ticks)
        ax.tick_params(labelsize=5.5, pad=-2)
        ax.set_box_aspect((1, 1, 1), zoom=0.95)
        caption = f'({chr(ord("a") + idx)}) {name.capitalize()}, SNR {snr_db:g}.'
        ax.text2D(0.5, -0.14, caption, transform=ax.transAxes, ha='center', va='top', fontsize=7)
    os.makedirs(OUT_DIR, exist_ok=True)
    # Tu (uplink pilot symbols) isn't shown anywhere in the image itself (no title, and the
    # per-panel captions only name the SNR/quantization state) -- carried in the filename instead,
    # read off `run['Tu']` rather than hardcoded so it can't drift out of sync with `run` above
    out_name = f"fsqvae_latent_space_dm_Tu{run['Tu']}.png"
    fig.savefig(os.path.join(OUT_DIR, out_name), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'wrote {out_name}')


# (N, Td, J) panels of sumrate_vs_bits_dm_all, and the panels of sumrate_vs_snr_mixed_Tu12
DM_ALL_CONFIGS = [(4, 4, 4), (8, 6, 8), (32, 8, 12)]
MIXED_PANELS = [
    ('DeepMIMO\n$N=8$, $J=8$', dict(channel_model='deepmimo', N=8, Td=6, J=8, budget=12)),
    ('DeepMIMO\n$N=32$, $J=20$', dict(channel_model='deepmimo', N=32, Td=8, J=20, budget=12)),
    ('GMM, $K=2$\n$N=8$, $J=8$', dict(channel_model='gmm', N=8, Td=8, J=8, budget=12, K=2)),
    ('GMM, $K=2$\n$N=32$, $J=20$', dict(channel_model='gmm', N=32, Td=32, J=20, budget=12, K=2)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-only', action='store_true',
                        help='only build figure 12 (needs torch -- see module docstring)')
    parser.add_argument('--skip-latent', action='store_true',
                        help='build figures 1-11 only, skip figure 12 (needs torch/einops)')
    args = parser.parse_args()

    if args.latent_only:
        fig_latent_space()
        return

    df = load_sumrate_table()

    # --- deepmimo: N=4/8/32 trio, same 12-bit budget throughout ---
    # fig_sumrate_vs_snr draws no title/legend annotation of the budget anywhere in the image
    # (unlike fig_fsq_vs_fsqm's per-line "Tu=" labels, or fig_gmm_k_sweep's title) -- so unlike
    # those, its filename carries the Tu value explicitly.
    # 1. mildest undersampling here (Td/N=4/4=1.0) -- pilots span the full channel rank
    fig_sumrate_vs_snr(df, 'sumrate_vs_snr_dm_N4_Tu12', 'deepmimo', N=4, Td=4, J=4, budget=12)
    # 2. mild undersampling (Td/N=6/8=0.75)
    fig_sumrate_vs_snr(df, 'sumrate_vs_snr_dm_N8_Tu12', 'deepmimo', N=8, Td=6, J=8, budget=12,
                      legend_loc='upper left')
    # 3. headline: severe undersampling (Td/N=8/32=0.25)
    fig_sumrate_vs_snr(df, 'sumrate_vs_snr_dm_N32_Tu12', 'deepmimo', N=32, Td=8, J=12, budget=12)
    # 4 & 5. rate/feedback-budget tradeoff at mild vs. severe undersampling, same SNR -- Tu is the
    # x-axis here, so it's already apparent in the image; no filename tag needed
    fig_sumrate_vs_bits(df, 'sumrate_vs_bits_dm_N8', 'deepmimo', N=8, Td=6, J=8, snr=10)
    fig_sumrate_vs_bits(df, 'sumrate_vs_bits_dm_N32', 'deepmimo', N=32, Td=8, J=12, snr=10)
    fig_sumrate_vs_bits_all(df, 'sumrate_vs_bits_dm_all', 'deepmimo', configs=DM_ALL_CONFIGS, snr=10)
    # 6. ablation: learned vs. DFT pilots, quantizer (Tu) held fixed -- each line's own legend
    # label already says "Tu=12"/"Tu=40", so no filename tag needed
    fig_fsq_vs_fsqm(df, 'fsq_vs_fsqm_dm_N32', N=32, J=12, tu_values=(12, 40))

    # --- gmm: same N=4/8/32 trio (K=1), same 12-bit budget ---
    fig_sumrate_vs_snr(df, 'sumrate_vs_snr_gmm_N4_Tu12_K1', 'gmm', N=4, Td=4, J=4, budget=12, K=1)
    for K in (1, 2):
        fig_sumrate_vs_snr(df, f'sumrate_vs_snr_gmm_N8_Tu12_K{K}', 'gmm', N=8, Td=8, J=8, budget=12, K=K,
                          legend_loc='upper left')
    # 9. the deepmimo-N=32 comparison repeated on gmm -- deliberately NOT a "generalizes"
    #    figure: CVQ/KVQ lead here, not FSQM. Tu16, not Tu12 like the two above -- gmm's DVQ
    #    has no 12-bit config, and the filename tag makes that budget difference visible rather
    #    than silently assumed the same as every other N=4/N=8/N=32 trio figure
    for K in (1, 2):
        fig_sumrate_vs_snr(df, f'sumrate_vs_snr_gmm_N32_Tu16_K{K}', 'gmm', N=32, Td=32, J=12, budget=16, K=K,
                          legend_loc='upper left')
    # K=2 at the 12-bit budget (no DVQ config at 12 bits, so that curve is absent)
    fig_sumrate_vs_snr(df, 'sumrate_vs_snr_gmm_N32_Tu12_K2', 'gmm', N=32, Td=32, J=12, budget=12, K=2,
                      legend_loc='upper left')
    fig_sumrate_vs_snr_mixed(df, 'sumrate_vs_snr_mixed_Tu12', MIXED_PANELS)
    # 10. the bits tradeoff on gmm -- contrast with #5's dm version (CVQ keeps climbing here)
    fig_sumrate_vs_bits(df, 'sumrate_vs_bits_gmm_N32_K1', 'gmm', N=32, Td=32, J=12, snr=10, K=1)
    # 11. gmm-only axis: CVQ's lead over FSQM grows with the mixture component count K --
    #     no dm equivalent of this plot, K isn't a dm parameter
    fig_gmm_k_sweep(df, 'sumrate_vs_snr_gmm_K_sweep_N32_Tu16', N=32, Td=32, J=12, budget=16)
    # same sweep at other budgets -- CVQ has no 20-bit config, so it's absent from Tu20
    for budget in (12, 20):
        fig_gmm_k_sweep(df, f'sumrate_vs_snr_gmm_K_sweep_N32_Tu{budget}', N=32, Td=32, J=12, budget=budget)

    if not args.skip_latent:
        try:
            fig_latent_space()
        except ImportError as e:
            print(f'\nskipped figure 12 (latent space): {e}\n'
                 f'run it from an environment with torch instead:\n'
                 f'  python {__file__} --latent-only')


if __name__ == '__main__':
    main()
