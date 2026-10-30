"""
Aggregates precoder/ experiment results into tidy tables for plotting.

Two channel models feed results into these tables, distinguished by the
`channel_model` column:

  * `gmm` -- precoder/gmm/, driven by a synthetic K-component Gaussian-mixture
    channel model (no per-user identity; every result is additionally keyed
    by `K`, the number of mixture components).
  * `deepmimo` -- precoder/baselines/, driven by a real, ray-traced DeepMIMO
    channel dataset (currently `asu_campus_3p5`). There is no GMM component
    count here, so `K` is always null for these rows.

This writes one table, aggregated_results.pkl/.csv -- single-shot sum-rate
results (one-shot pilot-estimate-then-quantize-then-WMMSE). Combined here:

      1. FSQ-VAE (main.py / gmm/main.py), under
         gmm/results_fsq_{1,2,3,...}/{scenario}/N*/K*/N*_J*_Td*_Tu*/best_sumrate.txt
         (gmm) or the top-level results_fsq_{1,2,3,...}/{scenario}/N*/N*_J*_Td*_Tu*/best_sumrate.txt
         (deepmimo -- no K* level at all, since this channel model has no GMM
         components; kept in its own top-level results_fsq_*/, entirely separate
         from gmm/'s own results_fsq*/, so gmm and deepmimo FSQ-VAE runs never
         collide on disk) -- each best_sumrate.txt is a single Python-list-literal
         file holding the sum-rate achieved at each of the fixed evaluation SNRs
         (FSQVAE_EVAL_SNRS_DB below), for whichever checkpoint had the best rate
         at the highest SNR point. The results_fsq*/ directories are independent
         training runs of the same experiment (see FSQVAE_RUN_DIR_PATTERN):
         `sumrate` is their mean and `sumrate_std` their (population) std, per
         (config, SNR), so the plots can put error bars on FSQ/FSQM like the
         baselines'. A config missing from some runs is averaged over the runs that
         have it (std 0, i.e. no error bar, if only one does); find_fsqvae_results
         prints which ones. results_fsq_eval/ (the model for the latent-space
         figure) is not one of these runs and is not read here.

      2. DFT-codebook baseline (dft.py), under
         gmm/wmmse_results/{scenario}/dft_{rotated,unrotated}/N*/K*/N*_K*_J*_Td*_O*_TuM*/snr*.txt
         (gmm) or
         baselines/wmmse_results/{scenario}/dft_rotated|dft3/N*/J*/N*_J*_Td*_O*_TuM*_snr*.txt
         (deepmimo -- note baselines/dft.py names its unrotated variant's
         directory literally "dft3", not "dft_unrotated"; mapped to the
         `dft_unrotated` method label here for consistency with gmm).

      3. CVQ (clustered vector quantization) baseline (cvq_proj.py), under
         .../cvq_proj/N*/K*/N*_K*_J*_Td*_O*_TuM*/snr*.txt (gmm) or
         .../cvq_proj/N*/J*/N*_J*_Td*_Tu*_TuM*_snr*.txt (deepmimo).

      4. Transform-VQ (transform_vq.py), under
         .../transform_vq/N*/K*/N*_K*_J*_Td*_O*_TuM0/snr*.txt (gmm; TuM is
         always 0 here, unused by this method, kept only so the directory
         name matches the shared pattern) or
         .../transform_vq/N*/J*/N*_J*_Td*_Tu*_snr*.txt (deepmimo; no TuM at
         all in the filename -- same reason). "O"/"Tu" here is the literal
         total feedback bits spent per user, same unit as cvq_proj.py's O
         (its 2**O-codeword codebook needs exactly O bits to index) -- set
         the two equal for a fair comparison. dft.py's O is an oversampling
         factor, not a bit count, and needs converting (log2(N*O) + TuM)
         before comparing to either of the other two.

      5. VQ-VAE (main_vq.py / gmm/main_vq.py), method `vqvae` -- laid out and
         averaged exactly like FSQ-VAE in (1), under results_vq_{1,2,3,...}/
         instead of results_fsq_{1,2,3,...}/ (see VQVAE_RUN_DIR_PATTERN). Its
         Tu is the same total feedback bits per user as FSQ-VAE's.

Run this from inside precoder/analysis/ (or anywhere -- paths are resolved
against this file's location, not the working directory):

    python3 aggregate_results.py

It writes aggregated_results.pkl + .csv -- the .csv is what
make_paper_figures.py loads (a DataFrame .pkl isn't safely portable across
numpy major versions, and every column here is plain str/int/float, so
nothing is lost by preferring csv).
The .pkl is still written alongside it for a quick type-preserving look
outside a notebook.

Missing directories are skipped with a printed note rather than treated as an
error -- e.g. if you haven't run cvq_proj.py yet, that method just won't appear
in the output table. Re-run this script any time after collecting more results;
it always does a fresh full scan, so there's nothing to invalidate or clean up.
"""
import os
import re
import ast

import numpy as np
import pandas as pd

# main.py and main_vq.py evaluate at exactly these SNRs every time they save a checkpoint's
# result (see the `for npdb in [...]` loop in gmm/main.py), in this order.
FSQVAE_EVAL_SNRS_DB = [-10, -5, -2.5, 0, 2.5, 5, 10, 15]

# main.py's per-run directory name, e.g. "N8_J4_Td8_Tu20"
FSQVAE_DIR_PATTERN = re.compile(r'^N(?P<N>\d+)_J(?P<J>\d+)_Td(?P<Td>\d+)_Tu(?P<Tu>\d+)$')

# Independent training runs of the same FSQ-VAE experiment, one results directory
# per run: results_fsq_1, results_fsq_2, ... (top-level, deepmimo, and gmm/). The
# digits-only suffix keeps results_fsq_eval (the latent-space figure's model)
# from being picked up as an extra run.
FSQVAE_RUN_DIR_PATTERN = re.compile(r'^results_fsq(?:_(?P<run>\d+))?$')

# The same, for main_vq.py / gmm/main_vq.py: results_vq, results_vq_1, results_vq_2, ...
VQVAE_RUN_DIR_PATTERN = re.compile(r'^results_vq(?:_(?P<run>\d+))?$')

# per-K directory name, e.g. "K3" (gmm/ only -- deepmimo has no GMM components)
K_DIR_PATTERN = re.compile(r'^K(?P<K>\d+)$')

# the N*/K*/J* parent directories a result is nested under
N_DIR_PATTERN = re.compile(r'^N(?P<N>\d+)$')
J_DIR_PATTERN = re.compile(r'^J(?P<J>\d+)$')

# gmm/dft.py's / gmm/cvq_proj.py's per-run directory name, e.g.
# "N8_K3_J4_Td8_O32_TuM4", nested two levels deep under N*/K*/
GMM_BASELINE_DIR_PATTERN = re.compile(
    r'^N(?P<N>\d+)_K(?P<K>\d+)_J(?P<J>\d+)_Td(?P<Td>\d+)_O(?P<O>\d+)_TuM(?P<TuM>\d+)$'
)

# gmm/dft.py's / gmm/cvq_proj.py's per-SNR result file name, e.g. "snr-2.5.txt"
SNR_FILE_PATTERN = re.compile(r'^snr(?P<snr>-?\d+(?:\.\d+)?)\.txt$')

# Legacy convention actually used by every result presently on disk under
# baselines/wmmse_results/{scenario}/{cvq_proj,dft_rotated,dft3,transform_vq}/N*/J*/ --
# same layout as gmm's GMM_BASELINE_DIR_PATTERN minus the "_K{K}_" segment (no GMM
# components in this channel model). Shared by all three methods, including
# transform_vq (whose directory name still carries a placeholder _TuM0, same as gmm's).
DEEPMIMO_LEGACY_CFG_DIR_PATTERN = re.compile(
    r'^N(?P<N>\d+)_J(?P<J>\d+)_Td(?P<Td>\d+)_O(?P<O>\d+)_TuM(?P<TuM>\d+)$'
)

# baselines/{cvq_proj,dft,transform_vq}.py's *current* source instead writes one
# filename per (config, SNR) directly under N*/J*/ (config and SNR both embedded
# in the filename) -- not yet seen in any result actually on disk, but scanned
# for going forward. See find_deepmimo_baseline_results for why both are checked.
DEEPMIMO_CVQ_FILE_PATTERN = re.compile(
    r'^N(?P<N>\d+)_J(?P<J>\d+)_Td(?P<Td>\d+)_Tu(?P<O>\d+)_TuM(?P<TuM>\d+)_snr(?P<snr>-?\d+(?:\.\d+)?)\.txt$'
)
DEEPMIMO_DFT_FILE_PATTERN = re.compile(
    r'^N(?P<N>\d+)_J(?P<J>\d+)_Td(?P<Td>\d+)_O(?P<O>\d+)_TuM(?P<TuM>\d+)_snr(?P<snr>-?\d+(?:\.\d+)?)\.txt$'
)
DEEPMIMO_TRANSFORM_VQ_FILE_PATTERN = re.compile(
    r'^N(?P<N>\d+)_J(?P<J>\d+)_Td(?P<Td>\d+)_Tu(?P<O>\d+)_snr(?P<snr>-?\d+(?:\.\d+)?)\.txt$'
)

# The first line of a dft.py / cvq_proj.py / transform_vq.py result file, e.g.
# "Average sum rate: 5.07 ± 0.06"
AVG_SUM_RATE_LINE = re.compile(r'^Average sum rate:\s*([-\d.]+)\s*±\s*([-\d.]+)')

def find_fsqvae_run_dirs(parent, run_dir_pattern=FSQVAE_RUN_DIR_PATTERN):
    """The results_fsq / results_fsq_<n> directories directly under `parent`
    (see FSQVAE_RUN_DIR_PATTERN), one per independent FSQ-VAE training run.
    Pass run_dir_pattern=VQVAE_RUN_DIR_PATTERN for VQ-VAE's results_vq*."""
    if not os.path.isdir(parent):
        return []
    return [e.path for e in sorted(os.scandir(parent), key=lambda e: e.name)
            if e.is_dir() and run_dir_pattern.match(e.name)]


def find_fsqvae_results(scenario_roots, pilot_type, channel_model='gmm', method='fsqvae'):
    """
    Scan {scenario_root}/N*/K*/N*_J*_Td*_Tu*/best_sumrate.txt (gmm/'s own
    gmm/results_fsq* -- channel_model='gmm', one K* level per the GMM component
    count) or {scenario_root}/N*/N*_J*_Td*_Tu*/best_sumrate.txt (the top-level,
    deepmimo results_fsq_* -- channel_model='deepmimo', no K* level at all since
    this channel model has no GMM components) and return one row per (config,
    evaluation SNR).

    `scenario_roots` is one scenario directory per independent training run (a str
    is accepted for a single run). A config's rows carry the mean of its runs'
    sum rates as `sumrate` and their population std (np.std, ddof=0) as `sumrate_std`; a config only some runs have is
    averaged over those runs, and one with a single run gets std 0.

    `method` is the label the rows get: 'fsqvae', or 'vqvae' for main_vq.py's
    results_vq*, which share this exact layout and file format.
    """
    if isinstance(scenario_roots, str):
        scenario_roots = [scenario_roots]

    rows = []
    per_run = {}  # (N, K, J, Td, Tu) -> [sumrates of each run that has this config]

    def collect(cfg_dir, K):
        cfg_match = FSQVAE_DIR_PATTERN.match(cfg_dir.name)
        if not cfg_match:
            return
        best_sumrate_path = os.path.join(cfg_dir.path, 'best_sumrate.txt')
        if not os.path.isfile(best_sumrate_path):
            return
        with open(best_sumrate_path, 'r') as f:
            sumrates = ast.literal_eval(f.read().strip())
        if len(sumrates) != len(FSQVAE_EVAL_SNRS_DB):
            print(f'[{method}/{pilot_type}/{channel_model}] {best_sumrate_path} has {len(sumrates)} values, '
                  f'expected {len(FSQVAE_EVAL_SNRS_DB)}, skipping')
            return
        cfg = cfg_match.groupdict()
        key = (int(cfg['N']), K, int(cfg['J']), int(cfg['Td']), int(cfg['Tu']))
        per_run.setdefault(key, []).append(sumrates)

    num_roots_found = 0
    for scenario_root in scenario_roots:
        if not os.path.isdir(scenario_root):
            print(f'[{method}/{pilot_type}/{channel_model}] no results under {scenario_root}, skipping')
            continue
        num_roots_found += 1
        for n_dir in sorted(os.scandir(scenario_root), key=lambda e: e.name):
            if not n_dir.is_dir() or not N_DIR_PATTERN.match(n_dir.name):
                continue
            if channel_model == 'gmm':
                for k_dir in sorted(os.scandir(n_dir.path), key=lambda e: e.name):
                    k_match = K_DIR_PATTERN.match(k_dir.name)
                    if not k_dir.is_dir() or not k_match:
                        continue
                    for cfg_dir in sorted(os.scandir(k_dir.path), key=lambda e: e.name):
                        if cfg_dir.is_dir():
                            collect(cfg_dir, int(k_match['K']))
            else:
                # deepmimo: config directories sit directly under N*/, no K* level
                for cfg_dir in sorted(os.scandir(n_dir.path), key=lambda e: e.name):
                    if cfg_dir.is_dir():
                        collect(cfg_dir, None)

    for (N, K, J, Td, Tu), runs in per_run.items():
        sumrates = np.array(runs)  # num_runs x num_snrs
        for snr_db, mean, std in zip(FSQVAE_EVAL_SNRS_DB, sumrates.mean(0), sumrates.std(0)):
            rows.append({
                'method': method,
                'channel_model': channel_model,
                'pilot_type': pilot_type,
                'N': N,
                'K': K,
                'J': J,
                'Td': Td,
                'O': Tu,  # Tu (uplink feedback bits) plays the same role as O elsewhere
                'TuM': None,
                'snr_db': snr_db,
                'sumrate': float(mean),
                'sumrate_std': float(std),
            })

    if per_run:
        max_runs = max(len(runs) for runs in per_run.values())
        short = [(key, len(runs)) for key, runs in sorted(per_run.items(), key=lambda kv: tuple(-1 if v is None else v for v in kv[0]))
                 if len(runs) < max_runs]
        print(f'[{method}/{pilot_type}/{channel_model}] {len(per_run)} configs from {num_roots_found} run dir(s); '
              f'{len(short)} have fewer than the max {max_runs} runs (no error bar if only 1)')
        for (N, K, J, Td, Tu), n in short:
            k_str = f'K{K}_' if K is not None else ''
            print(f'    N{N}_{k_str}J{J}_Td{Td}_Tu{Tu}: {n}/{max_runs} runs')
    return rows


def find_gmm_baseline_results(wmmse_root, method_dir_name, method_label):
    """
    Scan wmmse_results/{scenario}/{method_dir_name}/N*/K*/N*_K*_J*_Td*_O*_TuM*/snr*.txt
    and return one row per (config, SNR), reading the "Average sum rate: X ± Y"
    line each dft.py/cvq_proj.py/transform_vq.py result file starts with. gmm/ only
    (see find_deepmimo_baseline_results for baselines/'s different, K-less layout).
    """
    rows = []
    root = os.path.join(wmmse_root, method_dir_name)
    if not os.path.isdir(root):
        print(f'[{method_label}] no results under {root}, skipping')
        return rows

    for n_dir in sorted(os.scandir(root), key=lambda e: e.name):
        if not n_dir.is_dir() or not N_DIR_PATTERN.match(n_dir.name):
            continue
        for k_dir in sorted(os.scandir(n_dir.path), key=lambda e: e.name):
            if not k_dir.is_dir() or not K_DIR_PATTERN.match(k_dir.name):
                continue
            for cfg_dir in sorted(os.scandir(k_dir.path), key=lambda e: e.name):
                cfg_match = GMM_BASELINE_DIR_PATTERN.match(cfg_dir.name)
                if not cfg_dir.is_dir() or not cfg_match:
                    continue
                cfg = cfg_match.groupdict()

                for snr_file in sorted(os.scandir(cfg_dir.path), key=lambda e: e.name):
                    snr_match = SNR_FILE_PATTERN.match(snr_file.name)
                    if not snr_match:
                        continue
                    with open(snr_file.path, 'r') as f:
                        first_line = f.readline()
                    rate_match = AVG_SUM_RATE_LINE.match(first_line)
                    if not rate_match:
                        print(f'[{method_label}] could not parse {snr_file.path}, skipping')
                        continue

                    rows.append({
                        'method': method_label,
                        'channel_model': 'gmm',
                        'pilot_type': 'dft',  # the WMMSE baselines only ever use the DFT pilot matrix
                        'N': int(cfg['N']),
                        'K': int(cfg['K']),
                        'J': int(cfg['J']),
                        'Td': int(cfg['Td']),
                        'O': int(cfg['O']),
                        'TuM': int(cfg['TuM']),
                        'snr_db': float(snr_match['snr']),
                        'sumrate': float(rate_match.group(1)),
                        'sumrate_std': float(rate_match.group(2)),
                    })
    return rows


def find_deepmimo_baseline_results(wmmse_root, method_dir_name, method_label, flat_file_pattern):
    """
    Scan wmmse_results/{scenario}/{method_dir_name}/N*/J*/... and return one
    row per (config, SNR). baselines/ only (no K level at all -- no GMM
    components in this channel model).

    Every result actually on disk right now was written under an *older*
    convention this repo's baselines/{cvq_proj,dft,transform_vq}.py used
    before being refactored to their current, flat-filename form: a
    N{N}_J{J}_Td{Td}_O{O}_TuM{TuM}/ config directory (matching gmm/'s naming,
    minus the K level, but sitting directly under N*/ with no intermediate
    J*/ level) containing one snr*.txt per SNR, first line "Average sum
    rate: X ± Y" -- see DEEPMIMO_LEGACY_CFG_DIR_PATTERN. The *current* source
    instead writes one N{N}_J{J}_Td{Td}_{Tu,O}{bits}[_TuM{tum}]_snr{snr}.txt
    file directly under a J*/ directory per (config, SNR), matched here by
    `flat_file_pattern` (one of DEEPMIMO_{CVQ,DFT,TRANSFORM_VQ}_FILE_PATTERN,
    since that part of the filename differs per method) -- note cvq_proj.py
    also creates J*/ directories for an unrelated reason (K-means codebook
    caching under J*/centroids, J*/tst_idxs), so a J*/ directory existing is
    not on its own evidence of the current convention having been used; it's
    simply scanned for matching flat files, which may turn up none. Both
    conventions are scanned so neither the historical data nor anything
    written going forward is missed.
    """
    rows = []
    root = os.path.join(wmmse_root, method_dir_name)
    if not os.path.isdir(root):
        print(f'[{method_label}] no results under {root}, skipping')
        return rows

    def emit(cfg, snr, rate, std):
        rows.append({
            'method': method_label,
            'channel_model': 'deepmimo',
            'pilot_type': 'dft',
            'N': int(cfg['N']),
            'K': None,
            'J': int(cfg['J']),
            'Td': int(cfg['Td']),
            'O': int(cfg['O']),
            'TuM': int(cfg['TuM']) if cfg.get('TuM') is not None else None,
            'snr_db': float(snr),
            'sumrate': rate,
            'sumrate_std': std,
        })

    def parse_first_line(path):
        with open(path, 'r') as fh:
            first_line = fh.readline()
        rate_match = AVG_SUM_RATE_LINE.match(first_line)
        if not rate_match:
            print(f'[{method_label}] could not parse {path}, skipping')
            return None
        return float(rate_match.group(1)), float(rate_match.group(2))

    for n_dir in sorted(os.scandir(root), key=lambda e: e.name):
        if not n_dir.is_dir() or not N_DIR_PATTERN.match(n_dir.name):
            continue
        for entry in sorted(os.scandir(n_dir.path), key=lambda e: e.name):
            if not entry.is_dir():
                continue

            # legacy convention: a cfg directory sitting directly under N*/
            cfg_match = DEEPMIMO_LEGACY_CFG_DIR_PATTERN.match(entry.name)
            if cfg_match:
                cfg = cfg_match.groupdict()
                for snr_file in sorted(os.scandir(entry.path), key=lambda e: e.name):
                    snr_match = SNR_FILE_PATTERN.match(snr_file.name)
                    if not snr_match:
                        continue
                    parsed = parse_first_line(snr_file.path)
                    if parsed is None:
                        continue
                    emit(cfg, snr_match['snr'], *parsed)
                continue

            # current convention: a J*/ directory holding flat per-(config,SNR) files
            # (also where cvq_proj.py's unrelated codebook-cache dirs live, which
            # simply won't match flat_file_pattern below)
            if J_DIR_PATTERN.match(entry.name):
                for f in sorted(os.scandir(entry.path), key=lambda e: e.name):
                    if not f.is_file():
                        continue
                    cfg_match = flat_file_pattern.match(f.name)
                    if not cfg_match:
                        continue
                    cfg = cfg_match.groupdict()
                    parsed = parse_first_line(f.path)
                    if parsed is None:
                        continue
                    emit(cfg, cfg['snr'], *parsed)
    return rows


COLUMNS = ['method', 'channel_model', 'pilot_type', 'N', 'K', 'J', 'Td', 'O', 'TuM',
          'snr_db', 'sumrate', 'sumrate_std']

# precoder/ itself, regardless of what directory this script is run from -- this
# script lives in precoder/analysis/, so its parent's parent is precoder/. All the
# result-tree paths below are resolved against this, not against the current
# working directory, so `python3 aggregate_results.py` works the same whether
# you run it from precoder/, precoder/analysis/, or anywhere else.
PRECODER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GMM_ROOT = os.path.join(PRECODER_ROOT, 'gmm')
BASELINES_ROOT = os.path.join(PRECODER_ROOT, 'baselines')


if __name__ == '__main__':
    GMM_SCENARIO = 'gmm'

    rows = []
    # --- gmm/ (synthetic GMM channel model) ---
    # results_fsq_1, results_fsq_2, ... are independent training runs, averaged
    # together (see find_fsqvae_results)
    gmm_fsq_run_dirs = find_fsqvae_run_dirs(GMM_ROOT)
    rows += find_fsqvae_results([os.path.join(d, GMM_SCENARIO) for d in gmm_fsq_run_dirs], pilot_type='dft')
    rows += find_fsqvae_results([os.path.join(d, 'learn_M', GMM_SCENARIO) for d in gmm_fsq_run_dirs], pilot_type='learned')

    # VQ-VAE (gmm/main_vq.py), gmm/results_vq_<n>/ -- same layout as the FSQ-VAE runs above
    gmm_vq_run_dirs = find_fsqvae_run_dirs(GMM_ROOT, VQVAE_RUN_DIR_PATTERN)
    rows += find_fsqvae_results([os.path.join(d, GMM_SCENARIO) for d in gmm_vq_run_dirs], pilot_type='dft', method='vqvae')
    rows += find_fsqvae_results([os.path.join(d, 'learn_M', GMM_SCENARIO) for d in gmm_vq_run_dirs], pilot_type='learned', method='vqvae')

    gmm_wmmse_root = os.path.join(GMM_ROOT, 'wmmse_results', GMM_SCENARIO)
    rows += find_gmm_baseline_results(gmm_wmmse_root, 'dft_rotated', 'dft_rotated')
    rows += find_gmm_baseline_results(gmm_wmmse_root, 'dft_unrotated', 'dft_unrotated')
    rows += find_gmm_baseline_results(gmm_wmmse_root, 'cvq_proj', 'cvq_proj')
    rows += find_gmm_baseline_results(gmm_wmmse_root, 'transform_vq', 'transform_vq')

    # --- baselines/ (real, ray-traced DeepMIMO channel model) ---
    # discover whatever scenario(s) have result trees on disk, rather than hardcoding
    # 'asu_campus_3p5', so a newly-added scenario is picked up automatically
    deepmimo_wmmse_roots = []
    baselines_wmmse_dir = os.path.join(BASELINES_ROOT, 'wmmse_results')
    if os.path.isdir(baselines_wmmse_dir):
        deepmimo_wmmse_roots = [e.path for e in sorted(os.scandir(baselines_wmmse_dir), key=lambda e: e.name)
                                if e.is_dir()]
    else:
        print(f'[deepmimo] no results under {baselines_wmmse_dir}, skipping')

    for deepmimo_wmmse_root in deepmimo_wmmse_roots:
        rows += find_deepmimo_baseline_results(deepmimo_wmmse_root, 'dft_rotated', 'dft_rotated', DEEPMIMO_DFT_FILE_PATTERN)
        # baselines/dft.py names its unrotated variant's directory literally "dft3" (a
        # pre-existing quirk of that script, not a typo introduced here) -- map it to
        # the same 'dft_unrotated' method label gmm/ uses for the equivalent method
        rows += find_deepmimo_baseline_results(deepmimo_wmmse_root, 'dft3', 'dft_unrotated', DEEPMIMO_DFT_FILE_PATTERN)
        rows += find_deepmimo_baseline_results(deepmimo_wmmse_root, 'cvq_proj', 'cvq_proj', DEEPMIMO_CVQ_FILE_PATTERN)
        rows += find_deepmimo_baseline_results(deepmimo_wmmse_root, 'transform_vq', 'transform_vq', DEEPMIMO_TRANSFORM_VQ_FILE_PATTERN)

    # FSQ-VAE (main.py run against a deepmimo scenario), under the top-level
    # results_fsq_1/, results_fsq_2/, ... (one per independent training run) -- kept
    # entirely separate from gmm/'s own results_fsq*/ above: different root
    # (PRECODER_ROOT, not GMM_ROOT), no K* level (no GMM components in this channel
    # model), and scenario(s) discovered the same way as deepmimo_wmmse_roots above
    # rather than hardcoded to 'asu_campus_3p5'
    deepmimo_fsq_run_dirs = find_fsqvae_run_dirs(PRECODER_ROOT)
    if not deepmimo_fsq_run_dirs:
        print(f'[deepmimo] no results_fsq_<n>/ directories under {PRECODER_ROOT}, skipping FSQ-VAE')
    deepmimo_fsq_scenarios = sorted({e.name for d in deepmimo_fsq_run_dirs for e in os.scandir(d)
                                     if e.is_dir() and e.name != 'learn_M'})

    for scenario in deepmimo_fsq_scenarios:
        rows += find_fsqvae_results([os.path.join(d, scenario) for d in deepmimo_fsq_run_dirs],
                                    pilot_type='dft', channel_model='deepmimo')
        rows += find_fsqvae_results([os.path.join(d, 'learn_M', scenario) for d in deepmimo_fsq_run_dirs],
                                    pilot_type='learned', channel_model='deepmimo')

    # VQ-VAE (main_vq.py run against a deepmimo scenario), top-level results_vq_<n>/ --
    # discovered and read exactly like the FSQ-VAE runs above
    deepmimo_vq_run_dirs = find_fsqvae_run_dirs(PRECODER_ROOT, VQVAE_RUN_DIR_PATTERN)
    if not deepmimo_vq_run_dirs:
        print(f'[deepmimo] no results_vq_<n>/ directories under {PRECODER_ROOT}, skipping VQ-VAE')
    deepmimo_vq_scenarios = sorted({e.name for d in deepmimo_vq_run_dirs for e in os.scandir(d)
                                    if e.is_dir() and e.name != 'learn_M'})

    for scenario in deepmimo_vq_scenarios:
        rows += find_fsqvae_results([os.path.join(d, scenario) for d in deepmimo_vq_run_dirs],
                                    pilot_type='dft', channel_model='deepmimo', method='vqvae')
        rows += find_fsqvae_results([os.path.join(d, 'learn_M', scenario) for d in deepmimo_vq_run_dirs],
                                    pilot_type='learned', channel_model='deepmimo', method='vqvae')

    # pass `columns=` explicitly so this still has the right (empty) columns to
    # sort/group by below even if no result trees were found above -- an empty
    # `pd.DataFrame([])` has no columns at all, which would make the
    # sort_values/groupby calls below fail with a KeyError instead of just
    # producing an empty (but well-formed) table.
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.drop_duplicates(subset=['method', 'channel_model', 'pilot_type', 'N', 'K', 'J', 'Td', 'O', 'TuM', 'snr_db'], keep='last')
    df = df.sort_values(['channel_model', 'method', 'N', 'K', 'J', 'Td', 'O', 'TuM', 'snr_db']).reset_index(drop=True)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(out_dir, 'aggregated_results.pkl')
    csv_path = os.path.join(out_dir, 'aggregated_results.csv')
    df.to_pickle(pkl_path)
    df.to_csv(csv_path, index=False)

    print(f'\nWrote {len(df)} rows to:\n  {pkl_path}\n  {csv_path}')
    if df.empty:
        print('\nTable is empty -- no result directories were found under any of the paths logged above.')
    else:
        print('\nRows per (channel_model, method):')
        print(df.groupby(['channel_model', 'method']).size().to_string())
