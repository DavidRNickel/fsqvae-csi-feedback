"""
Clustering-based vector quantization (CVQ) baseline for the GMM channel model.

Adapted from precoder/baselines/cvq_proj.py (the ASU-campus version), replacing
the fixed real-channel dataset with on-the-fly GMM sampling. The key reason CVQ
is worth having for this channel model (see the DFT-scheme investigation): the
DFT/DVQ codebook is a 1-parameter angular curve in C^N, which is a poor match for
this GMM covariance's full-rank spatial structure (eigenvalues spread across the
whole space, not concentrated in one direction the way a physical LOS steering
vector is). CVQ instead fits its codebook directly to sampled channel vectors via
K-means, so it isn't limited to any particular manifold -- it should track the
achievable-rate ceiling much more closely for this channel model than the DFT
scheme does.

Quantization here intentionally applies no per-user rotation (corr_mtxs_sqrt is
the identity), matching the original ASU cvq_proj.py -- the K-means codebook
already spans the raw channel space directly, so no rotation is needed.

--num-gmm-components/--num-feedback-bits/--num-uts/--Td/--snr-db all take one or
more values and are swept internally (nested loops), matching cvq_proj.py's
convention in precoder/baselines. --num-gmm-components and --num-feedback-bits
are the outer two loops (in that order) because the K-means codebook is cached
by (N, K, Tu) -- see the codebook cache path below -- so grouping the sweep
this way means each codebook is built (or loaded from disk) exactly once per
(K, Tu) pair, no matter how many --num-uts/--Td/--snr-db values are swept
underneath it.
"""

import argparse
import os
from pathlib import Path
from distutils.util import strtobool

import numpy as np
import torch
from tqdm import tqdm
from scipy.linalg import pinv
from sklearn.cluster import MiniBatchKMeans

from precoding.wmmse import wmmse, sum_rate
from precoding.utils import estimate_channel_corr_mtx, estimate_channel_mp_pinv
from precoding.utils_dft import quantize_channels_rotated


def sample_gmm_channels(decomps, K, num_groups, num_uts, N, rng):
    comp_idxs = rng.integers(0, K, size=(num_groups, num_uts))
    z = np.sqrt(0.5) * (rng.standard_normal((num_groups, num_uts, N))
                         + 1j * rng.standard_normal((num_groups, num_uts, N)))
    L = decomps[comp_idxs]
    return np.einsum('gjnm,gjm->gjn', L, z)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CVQ baseline for the GMM channel model')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 10000))
    parser.add_argument('--scenario', type=str, default='gmm', help='channel model scenario name (used for output paths)')
    parser.add_argument('--num-bs-ant', type=int, default=8, help='number of antennae at the BS')
    parser.add_argument('--num-uts', type=int, nargs='+', default=[8], help='list of user-terminal counts to sweep')
    parser.add_argument('--num-gmm-components', type=int, nargs='+', default=[1], help='list of K: number of components in the GMM channel model to sweep')
    parser.add_argument('--snr-db', type=float, nargs='+', default=[15.], help='list of SNRs in dB to sweep')
    parser.add_argument('--Td', type=int, nargs='+', default=[8], help='list of numbers of pilot symbols to sweep')
    parser.add_argument('--num-feedback-bits', type=int, nargs='+', default=[12], help='list of log2(codebook size) values to sweep')
    parser.add_argument('--num-magnitude-bits', type=int, default=4, help='number of bits for quantizing channel magnitudes')
    parser.add_argument('--max-wmmse-iter', type=int, default=25, help='max number of iterations of WMMSE algorithm')
    parser.add_argument('--num-codebooks-to-test', type=int, default=1,
                        help='number of independently-fit K-means codebooks to average sum rate over')
    parser.add_argument('--use-true-channel', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--channel-inversion-type', type=str, default='lmmse', choices=['lmmse', 'mppinv'])
    parser.add_argument('--num-kmeans-train-samples', type=int, default=200000,
                        help='number of i.i.d. GMM channel draws used to fit the K-means codebook (only used when no cached codebook exists)')
    parser.add_argument('--kmeans-samples-per-cluster', type=int, default=30,
                        help='minimum number of training samples per codebook cluster; --num-kmeans-train-samples is scaled up '
                             'to satisfy this at large --num-feedback-bits values, keeping codebook quality roughly constant across Tu')
    parser.add_argument('--kmeans-batch-size', type=int, default=4096,
                        help='mini-batch size for MiniBatchKMeans fitting; bounds per-iteration cost independent of the training set size')
    args = parser.parse_args()

    assert all(td <= args.num_bs_ant for td in args.Td), "Number of pilot symbols cannot exceed the number of BS antennas."
    assert all(args.num_kmeans_train_samples >= 2**tu for tu in args.num_feedback_bits), \
        "Not enough K-means training samples for the specified number of feedback bits."

    N = args.num_bs_ant

    STATS_DIR = os.path.join('channel_stats', f'N{N}')
    all_channel_covs = torch.load(os.path.join(STATS_DIR, 'cov.pt'), map_location='cpu').numpy().astype(np.complex128)
    all_channel_decomps = torch.load(os.path.join(STATS_DIR, 'decomp.pt'), map_location='cpu').numpy().astype(np.complex128)

    rng = np.random.default_rng(seed=args.seed)

    # trigger numba JIT compilation once up front so it doesn't happen mid-sweep
    warmup_j = args.num_uts[0]
    mock_H = np.ones((warmup_j, 1, N), dtype=np.complex128)
    _ = wmmse(H=mock_H, P_max=1.0, sigma_2=np.ones((warmup_j,), dtype=np.complex128),
              alphas=np.ones((warmup_j,), dtype=np.complex128), max_wmmse_iter=1)

    noise_pwr = 1
    num_tst_groups = 500
    num_runs = args.num_codebooks_to_test

    for K in args.num_gmm_components:
        channel_decomps = all_channel_decomps[:K]
        avg_corr_mtx = all_channel_covs[:K].mean(axis=0)

        for tu in args.num_feedback_bits:
            # codebook caching keyed only by (N, K, num_feedback_bits) -- the K-means codebook
            # clusters individual per-user channel vectors and doesn't depend on J, Td, or SNR.
            # One codebook is fit (or loaded) per --num-codebooks-to-test index up front, cached
            # separately per index, and reused across the whole J/Td/SNR sweep below -- so each
            # codebook is only ever built once, no matter how many J/Td/SNR combinations use it.
            CB_ROOT = f'wmmse_results/{args.scenario}/cvq_proj/N{N}/K{K}'

            codebooks = []
            for i in range(num_runs):
                centroid_path = Path(f'{CB_ROOT}/centroids/Tu{tu}_cb{i}.npy')

                if centroid_path.is_file():
                    print(f'Loading cached K-means codebook (cb {i})...')
                    codebook = np.load(centroid_path).astype(np.complex128)
                else:
                    cb_seed = args.seed + i
                    cb_rng = np.random.default_rng(seed=cb_seed)
                    num_train_samples = max(args.num_kmeans_train_samples, args.kmeans_samples_per_cluster * 2**tu)
                    print(f'Fitting K-means codebook (cb {i}, {2**tu} clusters) '
                          f'on {num_train_samples} sampled GMM channels...')
                    chans_trn = sample_gmm_channels(channel_decomps, K, num_train_samples, 1, N, cb_rng).reshape(-1, N)

                    kmeans = MiniBatchKMeans(n_clusters=2**tu, init='k-means++', n_init='auto',
                                             batch_size=min(args.kmeans_batch_size, num_train_samples),
                                             max_iter=300, max_no_improvement=20, random_state=cb_seed, tol=1e-5)
                    kmeans.fit(np.hstack((chans_trn.real, chans_trn.imag)))
                    codebook_split = kmeans.cluster_centers_
                    codebook_split /= np.linalg.norm(codebook_split, axis=1, keepdims=True)
                    codebook = (codebook_split[:, :N] + 1j*codebook_split[:, N:]).astype(np.complex128)

                    os.makedirs(centroid_path.parent, exist_ok=True)
                    np.save(centroid_path, codebook)

                codebook /= np.linalg.norm(codebook, axis=1, keepdims=True)
                codebooks.append(codebook)

            for J in args.num_uts:
                sigma_2 = noise_pwr * np.ones((J,), dtype=np.complex128)
                # identity rotation: the K-means codebook already lives in the raw channel space
                corr_mtxs_sqrt = np.tile(np.eye(N), (num_tst_groups*J, 1, 1))

                for td in args.Td:
                    for snr in args.snr_db:
                        transmit_pwr = 10**(snr/10) * noise_pwr
                        pilot_matrix = np.fft.fft(np.eye(N))[:td]
                        pilot_matrix = np.sqrt(transmit_pwr) * (pilot_matrix / np.linalg.norm(pilot_matrix, axis=1, keepdims=True))
                        moore_penrose_pinv = pinv(pilot_matrix)

                        avg_sumrates = np.empty(num_runs)
                        nans, infs, large_vals = [], [], []
                        print(f'\nRunning CVQ+WMMSE with N={N}, K={K}, SNR={snr} dB, Tu={tu}, '
                              f'TuM={args.num_magnitude_bits}, Td={td}, J={J}, max WMMSE iterations={args.max_wmmse_iter}, '
                              f'use true channel={args.use_true_channel}')

                        for i in range(num_runs):
                            codebook = codebooks[i]
                            H = sample_gmm_channels(channel_decomps, K, num_tst_groups, J, N, rng)
                            # LMMSE gain needs Cov(channels.conj()) = conj(avg_corr_mtx); see gmm/dft.py for the derivation
                            corr_mtxs = np.tile(avg_corr_mtx.conj(), (num_tst_groups, J, 1, 1))

                            tot_rates = []
                            if not args.use_true_channel:
                                if args.channel_inversion_type == 'lmmse':
                                    H_quant = estimate_channel_corr_mtx(H, corr_mtxs, pilot_matrix, noise_pwr).transpose(0, 2, 1)
                                else:
                                    H_quant = estimate_channel_mp_pinv(H, pilot_matrix, moore_penrose_pinv, noise_pwr).transpose(0, 2, 1)
                                H_quant = quantize_channels_rotated(H_quant.reshape(-1, N), codebook, corr_mtxs_sqrt,
                                                                    magnitude_params=(args.num_magnitude_bits, 1.4*np.sqrt(N)), tree=None)
                                H_quant = H_quant.reshape(-1, J, N)
                            else:
                                H_quant = H

                            for hdx, h in enumerate(tqdm(H, desc=f"Processing channels (K={K}, J={J}, Tu={tu}, Td={td}, SNR={snr})")):
                                h_est = H_quant[hdx].copy()
                                try:
                                    v = wmmse(H=h_est[:, None, :], P_max=transmit_pwr, sigma_2=sigma_2,
                                            alphas=np.ones((J,), dtype=np.complex128), max_wmmse_iter=args.max_wmmse_iter)
                                    actual_sumrate = sum_rate(h[:, None, :], v, sigma_2).sum()
                                    tot_rates.append(actual_sumrate)
                                except np.linalg.LinAlgError:
                                    continue

                            tot_rates = np.array(tot_rates)
                            tr = np.nanmean(tot_rates)
                            print('average rate:', tr)
                            avg_sumrates[i] = tr

                            nnans = np.sum(np.isnan(tot_rates))
                            ninfs = np.sum(np.isinf(tot_rates))
                            nlarge = np.sum(tot_rates > 250)
                            nans.append(nnans.item())
                            infs.append(ninfs.item())
                            large_vals.append(nlarge.item())

                        print(f'Average over codebooks: {avg_sumrates.mean()} ± {np.nanstd(avg_sumrates)}')
                        final = {'avg_sumrate': avg_sumrates.mean(), 'std_sumrate': np.nanstd(avg_sumrates), 'tot_rates': tot_rates}
                        config_tag = f'N{N}_K{K}_J{J}_Td{td}_O{tu}_TuM{args.num_magnitude_bits}'
                        savedir = f'wmmse_results/{args.scenario}/cvq_proj/N{N}/K{K}/{config_tag}'
                        os.makedirs(savedir, exist_ok=True)

                        with open(f'{savedir}/snr{snr}.txt', 'w') as f:
                            f.write(f'Average sum rate: {final["avg_sumrate"]:.2f} ± {final["std_sumrate"]:.4f}\n')
                            f.write(f'Per-codebook average sumrates: {avg_sumrates}\n')
                            f.write(f'NaNs: {nans}\n')
                            f.write(f'Infs: {infs}\n')
                            f.write(f'Large vals: {large_vals}\n')
                            f.write(f'Total rates: {final["tot_rates"]}\n')
