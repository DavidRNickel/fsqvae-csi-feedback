"""
Clustering-based vector quantization (CVQ) + WMMSE baseline.

--num-uts/--Td/--num-feedback-bits/--snr-db all take one or more values and are
swept internally (nested loops, outermost to innermost in that order) --
instead of relying on a
shell script to invoke this once per combination. That keeps numba's JIT warm
across an entire sweep (only paid once per process) instead of once per
combination.

--num-uts and --num-feedback-bits are the outer two loops (in that order)
because the K-means codebook and its cached train/test split only depend on
(J, Tu) -- see the codebook cache path below -- so grouping the sweep this way
means each codebook is built (or loaded from disk) exactly once per (J, Tu)
pair, no matter how many --Td/--snr-db values are swept underneath it.
"""

import argparse
import os
from pathlib import Path
from distutils.util import strtobool

import numpy as np
from tqdm import tqdm

from scipy.linalg import pinv
from sklearn.cluster import MiniBatchKMeans

from wmmse import wmmse, sum_rate
from utils import estimate_channel_mp_pinv, estimate_channel_corr_mtx
from utils_dft import quantize_channels_rotated


if __name__=='__main__':
    # TODO: see what happens if you don't normalize the codebook to unit norm

    parser = argparse.ArgumentParser(description='PyTorch code for ...')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 10000), metavar='N', help='random seed (default: 456)')
    parser.add_argument('--scenario', type=str, default='asu_campus_3p5', help='deepmimo scenario to use for channel generation')
    parser.add_argument('--num-bs-ant', type=int, default=64, help='number of antennae at the BS')
    parser.add_argument('--num-uts', type=int, nargs='+', default=[8], help='list of J: number of user terminals to sweep')
    parser.add_argument('--snr-db', type=float, nargs='+', default=[0.], help='list of SNRs in dB to sweep')
    parser.add_argument('--Td', type=int, nargs='+', default=[14], help='list of numbers of pilot symbols to sweep')
    parser.add_argument('--num-feedback-bits', type=int, nargs='+', default=[12], help='list of numbers of feedback bits for channel quantization to sweep')
    parser.add_argument('--max-wmmse-iter', type=int, default=25, help='max number of iterations of WMMSE algorithm')
    parser.add_argument('--num-codebooks-to-test', type=int, default=1,
                        help='number of independently-fit K-means codebooks to average sum rate over')
    parser.add_argument('--use-true-channel', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--channel-inversion-type', type=str, default='lmmse', choices=['lmmse', 'mppinv'], help='type of channel inversion to use')
    parser.add_argument('--num-magnitude-bits', type=int, default=4, help='number of bits for quantizing channel magnitudes')
    parser.add_argument('--dirnum', type=str, default='', help='suffix appended to the results cache directory name')

    args = parser.parse_args()

    assert all(tu <= 20 for tu in args.num_feedback_bits), "Too many feedback bits may lead to excessive memory usage. Please reduce the number of feedback bits."
    assert all(td <= args.num_bs_ant for td in args.Td), "Number of pilot symbols cannot exceed the number of BS antennas."
    args.cache = 'experiments'

    loadfile = f'../channel_data/{args.scenario}/{args.num_bs_ant}x1/channels.npy'
    loadfile_corr = f'../channel_data/{args.scenario}/{args.num_bs_ant}x1/corr_matrices.npy'
    channels = np.load(loadfile).squeeze()
    corr_matrices = np.load(loadfile_corr).astype(np.complex128)

    noise_pwr = 1
    num_tst_groups = 500

    # trigger numba JIT compilation once up front so it doesn't happen mid-sweep
    print("Warming up WMMSE solver...")
    max_j = max(args.num_uts)
    mock_H = np.ones((max_j, 1, args.num_bs_ant), dtype=np.complex128)
    _ = wmmse(H=mock_H, P_max=1.0, sigma_2=np.ones((max_j,), dtype=np.complex128),
              alphas=np.ones((max_j,), dtype=np.complex128), max_wmmse_iter=1)
    print("Warm-up complete.")

    for ut in args.num_uts:
        for tu in args.num_feedback_bits:
            assert len(channels) >= 2**tu, "Not enough channels for the specified number of feedback bits."

            ROOT_DIR = f'wmmse_results{args.dirnum}/{args.scenario}/cvq_proj/N{args.num_bs_ant}/'
            savedir = os.path.join(ROOT_DIR, f'J{ut}')
            os.makedirs(savedir, exist_ok=True)

            num_tst_idxs = num_tst_groups * ut
            num_runs = args.num_codebooks_to_test

            # One codebook (and its own held-out test split) is fit or loaded per
            # --num-codebooks-to-test index up front, cached separately per index, and reused
            # across the whole Td/SNR sweep below -- so each codebook is only built once.
            codebooks, tst_idxs_list = [], []
            for i in range(num_runs):
                centroid_path = Path(f'{ROOT_DIR}/J{ut}/centroids/Tu{tu}_cb{i}.npy')
                tst_idxs_path = Path(f'{ROOT_DIR}/J{ut}/tst_idxs/Tu{tu}_cb{i}.npy')

                cached_tst_idxs = np.load(tst_idxs_path) if tst_idxs_path.is_file() else None
                cache_valid = (
                    centroid_path.is_file()
                    and cached_tst_idxs is not None
                    and len(cached_tst_idxs) == num_tst_idxs
                )

                if cache_valid:
                    print(f'Loading codebook and test indices (cb {i})...')
                    codebook = np.load(centroid_path).astype(np.complex128)
                    tst_idxs = cached_tst_idxs
                else:
                    print(f'Creating codebook (cb {i})')
                    cb_seed = args.seed + i
                    rng = np.random.default_rng(seed=cb_seed)
                    full_idxs = np.arange(len(channels))
                    assert (len(full_idxs) - num_tst_idxs) >= 2**tu, 'Not enough training channels for the specified number of feedback bits.'
                    trn_idxs = rng.choice(full_idxs, size=len(full_idxs)-num_tst_idxs, replace=False)

                    tst_idxs = np.setdiff1d(full_idxs, trn_idxs)[:num_tst_idxs]
                    assert len(tst_idxs) == num_tst_idxs, 'Not enough testing channels.'
                    chans_trn = channels[trn_idxs]

                    init = 'k-means++'
                    max_kmeans_iter = 300
                    tol = 1e-5
                    batch_size = min(4096, len(chans_trn))
                    kmeans = MiniBatchKMeans(n_clusters=2**tu, init=init, n_init="auto", batch_size=batch_size,
                                             max_iter=max_kmeans_iter, max_no_improvement=20, random_state=cb_seed, tol=tol)
                    y_kmeans = kmeans.fit_predict(np.hstack((chans_trn.real, chans_trn.imag)))
                    codebook_split = kmeans.cluster_centers_
                    codebook_split /= np.linalg.norm(codebook_split, axis=1, keepdims=True) # normalize to unit norm
                    codebook = (codebook_split[:,:args.num_bs_ant] + 1j*codebook_split[:,args.num_bs_ant:]).astype(np.complex128)
                    os.makedirs(centroid_path.parent, exist_ok=True)
                    np.save(centroid_path, codebook)
                    os.makedirs(tst_idxs_path.parent, exist_ok=True)
                    np.save(tst_idxs_path, tst_idxs)

                codebook /= np.linalg.norm(codebook, axis=1, keepdims=True) # normalize to unit norm
                codebooks.append(codebook)
                tst_idxs_list.append(tst_idxs)

            for td in args.Td:
                for snr in args.snr_db:
                    # make the pilot matrix
                    transmit_pwr = 10**(snr/10)*noise_pwr
                    pilot_matrix = np.fft.fft(np.eye(args.num_bs_ant))[:td] # DFT matrix for pilot symbols
                    pilot_matrix = np.sqrt(transmit_pwr) * (pilot_matrix / np.linalg.norm(pilot_matrix, axis=1, keepdims=True))
                    moore_penrose_pinv = pinv(pilot_matrix)
                    sigma_2 = noise_pwr * np.ones((ut,), dtype=np.complex128)

                    avg_sumrates = np.empty(num_runs)
                    nans, infs, large_vals = [], [], []
                    print(f'\nRunning WMMSE with N={args.num_bs_ant}, SNR={snr} dB, Tu={tu}, Td={td}, J={ut}, max WMMSE iterations={args.max_wmmse_iter}, use true channel={args.use_true_channel}')
                    for i in range(num_runs):
                        codebook = codebooks[i]
                        c_idxs = np.random.permutation(tst_idxs_list[i])
                        chans = channels[c_idxs]
                        corr_mtxs = corr_matrices[c_idxs].reshape(-1, ut, args.num_bs_ant, args.num_bs_ant)
                        corr_mtxs_sqrt = np.tile(np.eye(args.num_bs_ant), (num_tst_idxs, 1, 1))

                        tot_rates = []
                        H = chans.reshape(-1, ut, args.num_bs_ant).astype(np.complex128)
                        if not args.use_true_channel:
                            if args.channel_inversion_type == 'lmmse':
                                H_quant = estimate_channel_corr_mtx(H, corr_mtxs, pilot_matrix, noise_pwr).transpose(0,2,1)
                            else:
                                H_quant = estimate_channel_mp_pinv(H, pilot_matrix, moore_penrose_pinv, noise_pwr).transpose(0,2,1)
                            H_quant = quantize_channels_rotated(H_quant.reshape(-1, args.num_bs_ant),
                                                                codebook, corr_mtxs_sqrt,
                                                                magnitude_params=(args.num_magnitude_bits, 1.4*np.sqrt(args.num_bs_ant)),
                                                                tree=None)
                            H_quant = H_quant.reshape(-1, ut, args.num_bs_ant)
                        else:
                            H_quant = H

                        # k=num base stations, t=num transmit antennas, r=num receive antennas, i=num uts
                        for hdx, h in enumerate(tqdm(H, desc=f"Processing channels (J={ut}, Tu={tu}, Td={td}, SNR={snr})")):
                            # estimate and quantize the channels
                            h_est = H_quant[hdx].copy() # if not args.use_true_channel this is just H[hdx]

                            # calculate the beamforming vectors then do the sumrate
                            try:
                                v = wmmse(H=h_est[:,None,:],
                                        P_max=transmit_pwr,
                                        sigma_2=sigma_2,
                                        alphas=np.ones((ut,), dtype=np.complex128),
                                        max_wmmse_iter=args.max_wmmse_iter)
                                actual_sumrate = sum_rate(h[:,None,:], v, sigma_2).sum()
                                tot_rates.append(actual_sumrate)
                            except np.linalg.LinAlgError:
                                continue

                        # calculate the sum rate for this codebook
                        tot_rates = np.array(tot_rates)
                        tr = np.nanmean(tot_rates)
                        print('average rate:', tr)
                        avg_sumrates[i] = tr

                        # get the bad entries (NaNs, Infs, very large values)
                        # there shouldn't be any, but this is here for completeness
                        nnans = np.sum(np.isnan(tot_rates))
                        ninfs = np.sum(np.isinf(tot_rates))
                        nlarge = np.sum(tot_rates>250)
                        nans.append(nnans.item())
                        infs.append(ninfs.item())
                        large_vals.append(nlarge.item())

                    print(f'Average over codebooks: {avg_sumrates.mean()} ± {np.nanstd(avg_sumrates)}')
                    final = {'avg_sumrate': avg_sumrates.mean(), 'std_sumrate': np.nanstd(avg_sumrates), 'tot_rates': tot_rates}
                    with open(f'{savedir}/N{args.num_bs_ant}_J{ut}_Td{td}_Tu{tu}_TuM{args.num_magnitude_bits}_snr{snr}.txt', 'w') as f:
                        f.write(f'Average sum rate: {final["avg_sumrate"]:.2f} ± {final["std_sumrate"]:.4f}\n')
                        f.write(f'Per-codebook average sumrates: {avg_sumrates}\n')
                        f.write(f'NaNs: {nans}\n')
                        f.write(f'Infs: {infs}\n')
                        f.write(f'Large vals: {large_vals}\n')
                        f.write(f'Total rates: {final["tot_rates"]}\n')
