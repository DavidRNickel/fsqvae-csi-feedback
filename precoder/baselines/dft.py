# """
# This file was primarily written by hand. Some of the logic such as introducing
# perturbations was ideated and implemented with the help of Google AI
# """

"""
This script was debugged with the help of an LLM (although it kept
trying to get me to do some stuff that is wrong. Go figure...)
"""

import argparse
import sys, os, time
from tqdm import tqdm
from distutils.util import strtobool
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from scipy.spatial import KDTree
from wmmse import wmmse, sum_rate
from utils import estimate_channel_mp_pinv, estimate_channel_corr_mtx
from utils_dft import generate_dft_codebook, quantize_channels_rotated

if __name__=='__main__':
    save_results = False
    parser = argparse.ArgumentParser(description='PyTorch code for ...')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 10000), metavar='N', help='random seed (default: 456)')
    parser.add_argument('--scenario', type=str, default='asu_campus_3p5', help='deepmimo scenario to use for channel generation')
    parser.add_argument('--num-bs-ant', type=int, default=64, help='number of antennae at the BS')
    parser.add_argument('--num-uts', type=int, nargs='+', default=[8], help='list of J: number of user terminals to sweep')
    parser.add_argument('--snr-db', type=float, nargs='+', default=[0.], help='list of SNRs in dB to sweep')
    parser.add_argument('--Td', type=int, nargs='+', default=[14], help='list of numbers of pilot symbols to sweep')
    parser.add_argument('--max-wmmse-iter', type=int, default=25, help='max number of iterations of WMMSE algorithm')
    parser.add_argument('--num-codebooks-to-test', type=int, default=1, help='number of codebooks to test')
    parser.add_argument('--use-true-channel', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--rotate-codebook', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--channel-inversion-type', type=str, default='lmmse', choices=['lmmse', 'mppinv'], help='type of channel inversion to use')
    parser.add_argument('--num-feedback-bits', type=int, nargs='+', default=[12], help='list of numbers of feedback bits for channel angle quantization to sweep')
    parser.add_argument('--num-magnitude-bits', type=int, default=4, help='number of bits for quantizing channel magnitudes')
    args = parser.parse_args()

    # assert args.num_feedback_bits <= 20, "Too many feedback bits may lead to excessive memory usage. Please reduce the number of feedback bits."
    assert all(td <= args.num_bs_ant for td in args.Td), "Number of pilot symbols cannot exceed the number of BS antennas."
    args.cache = 'experiments'
    loadfile = f'../channel_data/{args.scenario}/{args.num_bs_ant}x1/channels.npy'
    loadfile_corr = f'../channel_data/{args.scenario}/{args.num_bs_ant}x1/corr_matrices.npy'
    loadfile_corr_sqrt = f'../channel_data/{args.scenario}/{args.num_bs_ant}x1/corr_matrices_sqrt.npy'

    channels = np.load(loadfile).squeeze()
    corr_matrices = np.load(loadfile_corr).astype(np.complex128)
    corr_matrices_sqrt = np.load(loadfile_corr_sqrt).astype(np.complex128)


    num_runs = args.num_codebooks_to_test
    avg_sumrates = np.empty(num_runs)

    print("Warming up WMMSE solver...")
    max_j = max(args.num_uts)
    mock_H = np.ones((max_j, 1, args.num_bs_ant), dtype=np.complex128)
    _ = wmmse(H=mock_H, P_max=1.0, sigma_2=np.ones((max_j,), dtype=np.complex128), alphas=np.ones(max_j, dtype=np.complex128), max_wmmse_iter=1)
    print("Warm-up complete.")

    MIN_RATE_THRESHOLD = 0. # artifact of earlier version
    for tu in args.num_feedback_bits:
        codebook = generate_dft_codebook(num_antennas=args.num_bs_ant, oversampling_factor=tu)
        for j in args.num_uts:
            for td in args.Td:
                for snr in args.snr_db:
                    noise_pwr = 1
                    transmit_pwr = 10**(snr/10)*noise_pwr
                    pilot_matrix = np.fft.fft(np.eye(args.num_bs_ant))[:td] # DFT matrix for pilot symbols
                    pilot_matrix = np.sqrt(transmit_pwr) * (pilot_matrix / np.linalg.norm(pilot_matrix, axis=1, keepdims=True))
                    moore_penrose_pinv = pinv(pilot_matrix)

                    sigma_2 = noise_pwr * np.ones((j,), dtype=np.complex128)
                    num_tst_groups = 500
                    num_tst_idxs = num_tst_groups * j

                    rng = np.random.default_rng(seed=args.seed)
                    full_idxs = np.arange(len(channels))
                    nans, infs, large_vals = [], [], []

                    print(f'\nRunning WMMSE with N={args.num_bs_ant}, SNR={snr} dB, O={tu}, TuM={args.num_magnitude_bits}, Td={td}, J={j}, max WMMSE iterations={args.max_wmmse_iter}, use true channel={args.use_true_channel}')

                    successful_runs = 0
                    while successful_runs < num_runs:
                        # load in the channel and correlation matrices
                        idxs = rng.choice(full_idxs, size=num_tst_idxs, replace=False)
                        chans = channels[idxs]
                        corr_mtxs = corr_matrices[idxs]
                        if args.rotate_codebook:
                            corr_mtxs_sqrt = corr_matrices_sqrt[idxs]
                        else:
                            corr_mtxs_sqrt = np.tile(np.eye(args.num_bs_ant), (num_tst_idxs, 1, 1))

                        H = chans.reshape(-1, j, args.num_bs_ant).astype(np.complex128)
                        corr_mtxs = corr_mtxs.reshape(-1, j, args.num_bs_ant, args.num_bs_ant)
                        corr_mtxs_sqrt = corr_mtxs_sqrt.reshape(-1, args.num_bs_ant, args.num_bs_ant)
                        tot_rates = []

                        # quantize the channels (or get perfect CSI when desired)
                        if not args.use_true_channel:
                            if args.channel_inversion_type == 'lmmse':
                                H_quant = estimate_channel_corr_mtx(H, corr_mtxs, pilot_matrix, noise_pwr).transpose(0,2,1)
                            else:
                                H_quant = estimate_channel_mp_pinv(H, pilot_matrix, moore_penrose_pinv, noise_pwr).transpose(0,2,1)
                            H_quant = quantize_channels_rotated(H_quant.reshape(-1, args.num_bs_ant), codebook, corr_mtxs_sqrt, magnitude_params=(args.num_magnitude_bits, 1.4*np.sqrt(args.num_bs_ant)))
                            H_quant = H_quant.reshape(-1, j, args.num_bs_ant)
                        else:
                            H_quant = H.reshape(-1, j, args.num_bs_ant)

                        # k=num base stations, t=num transmit antennas, r=num receive antennas, i=num uts
                        for hdx, h in enumerate(tqdm(H, desc="Processing channels")):
                            # estimate and quantize the channels
                            h_est = H_quant[hdx].copy() # if not args.use_true_channel this is just H[hdx]

                            # calculate the beamforming vectors then do the sumrate
                            try:
                                v = wmmse(H=h_est[:,None,:], P_max=transmit_pwr, sigma_2=sigma_2, alphas=np.ones((j,), dtype=np.complex128), max_wmmse_iter=args.max_wmmse_iter)
                                actual_sumrate = sum_rate(h[:,None,:], v, sigma_2).sum()
                                tot_rates.append(actual_sumrate)

                            # the try/except clause is here because, especially at low number of feedback bits,
                            # there are channels that get quantized to the same place, causing errors
                            except np.linalg.LinAlgError:
                                continue

                        # calculate the sum rate for this codebook
                        tot_rates = np.array(tot_rates)
                        tr = np.nanmean(tot_rates)

                        if tr < MIN_RATE_THRESHOLD:
                            print(f"Skipping run. Average rate {tr:.4f} is lower than threshold {MIN_RATE_THRESHOLD:.4f}.")
                            continue

                        print('average rate:', tr)
                        avg_sumrates[successful_runs] = tr
                        successful_runs += 1

                        # get the bad entries (NaNs, Infs, very large values)
                        # there shouldn't be any, but this is here for completeness
                        nnans = np.sum(np.isnan(tot_rates))
                        ninfs = np.sum(np.isinf(tot_rates))
                        nlarge = np.sum(tot_rates>250)
                        nans.append(nnans.item())
                        infs.append(ninfs.item())
                        large_vals.append(nlarge.item())

                    print(f'Overall average rate: {avg_sumrates.mean()}')
                    final = {'avg_sumrate': avg_sumrates.mean(), 'std_sumrate': np.nanstd(avg_sumrates), 'tot_rates': tot_rates}
                    if args.rotate_codebook:
                        dirext = '_rotated'
                    else:
                        dirext = '3'
                    savedir = f'wmmse_results/{args.scenario}/dft{dirext}/N{args.num_bs_ant}/J{j}'
                    os.makedirs(savedir, exist_ok=True)

                    with open(f'{savedir}/N{args.num_bs_ant}_J{j}_Td{td}_O{tu}_TuM{args.num_magnitude_bits}_snr{snr}.txt', 'w') as f:
                        f.write(f'Average sum rate: {final["avg_sumrate"]:.2f} ± {final["std_sumrate"]:.2f}\n')
                        f.write(f'Average sumrates: {avg_sumrates}\n')
                        f.write(f'NaNs: {nans}\n')
                        f.write(f'Infs: {infs}\n')
                        f.write(f'Large vals: {large_vals}\n')
                        f.write(f'Total rates: {final["tot_rates"]}\n')

                    print("\n" + "="*50 + "\n")