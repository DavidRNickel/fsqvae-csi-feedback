"""
Transform-coding (KLT + reverse water-filling + uniform scalar quantization)
baseline, in the same estimate-then-quantize-then-WMMSE pipeline as
cvq_proj.py/dft.py -- same output file format and directory-naming
convention, so results are directly comparable.

Needs no training data or codebook fitting (unlike cvq_proj.py's K-means
codebook): unlike gmm/transform_vq.py (whose scenario has no per-user
correlation, only a single population-average one, so it must share one KLT
eigenbasis across every user), this scenario's channel_data ships a real
per-user spatial correlation matrix (corr_matrices), so each user gets its
own KLT eigenbasis and its own reverse-water-filling bit allocation over
--num-feedback-bits total bits. There's nothing to cache: the "codebook" is just
each user's own correlation-matrix eigendecomposition, computed fresh every
run.

Semantic note on --num-feedback-bits: this is a total per-user bit count, not
an oversampling factor or codebook-size exponent -- directly comparable to
cvq_proj.py's --num-feedback-bits (set equal for a fair comparison); see
gmm/transform_vq.py's module docstring for how to convert dft.py's
--num-feedback-bits (an oversampling factor) to the same scale. There is no
separate --num-magnitude-bits here: the whole bit budget is already split
across the 2N real KLT coordinates by reverse water-filling.

--num-uts/--Td/--num-feedback-bits/--snr-db all take one or more values and
are swept internally (nested loops, outermost to innermost in that order) --
instead of relying on a
shell script to invoke this once per combination. That keeps numba's JIT warm
across an entire sweep (only paid once per process) instead of once per
combination.
"""

import argparse
import os
from distutils.util import strtobool

import numpy as np
from tqdm import tqdm
from scipy.linalg import pinv

from wmmse import wmmse, sum_rate
from utils import estimate_channel_mp_pinv, estimate_channel_corr_mtx
from utils_transform import transform_code_channels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform-coding (KLT + water-filling) baseline')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 10000), metavar='N', help='random seed (default: 456)')
    parser.add_argument('--scenario', type=str, default='asu_campus_3p5', help='deepmimo scenario to use for channel generation')
    parser.add_argument('--num-bs-ant', type=int, default=64, help='number of antennae at the BS')
    parser.add_argument('--num-uts', type=int, nargs='+', default=[8], help='list of J: number of user terminals to sweep')
    parser.add_argument('--snr-db', type=float, nargs='+', default=[0.], help='list of SNRs in dB to sweep')
    parser.add_argument('--Td', type=int, nargs='+', default=[14], help='list of numbers of pilot symbols to sweep')
    parser.add_argument('--num-feedback-bits', type=int, nargs='+', default=[12],
                        help='list of total per-user bit budgets to sweep, each split across the 2N real KLT coordinates by '
                             'reverse water-filling -- directly comparable to cvq_proj.py\'s --num-feedback-bits (set equal for a fair comparison)')
    parser.add_argument('--clip-sigma', type=float, default=3.5,
                        help='uniform-quantizer clip range per coordinate, in multiples of that coordinate\'s own standard deviation')
    parser.add_argument('--max-wmmse-iter', type=int, default=25, help='max number of iterations of WMMSE algorithm')
    parser.add_argument('--num-codebooks-to-test', type=int, default=1, help='number of independent test runs')
    parser.add_argument('--use-true-channel', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--channel-inversion-type', type=str, default='lmmse', choices=['lmmse', 'mppinv'], help='type of channel inversion to use')
    args = parser.parse_args()

    assert all(td <= args.num_bs_ant for td in args.Td), "Number of pilot symbols cannot exceed the number of BS antennas."

    loadfile = f'../channel_data/{args.scenario}/{args.num_bs_ant}x1/channels.npy'
    loadfile_corr = f'../channel_data/{args.scenario}/{args.num_bs_ant}x1/corr_matrices.npy'
    channels = np.load(loadfile).squeeze()
    corr_matrices = np.load(loadfile_corr).astype(np.complex128)

    noise_pwr = 1
    num_tst_groups = 500
    rng = np.random.default_rng(seed=args.seed)
    full_idxs = np.arange(len(channels))

    for ut in args.num_uts:
        savedir = f'wmmse_results/{args.scenario}/transform_vq/N{args.num_bs_ant}/J{ut}'
        os.makedirs(savedir, exist_ok=True)

        num_tst_idxs = num_tst_groups * ut
        sigma_2 = noise_pwr * np.ones((ut,), dtype=np.complex128)

        for tu in args.num_feedback_bits:
            for td in args.Td:
                for snr in args.snr_db:
                    transmit_pwr = 10**(snr/10) * noise_pwr
                    pilot_matrix = np.fft.fft(np.eye(args.num_bs_ant))[:td]
                    pilot_matrix = np.sqrt(transmit_pwr) * (pilot_matrix / np.linalg.norm(pilot_matrix, axis=1, keepdims=True))
                    moore_penrose_pinv = pinv(pilot_matrix)

                    num_runs = args.num_codebooks_to_test
                    avg_sumrates = np.empty(num_runs)
                    nans, infs, large_vals = [], [], []
                    print(f'\nRunning transform-VQ+WMMSE with N={args.num_bs_ant}, SNR={snr} dB, total_bits={tu}, '
                          f'Td={td}, J={ut}, max WMMSE iterations={args.max_wmmse_iter}, use true channel={args.use_true_channel}')

                    for i in range(num_runs):
                        idxs = rng.choice(full_idxs, size=num_tst_idxs, replace=False)
                        chans = channels[idxs]
                        corr_mtxs = corr_matrices[idxs].reshape(-1, ut, args.num_bs_ant, args.num_bs_ant)

                        H = chans.reshape(-1, ut, args.num_bs_ant).astype(np.complex128)
                        tot_rates = []

                        if not args.use_true_channel:
                            if args.channel_inversion_type == 'lmmse':
                                H_quant = estimate_channel_corr_mtx(H, corr_mtxs, pilot_matrix, noise_pwr).transpose(0, 2, 1)
                            else:
                                H_quant = estimate_channel_mp_pinv(H, pilot_matrix, moore_penrose_pinv, noise_pwr).transpose(0, 2, 1)

                            # per-sample KLT eigenbasis -- each (group, user) draw carries its own correlation matrix
                            # (a fresh random position per group, not a persistent per-slot identity across groups), so
                            # transform_code_channels (built for one shared eigenbasis per call) is applied one sample at
                            # a time rather than batched.
                            #
                            # corr_mtxs (as loaded from corr_matrices.npy / make_channels.py's compute_correlation_matrix)
                            # is Cov(channels.conj()) -- correct as-is for the LMMSE call above, but the KLT synthesis step
                            # (h = U c) needs the eigenbasis of the *unconjugated* Cov(channels), i.e. conj(corr_mtxs) --
                            # same distinction gmm/transform_vq.py's module docstring documents for avg_corr_mtx. Using
                            # corr_mtxs directly here (unconjugated) previously gave a KLT basis aligned with conj(h) instead
                            # of h, so reverse water-filling spent its whole bit budget on directions carrying ~0% of the
                            # true channel's energy -- measured: median |<top eigvec, h>|^2/||h||^2 was 0.016 with the bug,
                            # 0.998 after conjugating, on the real asu_campus_3p5 channel_data used by this scenario.
                            num_groups = H_quant.shape[0]
                            eigvals, eigvecs = np.linalg.eigh(corr_mtxs.conj())  # (num_groups, J, N), (num_groups, J, N, N)
                            H_quant_coded = np.empty(H_quant.shape, dtype=np.complex128)
                            for g in range(num_groups):
                                for j in range(ut):
                                    H_quant_coded[g, j] = transform_code_channels(H_quant[g, j], eigvecs[g, j], eigvals[g, j],
                                                                                   tu, clip_sigma=args.clip_sigma)
                            H_quant = H_quant_coded
                        else:
                            H_quant = H

                        for hdx, h in enumerate(tqdm(H, desc=f"Processing channels (J={ut}, Tu={tu}, Td={td}, SNR={snr})")):
                            h_est = H_quant[hdx].copy()
                            try:
                                v = wmmse(H=h_est[:, None, :], P_max=transmit_pwr, sigma_2=sigma_2,
                                         alphas=np.ones((ut,), dtype=np.complex128), max_wmmse_iter=args.max_wmmse_iter)
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

                    print(f'Overall average rate: {avg_sumrates.mean()}')
                    final = {'avg_sumrate': avg_sumrates.mean(), 'std_sumrate': np.nanstd(avg_sumrates), 'tot_rates': tot_rates}
                    with open(f'{savedir}/N{args.num_bs_ant}_J{ut}_Td{td}_Tu{tu}_snr{snr}.txt', 'w') as f:
                        f.write(f'Average sum rate: {final["avg_sumrate"]:.2f} ± {final["std_sumrate"]:.2f}\n')
                        f.write(f'Average sumrates: {avg_sumrates}\n')
                        f.write(f'NaNs: {nans}\n')
                        f.write(f'Infs: {infs}\n')
                        f.write(f'Large vals: {large_vals}\n')
                        f.write(f'Total rates: {final["tot_rates"]}\n')
