"""
Transform-coding (KLT + reverse water-filling + uniform scalar quantization)
baseline for the GMM channel model.

Adapted to the same estimate-then-quantize-then-WMMSE pipeline as dft.py and
cvq_proj.py, so it produces directly comparable results (same output file
format, same directory-naming convention, picked up by
../analysis/aggregate_results.py automatically).

Unlike cvq_proj.py, this needs no training data or codebook fitting: the
"codebook" is just the eigendecomposition of the receiver's correlation-matrix
estimate (avg_corr_mtx, already needed for the LMMSE gain), computed fresh
every run. There's nothing to cache.

IMPORTANT semantic note on --num-feedback-bits, and how to compare it against
the other baselines' O:

- vs. cvq_proj.py: directly comparable, set them equal. cvq_proj.py's
  --num-feedback-bits b builds a 2**b-codeword codebook, and indexing one of
  2**b codewords costs exactly b bits of feedback -- so cvq_proj.py's O
  already *is* a literal bit count, on the same scale as this file's
  --num-feedback-bits (the total bits spent per user, split across the
  channel's 2N real/imaginary KLT coordinates by reverse water-filling).
  Running both scripts with the same --num-feedback-bits value gives both
  methods the same real feedback-overhead budget per user.
- vs. dft.py: NOT directly comparable without converting first. dft.py's O
  is an oversampling factor, not a bit count -- its codebook has N*O
  codewords, so its actual feedback cost is log2(N*O) + TuM bits (TuM =
  --num-magnitude-bits, quantized separately there). Convert dft.py's O to
  bits via that formula before comparing it to this file's
  --num-feedback-bits.

The directory-naming convention keeps calling this "O" for consistency with
the other two baselines so the shared aggregator/plotting pipeline picks it
up the same way.
"""

import argparse
import os
from distutils.util import strtobool

import numpy as np
import torch
from tqdm import tqdm
from scipy.linalg import pinv

from precoding.wmmse import wmmse, sum_rate
from precoding.utils import estimate_channel_corr_mtx, estimate_channel_mp_pinv
from precoding.utils_transform import transform_code_channels


def sample_gmm_channels(decomps, K, num_groups, num_uts, N, rng):
    comp_idxs = rng.integers(0, K, size=(num_groups, num_uts))
    z = np.sqrt(0.5) * (rng.standard_normal((num_groups, num_uts, N))
                         + 1j * rng.standard_normal((num_groups, num_uts, N)))
    L = decomps[comp_idxs]
    return np.einsum('gjnm,gjm->gjn', L, z)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform-coding (KLT + water-filling) baseline for the GMM channel model')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 10000))
    parser.add_argument('--scenario', type=str, default='gmm', help='channel model scenario name (used for output paths)')
    parser.add_argument('--num-bs-ant', type=int, default=8, help='number of antennae at the BS')
    parser.add_argument('--num-uts', type=int, nargs='+', default=[8], help='list of user-terminal counts to sweep')
    parser.add_argument('--num-gmm-components', type=int, default=1, help='Number of components, K, in the GMM channel model')
    parser.add_argument('--snr-db', type=float, nargs='+', default=[15.], help='list of SNRs in dB to sweep')
    parser.add_argument('--Td', type=int, default=8, help='number of pilot symbols')
    parser.add_argument('--num-feedback-bits', type=int, default=32,
                        help='total bits spent per user, split across the 2N real KLT coordinates by reverse water-filling -- directly comparable to cvq_proj.py\'s --num-feedback-bits (set equal for a fair comparison); see module docstring for the dft.py conversion')
    parser.add_argument('--clip-sigma', type=float, default=3.5,
                        help='uniform-quantizer clip range per coordinate, in multiples of that coordinate\'s own standard deviation')
    parser.add_argument('--max-wmmse-iter', type=int, default=25, help='max number of iterations of WMMSE algorithm')
    parser.add_argument('--num-codebooks-to-test', type=int, default=1, help='number of independent test runs')
    parser.add_argument('--use-true-channel', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--channel-inversion-type', type=str, default='lmmse', choices=['lmmse', 'mppinv'])
    args = parser.parse_args()

    assert args.Td <= args.num_bs_ant, "Number of pilot symbols cannot exceed the number of BS antennas."

    N = args.num_bs_ant
    K = args.num_gmm_components

    STATS_DIR = os.path.join('channel_stats', f'N{N}')
    channel_covs = torch.load(os.path.join(STATS_DIR, 'cov.pt'), map_location='cpu')[:K].numpy().astype(np.complex128)
    channel_decomps = torch.load(os.path.join(STATS_DIR, 'decomp.pt'), map_location='cpu')[:K].numpy().astype(np.complex128)
    avg_corr_mtx = channel_covs.mean(axis=0)

    # eigenbasis for the KLT -- unconjugated avg_corr_mtx, matching dft.py's
    # avg_corr_mtx_sqrt convention: this operates on the channel estimate
    # *after* estimate_channel_corr_mtx's LMMSE gain has already applied the
    # y = pilots @ channels.conj() + noise pilot-model conjugate internally
    # and conjugated its own output back (see dft.py's derivation comment),
    # so by this point we're back in the same "channels" domain avg_corr_mtx
    # (unconjugated) already correctly describes.
    eigvals, eigvecs = np.linalg.eigh(avg_corr_mtx)

    rng = np.random.default_rng(seed=args.seed)

    noise_pwr = 1
    num_tst_groups = 500
    num_runs = args.num_codebooks_to_test

    for J in args.num_uts:
        for snr in args.snr_db:
            transmit_pwr = 10**(snr/10) * noise_pwr
            pilot_matrix = np.fft.fft(np.eye(N))[:args.Td]
            pilot_matrix = np.sqrt(transmit_pwr) * (pilot_matrix / np.linalg.norm(pilot_matrix, axis=1, keepdims=True))
            moore_penrose_pinv = pinv(pilot_matrix)
            sigma_2 = noise_pwr * np.ones((J,), dtype=np.complex128)

            avg_sumrates = np.empty(num_runs)
            nans, infs, large_vals = [], [], []
            print(f'\nRunning transform-VQ+WMMSE with N={N}, K={K}, SNR={snr} dB, total_bits={args.num_feedback_bits}, '
                  f'Td={args.Td}, J={J}, max WMMSE iterations={args.max_wmmse_iter}, use true channel={args.use_true_channel}')

            for i in range(num_runs):
                H = sample_gmm_channels(channel_decomps, K, num_tst_groups, J, N, rng)
                # LMMSE gain needs Cov(channels.conj()) = conj(avg_corr_mtx); see gmm/dft.py for the derivation
                corr_mtxs = np.tile(avg_corr_mtx.conj(), (num_tst_groups, J, 1, 1))

                tot_rates = []
                if not args.use_true_channel:
                    if args.channel_inversion_type == 'lmmse':
                        H_quant = estimate_channel_corr_mtx(H, corr_mtxs, pilot_matrix, noise_pwr).transpose(0, 2, 1)
                    else:
                        H_quant = estimate_channel_mp_pinv(H, pilot_matrix, moore_penrose_pinv, noise_pwr).transpose(0, 2, 1)
                    H_quant = transform_code_channels(H_quant, eigvecs, eigvals, args.num_feedback_bits, clip_sigma=args.clip_sigma)
                else:
                    H_quant = H

                for hdx, h in enumerate(tqdm(H, desc="Processing channels")):
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

            print(f'Overall average rate: {avg_sumrates.mean()}')
            final = {'avg_sumrate': avg_sumrates.mean(), 'std_sumrate': np.nanstd(avg_sumrates), 'tot_rates': tot_rates}
            # TuM is unused by this method (kept as 0 so the directory name still matches the
            # shared N*_K*_J*_Td*_O*_TuM* pattern aggregate_results.py's regex expects)
            config_tag = f'N{N}_K{K}_J{J}_Td{args.Td}_O{args.num_feedback_bits}_TuM0'
            savedir = f'wmmse_results/{args.scenario}/transform_vq/N{N}/K{K}/{config_tag}'
            os.makedirs(savedir, exist_ok=True)

            with open(f'{savedir}/snr{snr}.txt', 'w') as f:
                f.write(f'Average sum rate: {final["avg_sumrate"]:.2f} ± {final["std_sumrate"]:.2f}\n')
                f.write(f'Average sumrates: {avg_sumrates}\n')
                f.write(f'NaNs: {nans}\n')
                f.write(f'Infs: {infs}\n')
                f.write(f'Large vals: {large_vals}\n')
                f.write(f'Total rates: {final["tot_rates"]}\n')
