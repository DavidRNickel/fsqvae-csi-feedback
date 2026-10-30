import argparse
import sys, os, time
from tqdm import tqdm
from distutils.util import strtobool
import numpy as np
import torch
from scipy.linalg import pinv
from precoding.wmmse import wmmse, sum_rate
from precoding.utils import estimate_channel_mp_pinv, estimate_channel_corr_mtx
from precoding.utils_dft import generate_dft_codebook, quantize_channels_rotated


def hermitian_sqrtm(A):
    w, v = np.linalg.eigh(A)
    w = np.clip(w, a_min=0, a_max=None)
    return (v * np.sqrt(w)) @ v.conj().T


if __name__=='__main__':
    save_results = False
    parser = argparse.ArgumentParser(description='PyTorch code for ...')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 10000), metavar='N', help='random seed (default: 456)')
    parser.add_argument('--scenario', type=str, default='gmm', help='channel model scenario name (used for output paths)')
    parser.add_argument('--num-bs-ant', type=int, default=8, help='number of antennae at the BS')
    parser.add_argument('--num-uts', type=int, nargs='+', default=[4], help='list of user-terminal counts to sweep')
    parser.add_argument('--num-gmm-components', type=int, default=3, help='Number of components, K, in the GMM channel model')
    parser.add_argument('--snr-db', type=float, nargs='+', default=[10.], help='list of SNRs in dB to sweep')
    parser.add_argument('--Td', type=int, default=14, help='number of pilot symbols')
    parser.add_argument('--max-wmmse-iter', type=int, default=25, help='max number of iterations of WMMSE algorithm')
    parser.add_argument('--num-codebooks-to-test', type=int, default=1, help='number of codebooks to test')
    parser.add_argument('--use-true-channel', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--rotate-codebook', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--channel-inversion-type', type=str, default='lmmse', choices=['lmmse', 'mppinv'], help='type of channel inversion to use')
    parser.add_argument('--num-feedback-bits', type=int, default=12, help='number of feedback bits for channel angle quantization')
    parser.add_argument('--num-magnitude-bits', type=int, default=4, help='number of bits for quantizing channel magnitudes')
    args = parser.parse_args()

    # assert args.num_feedback_bits <= 20, "Too many feedback bits may lead to excessive memory usage. Please reduce the number of feedback bits."
    assert args.Td <= args.num_bs_ant, "Number of pilot symbols cannot exceed the number of BS antennas."

    # GMM channel model: K component covariance matrices and their Cholesky factors,
    # same source used by main.py
    N = args.num_bs_ant
    K = args.num_gmm_components
    ROOT_DIR = os.path.join('channel_stats', f'N{N}')
    channel_covs = torch.load(os.path.join(ROOT_DIR, 'cov.pt'), map_location='cpu')[:K].numpy().astype(np.complex128)
    channel_decomps = torch.load(os.path.join(ROOT_DIR, 'decomp.pt'), map_location='cpu')[:K].numpy().astype(np.complex128)

    # the estimator/quantizer doesn't know which of the K components a given user's
    # channel was drawn from, so it uses the average correlation matrix across all
    # components in place of the (unknown) true per-user correlation matrix
    avg_corr_mtx = channel_covs.mean(axis=0)
    avg_corr_mtx_sqrt = hermitian_sqrtm(avg_corr_mtx)

    num_runs = args.num_codebooks_to_test
    avg_sumrates = np.empty(num_runs)

    print("Warming up WMMSE solver...")
    warmup_j = args.num_uts[0]
    mock_H = np.ones((warmup_j, 1, args.num_bs_ant), dtype=np.complex128)
    _ = wmmse(H=mock_H, P_max=1.0, sigma_2=np.ones((warmup_j,), dtype=np.complex128), alphas=np.ones(warmup_j, dtype=np.complex128), max_wmmse_iter=1)
    print("Warm-up complete.")

    MIN_RATE_THRESHOLD = 0. # artifact of earlier version
    for tu in [args.num_feedback_bits]:
        args.num_feedback_bits = tu
        codebook = generate_dft_codebook(num_antennas=args.num_bs_ant, oversampling_factor=args.num_feedback_bits)
        for j in args.num_uts:
            for snr in args.snr_db:
                noise_pwr = 1
                transmit_pwr = 10**(snr/10)*noise_pwr
                pilot_matrix = np.fft.fft(np.eye(args.num_bs_ant))[:args.Td] # DFT matrix for pilot symbols
                pilot_matrix = np.sqrt(transmit_pwr) * (pilot_matrix / np.linalg.norm(pilot_matrix, axis=1, keepdims=True))
                moore_penrose_pinv = pinv(pilot_matrix)

                sigma_2 = noise_pwr * np.ones((j,), dtype=np.complex128)
                num_tst_groups = 350
                num_tst_idxs = num_tst_groups * j

                rng = np.random.default_rng(seed=args.seed)
                nans, infs, large_vals = [], [], []

                print(f'\nRunning WMMSE with N={args.num_bs_ant}, SNR={snr} dB, O={args.num_feedback_bits}, TuM={args.num_magnitude_bits}, Td={args.Td}, J={j}, K={args.num_gmm_components}, max WMMSE iterations={args.max_wmmse_iter}, use true channel={args.use_true_channel}')

                successful_runs = 0
                while successful_runs < num_runs:
                    # sample the GMM channels: each user's channel is drawn from a
                    # randomly chosen one of the K mixture components
                    comp_idxs = rng.integers(0, K, size=(num_tst_groups, j))
                    # CN(0,I): real/imag parts each variance 0.5 so E[|z|^2]=1 per entry
                    z = np.sqrt(0.5) * (rng.standard_normal((num_tst_groups, j, args.num_bs_ant))
                        + 1j * rng.standard_normal((num_tst_groups, j, args.num_bs_ant)))
                    L = channel_decomps[comp_idxs] # (num_tst_groups, num_uts, N, N)
                    H = np.einsum('gjnm,gjm->gjn', L, z)

                    if args.rotate_codebook:
                        corr_mtxs_sqrt = np.tile(avg_corr_mtx_sqrt, (num_tst_idxs, 1, 1))
                    else:
                        corr_mtxs_sqrt = np.tile(np.eye(args.num_bs_ant), (num_tst_idxs, 1, 1))
                    # estimate_channel_corr_mtx's LMMSE gain estimates channels.conj() internally
                    # (matching the y = pilots @ channels.conj() + noise pilot model), so it needs
                    # Cov(channels.conj()) = conj(avg_corr_mtx), not avg_corr_mtx itself. The DFT-codebook
                    # rotation below is applied after that estimate is conjugated back, so it correctly
                    # keeps the unconjugated avg_corr_mtx_sqrt.
                    corr_mtxs = np.tile(avg_corr_mtx.conj(), (num_tst_groups, j, 1, 1))
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
                rot_tag = 'rotated' if args.rotate_codebook else 'unrotated'
                config_tag = f'N{args.num_bs_ant}_K{args.num_gmm_components}_J{j}_Td{args.Td}_O{args.num_feedback_bits}_TuM{args.num_magnitude_bits}'
                savedir = f'wmmse_results/{args.scenario}/dft_{rot_tag}/N{args.num_bs_ant}/K{args.num_gmm_components}/{config_tag}'
                os.makedirs(savedir, exist_ok=True)

                with open(f'{savedir}/snr{snr}.txt', 'w') as f:
                    f.write(f'Average sum rate: {final["avg_sumrate"]:.2f} ± {final["std_sumrate"]:.2f}\n')
                    f.write(f'Average sumrates: {avg_sumrates}\n')
                    f.write(f'NaNs: {nans}\n')
                    f.write(f'Infs: {infs}\n')
                    f.write(f'Large vals: {large_vals}\n')
                    f.write(f'Total rates: {final["tot_rates"]}\n')

                print("\n" + "="*50 + "\n")
