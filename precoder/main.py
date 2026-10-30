import os
import argparse
import json
from tqdm import tqdm

import numpy as np
import torch

from fsqvae import FSQVAE
from common_setup import (add_shared_args, finalize_args, build_network_conf,
                          load_train_val_test_channels, build_optimizer_and_scheduler)

parser = argparse.ArgumentParser(description='PyTorch code for ...')
add_shared_args(parser)
args = parser.parse_args()
finalize_args(args)

fsqvae_config, network_conf, N, J = build_network_conf(args)

# set cache folder
if args.use_dft_matrix:
    args.cache = f'results_fsq{args.dirnum}/{args.scenario}/N{args.num_bs_ant}'
else:
    args.cache = f'results_fsq{args.dirnum}/learn_M/{args.scenario}/N{args.num_bs_ant}'
if args.experiment is None:
    args.experiment = f'N{args.num_bs_ant}_J{args.num_uts}_Td{args.Td}_Tu{args.Tu}'

args.path = os.path.join(args.cache, args.experiment)
os.makedirs(args.path, exist_ok=True)

# init model
model = FSQVAE(fsqvae_config, network_conf, do_simple_bound=args.do_simple_bound).to(args.device)

optim, scheduler, num_epochs = build_optimizer_and_scheduler(model, args)

EVAL_SNRS_DB = [-10, -5, -2.5, 0, 2.5, 5, 10, 15]


def evaluate(chans):
    """Sum rate at each of EVAL_SNRS_DB on a fixed set of J-user groups, plus the per-user
    rate and power distributions (sorted, averaged over groups) at each SNR."""
    out_sumrate = []
    out_rate_distro = {'raw': dict(), 'norm': dict()}
    out_powers = dict()
    for npdb in EVAL_SNRS_DB:
        noise_pwr_linear = 10**(-npdb/10)
        sqrt_noise_pwr_linear = torch.tensor(np.sqrt(noise_pwr_linear), dtype=torch.float, device=args.device)
        noise = sqrt_noise_pwr_linear * torch.randn(size=(chans.shape[0], args.Td, J), dtype=torch.cfloat, device=args.device) # CN(0,sigma^2)

        beam_vecs, codebook_idxs = model((chans, noise))

        # check the power constraint
        v_norms_squared = torch.sum(beam_vecs**2, dim=(-2, -1))
        is_normalized = torch.allclose(v_norms_squared, torch.ones_like(v_norms_squared), atol=1e-5)
        assert is_normalized, f"Normalization check failed at SNR={npdb} dB! Max norm squared is {v_norms_squared.max().item():.4f}, Min is {v_norms_squared.min().item():.4f}."

        # calculate the sumrate
        HV_mag_2 = torch.abs(torch.bmm(beam_vecs[:,:,:N] + 1j*beam_vecs[:,:,N:], chans.conj()))**2
        diags = torch.diagonal(HV_mag_2, dim1=1, dim2=2)
        interference = noise_pwr_linear + (HV_mag_2 - torch.diag_embed(diags)).sum(1)
        sumrate = torch.log2(1 + diags / interference)
        out_sumrate.append(sumrate.sum(1).mean().item())

        # check what proportion of the sumrate is achieved by each user
        sorted_rates, _ = sumrate.sort(dim=-1, descending=True)
        rate_fractions = sorted_rates / (sumrate.sum(dim=-1, keepdim=True) + 1e-8) # just in case something stupid happened and sumrates are 0
        out_rate_distro['raw'][npdb] = sorted_rates.mean(dim=0).tolist()
        out_rate_distro['norm'][npdb] = rate_fractions.mean(dim=0).tolist()

        # check what proportion of the power is allocated to each user
        bv_norms = (beam_vecs.norm(p=2, dim=-1))**2
        bv_norms, _ = bv_norms.sort(dim=-1, descending=True)
        out_powers[npdb] = bv_norms.mean(dim=0).tolist()

    return out_sumrate, out_rate_distro, out_powers


# system parameters
# load in the channels, split 70/15/15 into train/val/test
chans_train, chans_val, chans_test = load_train_val_test_channels(args, N, J)

best_val_sumrate = 0.0
dist = torch.distributions.Uniform(low=-10, high=20)
for epoch in tqdm(range(1, num_epochs + 1), desc='Training epoch', miniters=100, maxinterval=float('inf')):
    optim.zero_grad()
    noise_pwr_linear = 10**(-dist.sample() / 10)
    sqrt_noise_pwr_linear = torch.sqrt(noise_pwr_linear).to(args.device)

    idxs = torch.randperm(chans_train.size(0))
    H = chans_train[idxs]
    H = H.reshape(-1, J, N).permute(0, 2, 1)[:750] # B x N x J
    noise = sqrt_noise_pwr_linear * torch.randn(size=(H.shape[0], args.Td, J), dtype=torch.cfloat, device=args.device) # CN(0,sigma^2)

    # make the beamforming vectors
    beam_vecs, codebook_idxs = model((H, noise))
    HV_mag_2 = torch.abs(torch.bmm(beam_vecs[:,:,:N] + 1j*beam_vecs[:,:,N:], H.conj()))**2
    diags = torch.diagonal(HV_mag_2, dim1=1, dim2=2)

    # calculate sumrate
    interference = noise_pwr_linear + (HV_mag_2 - torch.diag_embed(diags)).sum(1)
    sumrate = torch.log2(1 + diags / interference)
    loss = -sumrate.sum(1).mean()

    # calculate loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
    optim.step()
    if scheduler is not None:
        scheduler.step()

    # model selection on the validation set (sum rate at the highest SNR); the checkpoint that
    # wins is then scored once on the test set, and those are the numbers written out
    if epoch % args.save_interval == 0:
        with torch.no_grad():
            model.eval()
            val_sumrate, _, _ = evaluate(chans_val)
            if val_sumrate[-1] > best_val_sumrate:
                best_val_sumrate = val_sumrate[-1]
                test_sumrate, test_rate_distro, test_powers = evaluate(chans_test)
                if args.save_model:
                    torch.save(model.state_dict(), os.path.join(args.path, 'best_model.pt'))
                with open(os.path.join(args.path, 'best_sumrate.txt'), 'w') as f:
                    f.write(str(test_sumrate))
                with open(os.path.join(args.path, 'best_val_sumrate.txt'), 'w') as f:
                    f.write(str(val_sumrate))
                with open(os.path.join(args.path, 'rate_distro.txt'), 'w', encoding='utf-8') as f:
                    json.dump(test_rate_distro, f, indent=4)
                with open(os.path.join(args.path, 'pwr_distro.txt'), 'w', encoding='utf-8') as f:
                    json.dump(test_powers, f, indent=4)

    model.train()
