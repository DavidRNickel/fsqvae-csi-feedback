import os
import sys
import argparse
import random
from distutils.util import strtobool
from tqdm import tqdm

import numpy as np

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common_setup import get_fsq_levels

from models.fsqvae import FSQVAE
from channels.gmm_simple import sample_gmm

parser = argparse.ArgumentParser(description='PyTorch code for ...')
parser.add_argument('--lr', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--experiment', default=None, help='name of experiment')

# seed
parser.add_argument('--seed', type=int, default=np.random.randint(0, 10000), metavar='N',
                    help='random seed (default: 456)')

# log
parser.add_argument('--eval-interval', type=int, default=100,
                    help='eval interval (step)')
parser.add_argument('--save-interval', type=int, default=1000,
                    help='eval interval (step)')
parser.add_argument('--save-model', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)

parser.add_argument('--scenario', type=str, default='gmm', help='deepmimo scenario to use for channel generation')

# encoder/decoder parameters; turned into dictionaries below
parser.add_argument('--ut-d-model', type=int, default=96)
parser.add_argument('--ut-nhead', type=int, default=4)
parser.add_argument('--ut-dim-feedforward', type=int, default=192)
parser.add_argument('--ut-activation', type=str, default='relu')
parser.add_argument('--ut-num-layers', type=int, default=2)

parser.add_argument('--bs-d-model', type=int, default=128)
parser.add_argument('--bs-nhead', type=int, default=4)
parser.add_argument('--bs-dim-feedforward', type=int, default=256)
parser.add_argument('--bs-activation', type=str, default='relu')
parser.add_argument('--bs-num-layers', type=int, default=3)

# network setup
parser.add_argument('--num-bs-ant', type=int, default=3, help='number of antennae at the BS')
parser.add_argument('--num-uts', type=int, default=2, help='number of user terminals')
parser.add_argument('--snr-db', type=float, default=15, help='SNR in dB')
parser.add_argument('--evolution-term-alpha', type=float, default=0.90, help='H(t+1)=a*H(t)+sqrt(1-a^2)e(t),e~CN(0,I)')
parser.add_argument('--Td', type=int, default=3, help='number of pilot symbols')

# FSQ-VAE setup
parser.add_argument('--Tu', type=int, default=8, help='number of uplink symbols')
parser.add_argument('--num-gmm-components', type=int, default=3, help='Number of components, K, in the GMM.')
parser.add_argument('--do-simple-bound', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=False)
parser.add_argument('--use-dft-matrix', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
parser.add_argument('--dirnum', type=str, default='', help='suffix appended to the results cache directory name')

args = parser.parse_args()
# make sure everything is valid
assert args.batch_size >= 2 # the slicing/masking mechanisms in fsq-vae won't work if this is one element
assert args.num_bs_ant>=1 and args.num_uts>=1
assert args.Td>=1
assert args.Tu>=4

if args.seed >= 0:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

if torch.cuda.is_available():
    args.cuda = True
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# make the args into dictionaries needed by VQ-VAE
ut_encoder_layer_config = {'d_model': args.ut_d_model,
                           'nhead': args.ut_nhead,
                           'dim_feedforward': args.ut_dim_feedforward,
                           'dropout': 0.0,
                           'activation': args.ut_activation,
                           'layer_norm_eps': 1e-5,
                           'batch_first': True,
                           'norm_first': True}

ut_encoder_config = {'num_layers': args.ut_num_layers,
                     'norm': None,
                     'enable_nested_tensor': True,
                     'mask_check': True}

bs_encoder_layer_config = {'d_model': args.bs_d_model,
                           'nhead': args.bs_nhead,
                           'dim_feedforward': args.bs_dim_feedforward,
                           'dropout': 0.0,
                           'activation': args.bs_activation,
                           'layer_norm_eps': 1e-5,
                           'batch_first': True,
                           'norm_first': True}

bs_encoder_config = {'num_layers': args.bs_num_layers,
                     'norm': None,
                     'enable_nested_tensor': True,
                     'mask_check': True}

levels = get_fsq_levels(args.Tu)

fsqvae_config= {'ut_encoder_layer_config': ut_encoder_layer_config, 
                'ut_encoder_config': ut_encoder_config, 
                'bs_encoder_layer_config': bs_encoder_layer_config, 
                'bs_encoder_config': bs_encoder_config, 
                'levels': levels} 

assert args.Td <= args.num_bs_ant, 'number of pilot symbols must be less than or equal to the number of BS antennae for the current pilot design'

if args.use_dft_matrix:
    M = np.fft.fft(np.eye(args.num_bs_ant)) # DFT matrix for pilot symbols
    M = torch.from_numpy(M)[:args.Td].to(args.device).to(torch.cfloat) # DFT matrix for pilot symbols
    M = F.normalize(M, p=2, dim=1) # normalize the rows of M to have unit norm (0 normalizes columns)
else:
    M = None
N = args.num_bs_ant
J = args.num_uts

network_conf = {'N': args.num_bs_ant,
                'J': args.num_uts,
                'Td': args.Td,
                'snr_db': args.snr_db,
                'alpha': args.evolution_term_alpha,
                'use_dft_matrix': args.use_dft_matrix,
                'pilot_matrix': M}

# set cache folder
if args.use_dft_matrix:
    args.cache = f'results_fsq{args.dirnum}/{args.scenario}/N{args.num_bs_ant}/K{args.num_gmm_components}'
else:
    args.cache = f'results_fsq{args.dirnum}/learn_M/{args.scenario}/N{args.num_bs_ant}/K{args.num_gmm_components}'
if args.experiment is None:
    args.experiment = f'N{args.num_bs_ant}_J{args.num_uts}_Td{args.Td}_Tu{args.Tu}'

args.path = os.path.join(args.cache, args.experiment)
os.makedirs(args.path, exist_ok=True)

# init model
model = FSQVAE(fsqvae_config, network_conf, do_simple_bound=args.do_simple_bound).to(args.device)

# init optimizer
optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

# init scheduler
num_epochs = 15000
warmup_steps = 500
warmup = LambdaLR(optim, lr_lambda=lambda step: min((step+1) / warmup_steps, 1))
cosine = CosineAnnealingLR(optim, T_max=num_epochs-warmup_steps)
scheduler = SequentialLR(optim, schedulers=[warmup, cosine], milestones=[warmup_steps])

EVAL_SNRS_DB = [-10, -5, -2.5, 0, 2.5, 5, 10, 15]


def evaluate(chans):
    """Sum rate at each of EVAL_SNRS_DB on a fixed set of J-user groups."""
    out_sumrate = []
    for npdb in EVAL_SNRS_DB:
        noise_pwr_linear = 10**(-npdb/10)
        sqrt_noise_pwr_linear = torch.tensor(np.sqrt(noise_pwr_linear), dtype=torch.float, device=args.device)
        noise = sqrt_noise_pwr_linear * torch.randn(size=(chans.shape[0], args.Td, J), dtype=torch.cfloat, device=args.device) # CN(0,sigma^2)

        beam_vecs, codebook_idxs = model((chans, noise))
        HV_mag_2 = torch.abs(torch.bmm(beam_vecs[:,:,:N] + 1j*beam_vecs[:,:,N:], chans.conj()))**2
        diags = torch.diagonal(HV_mag_2, dim1=1, dim2=2)
        interference = noise_pwr_linear + (HV_mag_2 - torch.diag_embed(diags)).sum(1)
        sumrate = torch.log2(1 + diags / interference)
        out_sumrate.append(sumrate.sum(1).mean().item())

    return out_sumrate


# system parameters
K, N, J = args.num_gmm_components, args.num_bs_ant, args.num_uts
ROOT_DIR = os.path.join('channel_stats', f'N{N}')
channel_covs = torch.load(os.path.join(ROOT_DIR, 'cov.pt'), map_location=args.device)[:K]
channel_decomps = torch.load(os.path.join(ROOT_DIR, 'decomp.pt'), map_location=args.device)[:K]
trn_idxs = torch.randint(high=K, size=(10*args.batch_size, J))
H = sample_gmm(L=channel_decomps, idxs=trn_idxs, device=args.device)
H = H.to(torch.cfloat).to(args.device) # B x N x J
test_idxs = torch.randint(high=K, size=(500, J))
chans_test = sample_gmm(L=channel_decomps, idxs=test_idxs, device=args.device)
# 4x the test set's size to cut the noise in checkpoint selection
val_idxs = torch.randint(high=K, size=(2000, J))
chans_val = sample_gmm(L=channel_decomps, idxs=val_idxs, device=args.device)

best_val_sumrate = 0.0
dist = torch.distributions.Uniform(low=-10, high=20)
for epoch in tqdm(range(1, num_epochs + 1), desc='Training epoch', miniters=100, maxinterval=float('inf')):
    optim.zero_grad()
    noise_pwr_linear = 10**(-dist.sample() / 10)
    sqrt_noise_pwr_linear = torch.sqrt(noise_pwr_linear).to(args.device)

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
            val_sumrate = evaluate(chans_val)
            if val_sumrate[-1] > best_val_sumrate:
                best_val_sumrate = val_sumrate[-1]
                test_sumrate = evaluate(chans_test)
                if args.save_model:
                    torch.save(model.state_dict(), os.path.join(args.path, 'best_model.pt'))
                with open(os.path.join(args.path, 'best_sumrate.txt'), 'w') as f:
                    f.write(str(test_sumrate))
                with open(os.path.join(args.path, 'best_val_sumrate.txt'), 'w') as f:
                    f.write(str(val_sumrate))
    model.train()
