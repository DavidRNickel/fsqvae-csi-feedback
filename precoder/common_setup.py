"""
Shared CLI-argument, config-dict, channel-loading, and optimizer/scheduler
setup used identically across main.py and main_vq.py (the deepmimo FSQ-VAE and
VQ-VAE trainers; gmm/main.py and gmm/main_vq.py borrow get_fsq_levels/get_vq_config).

Only this boilerplate is factored out here -- each file's actual identity
(its docstring, its own extra CLI flags, the model class it instantiates,
its loss, and its training/eval loop shape) stays in its own file, so each
one is still readable standalone without having to open this module too.
"""

import random
from distutils.util import strtobool

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
import torch.nn.functional as F


def add_shared_args(parser):
    """
    Add every CLI flag common to all main*.py training scripts. Call this
    before adding any variant-specific flags, then parser.parse_args().
    """
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default:True)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=5000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--warmup-steps', type=int, default=1000, metavar='N',
                        help='Warmup before cosine annealing (default: 1000)')
    parser.add_argument('--experiment', default=None, help='name of experiment')
    parser.add_argument('--resume', dest='resume', action='store_true', default=True,
                        help='flag to resume the experiments')
    parser.add_argument('--no-resume', dest='resume', action='store_false', default=True,
                        help='flag to resume the experiments')
    parser.add_argument('--exp-num', type=int, default=0,
                        help='experiment number')

    parser.add_argument('--seed', type=int, default=np.random.randint(0, 10000), metavar='N',
                        help='random seed (default: 456)')

    parser.add_argument('--log-interval', type=int, default=100,
                        help='log print-out interval (step)')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='eval interval (step)')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='eval interval (step)')
    parser.add_argument('--save-model', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)

    parser.add_argument('--scenario', type=str, default='asu_campus_3p5', help='deepmimo scenario to use for channel generation')
    parser.add_argument('--dirnum', type=str, default='_new', help='suffix appended to the results cache directory name')

    # encoder/decoder parameters; turned into dictionaries by build_network_conf()
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
    parser.add_argument('--do-simple-bound', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=False)
    parser.add_argument('--use-dft-matrix', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)


def finalize_args(args):
    """Validate args, seed every RNG, and pick a device. Call right after parser.parse_args()."""
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
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_fsq_levels(Tu):
    """The single source of truth for the FSQ `levels` schedule as a function
    of --Tu, shared by every main*.py (via build_network_conf) and by
    gmm/main.py, so the two stay consistent."""
    if Tu == 8:
        levels = [4, 4, 4, 4]
    elif Tu == 12:
        levels = 3*[16]
    elif Tu == 16:
        levels = [16, 16, 8, 8, 4]
    elif Tu == 20:
        levels = 6*[8] + [4]
    elif Tu == 40:
        levels = 4*[16] + 8*[8]
    else:
        raise ValueError(f'Unsupported number of uplink symbols: {Tu}')
    return levels


def get_vq_config(Tu, num_subvectors=4, code_dim=2):
    """The single source of truth for the VQ-VAE codebook schedule as a function
    of --Tu, shared by main_vq.py and gmm/main_vq.py -- the VQ counterpart of
    get_fsq_levels. Follows the sub-vector scheme of Allaparapu et al. (ICASSP
    2026), Sec. 3.1: each user's latent of num_subvectors*code_dim entries is
    split into num_subvectors sub-vectors of length code_dim, all quantized
    against one shared codebook, so Tu = num_subvectors * log2(codebook_size)
    bits. With the defaults (the paper's N_E=2, N_L=8), Tu=8/12/16/20/40 gives
    codebook_size=4/8/16/32/1024."""
    if Tu % num_subvectors != 0:
        raise ValueError(f'Tu={Tu} bits cannot be split evenly across {num_subvectors} sub-vectors')
    return {'codebook_size': 2**(Tu // num_subvectors),
            'num_subvectors': num_subvectors,
            'code_dim': code_dim}


def build_network_conf(args):
    """
    Build the FSQVAE encoder/decoder config dicts, the FSQ `levels` schedule
    for --Tu, the DFT (or learned) pilot matrix, and the per-user
    network_conf dict every model class in this repo is constructed from.

    Returns (fsqvae_config, network_conf, N, J).
    """
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
                         'mask_check': False}

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
                         'mask_check': False}

    levels = get_fsq_levels(args.Tu)

    fsqvae_config = {'ut_encoder_layer_config': ut_encoder_layer_config,
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

    return fsqvae_config, network_conf, N, J


# Fixed seed for the train/val/test split, deliberately independent of --seed: every training
# run (and every seed averaged together in analysis/aggregate_results.py) is then selected on
# the same validation set and scored on the same test set.
SPLIT_SEED = 0


def load_train_val_test_channels(args, N, J):
    """Load the channel dataset, split it 70/15/15 into train/val/test, and reshape.

    chans_train stays flat (num_users, N) -- the training loops regroup it into random J-user
    sets every step. chans_val and chans_test are fixed J-user groups, (num_groups, N, J).
    """
    loadfile = f'channel_data/{args.scenario}/{N}x1/channels.npy'
    channels = np.load(loadfile).squeeze() # (tot_num_uts, num_bs_ant)
    chans_train, chans_rest = train_test_split(channels, test_size=0.3, random_state=SPLIT_SEED)
    chans_val, chans_test = train_test_split(chans_rest, test_size=0.5, random_state=SPLIT_SEED)

    def _to_tensor(chans):
        # drop the remainder so the number of channels is divisible by the number of user terminals
        return torch.from_numpy(chans[:len(chans) - len(chans) % J]).requires_grad_(False).to(args.device).to(torch.cfloat)

    def _to_groups(chans):
        return _to_tensor(chans).reshape(-1, J, N).permute(0, 2, 1)

    return _to_tensor(chans_train), _to_groups(chans_val), _to_groups(chans_test)


def build_optimizer_and_scheduler(model, args):
    """AdamW + linear-warmup/cosine-annealing schedule shared by every main*.py variant."""
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    num_epochs = 15000
    warmup_steps = 500
    warmup = LambdaLR(optim, lr_lambda=lambda step: min((step+1) / warmup_steps, 1))
    cosine = CosineAnnealingLR(optim, T_max=num_epochs-warmup_steps)
    scheduler = SequentialLR(optim, schedulers=[warmup, cosine], milestones=[warmup_steps])

    return optim, scheduler, num_epochs
