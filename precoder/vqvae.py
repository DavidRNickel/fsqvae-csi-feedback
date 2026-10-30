"""
VQ-VAE counterpart of fsqvae.py's FSQVAE: the same UT encoder, BS decoder and
pilot matrix, with FSQ swapped for a learned vector quantizer.

The quantizer follows the sub-vector scheme of Allaparapu et al., "Precoder
Design in Multi-User FDD Systems with VQ-VAE and GNN" (ICASSP 2026), Sec. 3.1:
each user's latent of num_subvectors*code_dim entries is split into
num_subvectors sub-vectors of length code_dim, and every sub-vector is snapped
to its nearest codeword in ONE codebook shared by all sub-vectors and all users,
so each user feeds back num_subvectors*log2(codebook_size) bits. See
get_vq_config() in common_setup.py for the Tu -> codebook schedule.

Training is plain VQ-VAE (van den Oord et al. 2017: straight-through gradient,
codebook loss, commitment loss) plus exactly two additions. Both act inside a
training step, so neither adds steps on top of main_vq.py's 15k:
  * k-means initialization of the codebook from the first training batch's latents
  * dead-code restart: every training step, each codeword that no sub-vector in
    the batch was assigned to is moved onto a randomly chosen sub-vector of that batch
"""
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from enc_dec_modules import *
from fsqvae import UnitNormRows

#
#
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, code_dim, beta=0.25, kmeans_iters=10):
        """
        codebook_size: number of codewords, shared by every sub-vector
        code_dim: length of each sub-vector and codeword (N_E in the paper)
        beta: commitment loss weight
        kmeans_iters: Lloyd iterations for the one-time codebook initialization
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.beta = beta
        self.kmeans_iters = kmeans_iters

        self.codebook = nn.Embedding(codebook_size, code_dim)
        # persistent, so a reloaded checkpoint isn't re-initialized on its first training step
        self.register_buffer('initialized', torch.tensor(False))

    @torch.no_grad()
    def kmeans_init(self, z):
        """Lloyd's k-means on the sub-vectors z (M x code_dim), seeded with codebook_size distinct rows of z."""
        assert z.shape[0] >= self.codebook_size, f'need at least {self.codebook_size} sub-vectors to initialize the codebook, got {z.shape[0]}'
        centroids = z[torch.randperm(z.shape[0], device=z.device)[:self.codebook_size]].clone()
        for _ in range(self.kmeans_iters):
            assign = torch.cdist(z, centroids).argmin(dim=-1)
            counts = torch.bincount(assign, minlength=self.codebook_size)
            sums = torch.zeros_like(centroids).index_add_(0, assign, z)
            nonempty = counts > 0 # an empty cluster keeps its previous centroid
            centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(-1)
        self.codebook.weight.copy_(centroids)
        self.initialized.fill_(True)

    @torch.no_grad()
    def restart_dead_codes(self, z, indices):
        """Move every codeword that no row of z (M x code_dim) was assigned to onto a distinct, randomly chosen row of z."""
        dead = torch.bincount(indices, minlength=self.codebook_size) == 0
        num_dead = int(dead.sum())
        if num_dead > 0:
            self.codebook.weight[dead] = z[torch.randperm(z.shape[0], device=z.device)[:num_dead]]

    def forward(self, z):
        """
        z: B x ... x (num_subvectors*code_dim), unquantized latents

        Returns
            the quantized latents, same shape as z, with a straight-through gradient to z
            the codeword index of every sub-vector, B x ... x num_subvectors
            the VQ loss, codebook + beta*commitment: each is a squared error summed over all of
                a sample's sub-vectors (all of its users) and averaged over B, the same
                reduction main_vq.py applies to the sum rate
        """
        bs = z.shape[0]
        z_sub = z.reshape(-1, self.code_dim)

        if self.training and not self.initialized:
            self.kmeans_init(z_sub.detach())

        with torch.no_grad():
            indices = torch.cdist(z_sub, self.codebook.weight).argmin(dim=-1)
        z_q = self.codebook(indices)

        codebook_loss = (z_q - z_sub.detach()).pow(2).sum() / bs
        commitment_loss = (z_sub - z_q.detach()).pow(2).sum() / bs
        vq_loss = codebook_loss + self.beta * commitment_loss

        # in-place on the codebook, after the lookup above -- safe for backward, since the
        # embedding's gradient depends only on `indices`, not on the codebook's values
        if self.training:
            self.restart_dead_codes(z_sub.detach(), indices)

        z_q = z_sub + (z_q - z_sub).detach() # straight-through
        return z_q.reshape(z.shape), indices.reshape(*z.shape[:-1], -1), vq_loss

#
#
class VQVAE(nn.Module):
    def __init__(self, conf, network_conf, codebook_size, num_subvectors, code_dim, beta=0.25):
        """
        Params (in conf): the same encoder/decoder config dicts FSQVAE takes (its `levels`
            entry, if present, is ignored)
            ut_encoder_layer_config: config dict for TransformerEncoderLayer at UT
            bs_encoder_layer_config: config dict for TransformerEncoderLayer at BS
            bs_encoder_config: config dict for TransformerEncoder at BS

        codebook_size, num_subvectors, code_dim: see get_vq_config() in common_setup.py
        beta: commitment loss weight
        """
        super().__init__()

        # config dictionaries for transformer heads
        ut_encoder_layer_config = conf['ut_encoder_layer_config']
        bs_encoder_layer_config = conf['bs_encoder_layer_config']
        bs_encoder_config = conf['bs_encoder_config']

        # parameters for embedding
        self.codebook_size = codebook_size
        self.num_subvectors = num_subvectors
        self.unquantized_latent_dim = num_subvectors * code_dim

        # network parameters
        self.N, self.J = network_conf['N'], network_conf['J']
        self.Td = network_conf['Td']
        self.use_dft_matrix = network_conf['use_dft_matrix']

        self.relu = nn.ReLU()

        # everything for the UTs
        self.dm_ut = ut_encoder_layer_config['d_model']

        self.encoder_seq = nn.Sequential(nn.Linear(2*self.Td, 2*self.dm_ut),
                                         self.relu,
                                         nn.Linear(2*self.dm_ut, 2*self.dm_ut),
                                         self.relu,
                                         nn.Linear(2*self.dm_ut, 1*self.dm_ut),
                                         self.relu,
                                         nn.Linear(1*self.dm_ut, self.unquantized_latent_dim))

        # everything for the BS
        self.vq = VectorQuantizer(codebook_size, code_dim, beta=beta)

        self.dm_bs = bs_encoder_layer_config['d_model']
        self.dec_ff_d_model = nn.Linear(self.unquantized_latent_dim, self.dm_bs)
        self.dec_pos_encoding = PositionalEncoding(d_model=self.dm_bs,
                                                   max_len=self.J)
        self.decoder = EncoderHead(bs_encoder_layer_config,
                                   bs_encoder_config,
                                   self.dec_pos_encoding)
        self.dec_ff_to_2N = nn.Linear(self.dm_bs, 2*self.N)

        if self.use_dft_matrix:
            self.pilot_matrix = network_conf['pilot_matrix']
        else:
            self.pilot_matrix = nn.Parameter(torch.randn((self.Td, self.N), dtype=torch.cfloat), requires_grad=True)
            parametrize.register_parametrization(self, "pilot_matrix", UnitNormRows())

    def forward(self, obs):
        H, noise = obs
        Y = self.pilot_matrix @ (H.conj()) + noise
        Y = torch.cat([Y.real, Y.imag], dim=-2).requires_grad_(True) # B x 2Td x J

        yy = Y.contiguous()
        bs = yy.shape[0]

        x = self.encoder_seq(yy.permute(0,2,1).reshape(-1, 2*self.Td)).reshape(bs, self.J, -1) # B x J x unquant_latent_dim

        # quantize the latent space, one shared codebook across every sub-vector
        x_quantized, x_quant_indices, vq_loss = self.vq(x) # B x J x unquant_latent_dim, B x J x num_subvectors, scalar

        z = self.dec_ff_d_model(x_quantized) # B x J x d_model_bs
        x_recon = self.decoder(self.relu(z), src_key_padding_mask=None) # B x J x d_model_bs
        x_recon = self.dec_ff_to_2N(self.relu(x_recon)) # B x J x 2N
        # normalize each matrix to have unit norm to satisfy power constraint
        x_recon = x_recon / torch.linalg.matrix_norm(x_recon, ord='fro', keepdim=True)

        return x_recon, x_quant_indices, vq_loss
