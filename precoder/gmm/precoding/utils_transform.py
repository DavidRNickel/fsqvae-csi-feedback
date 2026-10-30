"""
Transform-coding channel quantizer: Karhunen-Loeve transform (KLT) + reverse
water-filling bit allocation + uniform scalar quantization per coefficient.

A channel h ~ CN(0, R) has independent KLT coefficients c = U^H h where
R = U diag(eigvals) U^H, c_i ~ CN(0, eigvals[i]). Splitting each c_i into its
real and imaginary parts gives 2N independent real Gaussian coordinates, each
with variance eigvals[i]/2. A fixed total bit budget is then split across
those 2N coordinates via reverse water-filling (more bits where the variance
is larger, zero bits below a cutoff -- Cover & Thomas, "Elements of
Information Theory", 2nd ed., Theorem 10.3.3), and each coordinate is
uniform-scalar-quantized at its assigned rate.

This needs no training data or codebook fitting (unlike cvq_proj.py's
K-means codebook) -- only the eigendecomposition of a correlation-matrix
estimate the caller already has.
"""
import numpy as np


def reverse_waterfilling_bits(variances, total_bits):
    """
    Greedy integer bit allocation across independent real Gaussian scalar
    sources with per-coordinate variance `variances[i]`, converging to the
    reverse-water-filling solution: minimize
        sum_i variances[i] * 2**(-2*bits[i])   subject to   sum(bits) = total_bits.
    Each of the `total_bits` individual bits is greedily given to whichever
    coordinate currently has the largest marginal distortion reduction
    (Segall, "Bit allocation and encoding for vector sources," IEEE Trans.
    Inf. Theory, 1976; Gersho & Gray, "Vector Quantization and Signal
    Compression," 1992, Ch. 8) -- equivalent to reverse water-filling for
    integer rates, without needing to solve for the water level directly.

    variances: (M,) array of per-coordinate source variances.
    total_bits: int, total bits to distribute across the M coordinates.
    Returns: (M,) int array of bits per coordinate, summing to total_bits.
    """
    variances = np.asarray(variances, dtype=np.float64)
    M = len(variances)
    bits = np.zeros(M, dtype=np.int64)
    for _ in range(total_bits):
        # D_i(b) - D_i(b+1) = variances[i]*2**(-2b)*(3/4); the constant 3/4 doesn't
        # affect the argmax, so compare current distortion variances[i]*2**(-2*bits[i])
        current_D = variances * (2.0 ** (-2.0 * bits))
        i = int(np.argmax(current_D))
        bits[i] += 1
    return bits


def quantize_scalar_uniform(values, bits, std, clip_sigma=3.5):
    """
    Independent uniform scalar quantization of each entry of `values`, using
    2**bits[i] levels spanning +/- clip_sigma*std[i]. `bits`/`std` broadcast
    against `values`' shape (e.g. one bits/std value per column, shared
    across rows/samples).

    A coordinate with bits==0 has exactly 1 "level" spanning the whole clip
    range, whose center is exactly 0 -- so it naturally reconstructs as 0
    (the MMSE estimate of a zero-mean Gaussian given zero bits of
    information) with no special-casing needed.

    This is a practical simplification of the PDF-matched, non-uniform
    Lloyd-Max quantizer (Lloyd 1957/1982; Max 1960) that would be exactly
    rate-distortion-optimal for a Gaussian source. It
    already captures the effect being tested here (spending rate where the
    eigenvalues are largest, instead of uniformly across all antennas); a
    true Lloyd-Max quantizer would tighten the constant factor further
    without changing that story.
    """
    values = np.asarray(values, dtype=np.float64)
    bits = np.broadcast_to(np.asarray(bits, dtype=np.float64), values.shape)
    std = np.broadcast_to(np.asarray(std, dtype=np.float64), values.shape)

    num_levels = 2.0 ** bits  # bits==0 -> 1 level
    half_range = clip_sigma * np.maximum(std, 1e-12)
    step = (2.0 * half_range) / num_levels

    clipped = np.clip(values, -half_range, half_range - 1e-9 * step)
    level_idx = np.clip(np.floor((clipped + half_range) / step), 0, num_levels - 1)
    return -half_range + (level_idx + 0.5) * step


def transform_code_channels(H_est, eigvecs, eigvals, total_bits, clip_sigma=3.5):
    """
    KLT-transform-code a batch of channel estimates against a shared
    correlation-matrix eigenbasis, allocating `total_bits` per user across
    the 2N real/imaginary KLT coefficients via reverse water-filling, then
    uniform-scalar-quantizing each one.

    H_est: (..., N) complex array of channel estimates (any leading batch
           shape; flattened internally and reshaped back).
    eigvecs: (N, N) complex array, columns = eigenvectors of the receiver's
             correlation-matrix estimate R (so c = U^H h, h = U c).
    eigvals: (N,) real array, eigenvalues corresponding to eigvecs' columns
             (any order -- reverse water-filling doesn't require sorting).
    total_bits: total bits to spend per user, split across the 2N real KLT
                coordinates.
    Returns: array the same shape as H_est -- the transform-coded (quantized
             then reconstructed) channel estimate.
    """
    orig_shape = H_est.shape
    N = orig_shape[-1]
    H_flat = H_est.reshape(-1, N)

    # analysis: c = U^H h: for row vectors, c_row = h_row @ conj(U)
    C = H_flat @ eigvecs.conj()

    variances_per_eig = np.maximum(np.asarray(eigvals).real, 0.0) / 2.0  # (N,): each real/imag part's variance
    variances_2N = np.repeat(variances_per_eig, 2)  # interleaved [v0,v0,v1,v1,...] for reverse_waterfilling_bits
    bits_2N = reverse_waterfilling_bits(variances_2N, total_bits)
    bits_re, bits_im = bits_2N[0::2], bits_2N[1::2]  # (N,) each

    std = np.sqrt(variances_per_eig)  # (N,), shared by real and imaginary parts

    C_hat_re = quantize_scalar_uniform(C.real, bits_re[None, :], std[None, :], clip_sigma)
    C_hat_im = quantize_scalar_uniform(C.imag, bits_im[None, :], std[None, :], clip_sigma)
    C_hat = C_hat_re + 1j * C_hat_im

    # synthesis: h = U c: for row vectors, h_row = c_row @ U^T
    H_hat_flat = C_hat @ eigvecs.T
    return H_hat_flat.reshape(orig_shape)
