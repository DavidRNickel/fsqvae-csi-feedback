import numpy as np

#
#
def generate_dft_codebook(num_antennas, oversampling_factor):
    total_codewords = num_antennas * oversampling_factor
    ant_idx = np.arange(num_antennas).reshape(-1, 1)  # Shape: (num_antennas, 1)
    cw_idx = np.arange(total_codewords).reshape(1, -1)  # Shape: (1, total_codewords)
    phases = -1j * 2 * np.pi * (ant_idx * cw_idx) / total_codewords
    codebook = np.exp(phases)
    codebook /= np.sqrt(num_antennas)
    # tranpose to match the logic of other codebook generation schemes 
    return codebook.T

#
# Function generated with the help of an LLM
def quantize_magnitude(magnitude_array, num_bits, max_val=3.0):
    num_levels = 2 ** num_bits
    step_size = max_val / num_levels
    mag_array = np.asarray(magnitude_array, dtype=float)
    clipped_mags = np.clip(mag_array, 0, max_val - 1e-5)
    q_indices = np.floor(clipped_mags / step_size).astype(int)
    reconstructed_magnitudes = (q_indices * step_size) + (step_size / 2)
    
    return q_indices, reconstructed_magnitudes.reshape(-1, 1)

def quantize_channels_rotated(channels, codebook, corr_mtx_sqrts, magnitude_params=None, tree=None):
    if not np.isfinite(channels).all():
        channels = np.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0)

    if tree is not None:
        channels_split = np.hstack((channels.real, channels.imag))
        _, closest_codeword_indices = tree.query(channels_split, workers=-1)
        codewords = codebook[closest_codeword_indices]
    
    else:
        total_users, N = channels.shape
        corr_mtx_sqrts_3d = corr_mtx_sqrts.reshape(total_users, N, N)
        
        rotated_codebook = corr_mtx_sqrts_3d @ codebook.T
        
        norms = np.linalg.norm(rotated_codebook, axis=1, keepdims=True)
        if magnitude_params is not None:
            rotated_C_normalized = rotated_codebook / (norms + 1e-12)
        else:
            rotated_C_normalized = rotated_codebook
        
        similarity = channels.conj()[:, np.newaxis, :] @ rotated_C_normalized
        similarity = similarity.squeeze(1) 
        projections = np.abs(similarity)
        closest_codeword_indices = projections.argmax(axis=1)
        codewords = np.take_along_axis(rotated_C_normalized, closest_codeword_indices[:, np.newaxis, np.newaxis], axis=2).squeeze(2)  # Squeezes out the trailing 1, leaving shape (Total_Users, N) 
    
    if magnitude_params is not None:
        if tree is not None:
            magnitudes_raw = np.abs(channels.conj() * codewords).sum(axis=1)
        else:
            magnitudes_raw = projections.max(axis=1)
        num_bits, max_val = magnitude_params
        _, magnitudes = quantize_magnitude(magnitudes_raw, num_bits, max_val)
        # idxs = np.abs(projections - magnitude_params).argmin(axis=-1)

        # # Get the actual closest values
        # magnitudes = magnitude_params[idxs].reshape(-1, 1)
        return magnitudes * codewords
    else:
        return codewords
    
