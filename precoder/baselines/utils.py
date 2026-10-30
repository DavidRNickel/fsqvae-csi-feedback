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

#
# Find the closest point in the codebook to each channel estimate at the UT
# by applying the KDTree algorithm to the real and imaginary parts for 
# computational efficiency.
def quantize_channels(channels, codebook, tree, magnitude_params=None):
    channels_split = np.hstack((channels.real, channels.imag))
    _, closest_codeword_indices = tree.query(channels_split, workers=-1)
    codewords = codebook[closest_codeword_indices]
    
    if magnitude_params is not None:
        assert len(magnitude_params) == 2, "magnitude_params must be a tuple of (num_bits, max_val)"
        num_bits, max_val = magnitude_params
        effective_gains = np.abs((channels.conj() * codewords).sum(axis=1))
        _, magnitudes = quantize_magnitude(effective_gains, num_bits, max_val)
    else:
        return codewords
        
    return magnitudes * codewords

#
# Invert the channel at the UT. Note that this finds the conjugate.
def estimate_channel_mp_pinv(channels, pilots, mp_pinv, noise_pwr):
    if channels.ndim ==2:
        y = pilots @ channels.conj().T 
    else:
        y = pilots @ channels.conj().transpose(0,2,1)
    y = y + np.sqrt(noise_pwr / 2) * (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape))
    h_est = mp_pinv @ y

    return h_est.conj()

#
# Invert the channel at the UT. Note that this finds the conjugate,
# hence we return the conjugate.
def estimate_channel_corr_mtx(channels, corr_mtxs, pilots, noise_pwr):
    if channels.ndim ==2:
        y = pilots @ channels.conj().T 
    else:
        y = pilots @ channels.conj().transpose(0,2,1)
    y = y + np.sqrt(noise_pwr / 2) * (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape))

    num_groups, num_uts, num_bs_ant = channels.shape
    num_pilots = pilots.shape[0]
    h_est = np.zeros((num_groups, num_bs_ant, num_uts), dtype=np.complex64)
    I = np.eye(num_pilots)
    pilots_H = pilots.conj().T
    for i in range(num_groups):
        R = corr_mtxs[i]
        for j in range(num_uts):
            tmp1 = pilots @ R[j] @ pilots_H + noise_pwr * I
            tmp2 = np.linalg.solve(tmp1, I) 
            h_est[i,:,j] = (R[j] @ pilots_H @ tmp2 @ y[i,:,j].reshape(num_pilots, 1)).reshape(num_bs_ant,)

    return h_est.conj()