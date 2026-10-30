import deepmimo as dm
import os, sys
import numpy as np
from numba import njit, prange

DISTANCE_THRESHOLD = 8.0  # meters
# Function generated with the help of an LLM
@njit(fastmath=True)
def compute_correlation_matrix(channels_2d, user_locations):
    """
    channels_2d: shape (num_users, N)
    user_locations: shape (num_users, 3)
    """
    num_users = user_locations.shape[0]
    num_bs_ant = channels_2d.shape[1]
    
    corr_matrices = np.zeros((num_users, num_bs_ant, num_bs_ant), dtype=np.complex64)
    outer_products = np.zeros_like(corr_matrices, dtype=np.complex64)

    for i in range(num_users):
        h = channels_2d[i]
        h_col = h.conj().reshape(num_bs_ant, 1)
        h_row = h.reshape(1, num_bs_ant)
        outer_products[i] = h_col @ h_row
    
    for i in range(num_users):
        loc = user_locations[i]
        
        dist = np.sqrt(np.sum((loc - user_locations) ** 2, axis=1))
        close_users = np.where(dist <= DISTANCE_THRESHOLD)[0]
        
        for user in close_users:
            # h = channels_2d[user] 
            # h_col = h.conj().reshape(num_bs_ant, 1)
            # h_row = h.reshape(1, num_bs_ant)
            # corr_matrices[i] += h_col @ h_row
            corr_matrices[i] += outer_products[user]
            
        if close_users.size > 0:
            corr_matrices[i] /= close_users.size
        else:
            corr_matrices[i] = np.eye(num_bs_ant, dtype=np.complex64)
            
    return corr_matrices


# Function generated with the help of an LLM
@njit(fastmath=True)
def compute_correlation_matrix_sqrt(channels_2d, user_locations):
    """
    channels_2d: shape (num_users, N)
    user_locations: shape (num_users, 3)
    """
    num_users = user_locations.shape[0]
    num_bs_ant = channels_2d.shape[1]
    
    # This will now store the Hermitian square root matrices (U * Sigma^1/2 * U^H)
    corr_mtx_sqrts = np.zeros((num_users, num_bs_ant, num_bs_ant), dtype=np.complex64)
    outer_products = np.zeros_like(corr_mtx_sqrts, dtype=np.complex64)
    
    for i in range(num_users):
        h = channels_2d[i]
        h_col = h.reshape(num_bs_ant, 1)
        h_row = h.conj().reshape(1, num_bs_ant)
        outer_products[i] = h_col @ h_row
    
    
    for i in prange(num_users):
        loc = user_locations[i]
        
        # Calculate distance to all other users
        dist = np.sqrt(np.sum((loc - user_locations) ** 2, axis=1))
        close_users = np.where(dist <= DISTANCE_THRESHOLD)[0]
        
        # 1. Compute the raw Hermitian spatial correlation matrix R = E[h h^H]
        R_temp = np.zeros((num_bs_ant, num_bs_ant), dtype=np.complex64)
        for user in close_users:
            R_temp += outer_products[user]
            
        if close_users.size > 0:
            R_temp /= close_users.size
            
            # 2. Compute Eigenvalue Decomposition via Numba-compatible np.linalg.eigh
            # eigh yields real eigenvalues (S) and complex unitary eigenvectors (U)
            S, U = np.linalg.eigh(R_temp)
            
            # 3. Clean numerical noise (< 0) and take the square root of eigenvalues
            S_clipped = np.maximum(S, 0.0)
            S_sqrt = np.sqrt(S_clipped)
            
            # 4. Construct the Hermitian square root U * Sigma^(1/2) * U^H, using row-wise
            # broadcasting to avoid creating a full diagonal matrix in memory. The trailing U^H
            # matters: U * Sigma^(1/2) alone also squares to R, but rotating the DFT codebook by
            # it equals R^(1/2) @ (U @ codebook), which scrambles the steering-vector codewords
            # (U @ c) before the rotation and costs ~1-3.5 bits/s/Hz of sum rate.
            # (S_sqrt is promoted to float64 by np.maximum, so cast back to keep both matmul operands complex64)
            U_S_sqrt = U * S_sqrt.astype(np.float32).reshape(1, num_bs_ant)
            corr_mtx_sqrts[i] = U_S_sqrt @ np.ascontiguousarray(U.conj().T)
            
        else:
            # If no close users, return identity matrix (identity matrix is its own sqrt)
            corr_mtx_sqrts[i] = np.eye(num_bs_ant, dtype=np.complex64)
            
    return corr_mtx_sqrts


if __name__ == '__main__':
    scenario = 'asu_campus_3p5'
    # scenario = 'i2_28b'
    # BS antenna count from the command line (python3 make_channels.py 32); run once per N in {4, 8, 32}
    N_tx = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    N_rx = 1
    savedir = os.path.join('channel_data', scenario, f'{N_tx}x{N_rx}')
    
    dm.download(scenario)
    dataset = dm.load(scenario)
    
    params = dm.ChannelParameters()
    params.bs_antenna.shape = [N_tx, 1]
    params.bs_antenna.spacing = 0.5
    params.bs_antenna.rotation = [0, 0, 0]
    params.ue_antenna.shape = [N_rx, 1]
    params.ue_antenna.spacing = 0.5
    params.ue_antenna.rotation = [0, 0, 0]
    params.doppler = False
    params.freq_domain = False
    params.num_paths = 1
    
    # Extract channels and strip the paths dimension -> (num_ues, 1, N)
    channels = dataset.compute_channels(params).squeeze(-1)
    
    # Create mask and apply it to BOTH channels and locations uniformly
    valid_channel_mask = ~np.isclose(channels, np.zeros_like(channels)).all(axis=(1, 2))
    channels = channels[valid_channel_mask]
    
    user_locations = dataset.rx_pos.astype(np.float64)
    user_locations = user_locations[valid_channel_mask] 
    
    # Flatten the unneeded N_rx=1 dimension to make it an easy (num_uts, N) array
    channels_2d = channels[:, 0, :]
    
    # Normalize globally to stabilize simulation scale
    channels_2d /= np.std(channels_2d)
    
    # Compute the true spatial correlations
    print('Computing correlation matrices...')
    corr_matrices = compute_correlation_matrix(channels_2d, user_locations)
    print('Computing correlation matrix square roots...')
    corr_matrices_sqrt = compute_correlation_matrix_sqrt(channels_2d, user_locations)
    
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        
    np.save(os.path.join(savedir, 'channels.npy'), channels_2d)
    np.save(os.path.join(savedir, 'corr_matrices.npy'), corr_matrices)
    np.save(os.path.join(savedir, 'corr_matrices_sqrt.npy'), corr_matrices_sqrt)
    print("Data processed and saved successfully without rank deficiency!")
