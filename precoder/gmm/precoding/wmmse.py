import sys
import numpy as np
from numba import njit

@njit(fastmath=True)
def sum_rate(H, V, sigma_2):
    J, R, N = H.shape # num_uts, num_rx_ant(=1), num_bs_ant
    V_outer = np.zeros((J, N, N), dtype=np.complex128)
    I_R = np.eye(R, dtype=np.complex128)
    rates = np.zeros(J, dtype=np.float64)
    
    for j in range(J):
        V_outer[j] = V[j] @ V[j].conj().T
        
    for j in range(J):
        # Reset tmp for every user
        tmp = np.zeros((R, R), dtype=np.complex128)
        for ell in range(J):
            if j != ell:
                tmp += H[j] @ V_outer[ell] @ H[j].conj().T
        tmp += sigma_2[j] * I_R
        
        inv_tmp = np.linalg.solve(tmp, I_R)
        argument = I_R + H[j] @ V_outer[j] @ H[j].conj().T @ inv_tmp
        
        # Safe determinant parsing for Numba
        det_val = np.linalg.det(argument)
        rates[j] = np.log2(det_val.real)
        
    return rates

@njit(fastmath=True)
def calc_U(H, V, sigma_2, R=1):
    J, N, d = V.shape # num_uts, num_bs_ant, num_datastreams
    U = np.zeros((J, R, d), dtype=np.complex128)
    I_R = np.eye(R, dtype=np.complex128)
    
    V_outer = np.zeros((J, N, N), dtype=np.complex128)
    for ell in range(J):
        V_outer[ell] = V[ell] @ V[ell].conj().T
        
    for j in range(J):
        tmp = np.zeros((R, R), dtype=np.complex128)
        for ell in range(J):
            tmp += H[j] @ V_outer[ell] @ H[j].conj().T
        tmp += sigma_2[j] * I_R
        
        inv_tmp = np.linalg.solve(tmp, I_R)
        U[j] = inv_tmp @ H[j] @ V[j]
    return U

@njit()
def calc_W(U, H, V):
    J, _, d = V.shape
    I_d = np.eye(d, dtype=np.complex128)
    W = np.zeros((J, d, d), dtype=np.complex128)
    
    for j in range(J):
        tmp = U[j].conj().T @ H[j] @ V[j]
        W[j] = np.linalg.solve(I_d - tmp, I_d)
    return W

@njit(fastmath=True)
def calc_V(H, U, W, alphas, P_max, eps=1e-4):
    J, _, N = H.shape # num_uts, num_rx_ant(=1), num_bs_ant
    d = W.shape[2]
    I_N = np.eye(N, dtype=np.complex128)
    
    mu_low = 0.0
    mu_high = 100.0  # Increased range for safer bisection convergence
    max_bisect_iter = 100
    
    V = np.zeros((J, N, d), dtype=np.complex128)
    
    for _ in range(max_bisect_iter):
        mu = (mu_low + mu_high) / 2.0
        mat_sum = np.zeros((N, N), dtype=np.complex128)
        
        for j in range(J):
            mat_sum += alphas[j] * (H[j].conj().T @ U[j] @ W[j] @ U[j].conj().T @ H[j])
        mat_sum += mu * I_N
        
        inv_mat = np.linalg.solve(mat_sum, I_N)
        
        P_tmp = 0.0
        for j in range(J):
            V[j] = alphas[j] * (inv_mat @ H[j].conj().T @ U[j] @ W[j])
            P_tmp += np.sum(np.abs(V[j])**2)
            
        if P_tmp > P_max:
            mu_low = mu
        else:
            mu_high = mu
            
        if np.abs(P_tmp - P_max) <= eps:
            break
            
    return V

@njit(fastmath=True)
def wmmse(H, P_max, sigma_2, alphas, num_datastreams=1, max_wmmse_iter=300, eps=1e-6):
    J, R, N = H.shape # num_uts, num_rx_ant(=1), num_bs_ant
    d = num_datastreams
    
    W = np.zeros((J, d, d), dtype=np.complex128)
    for j in range(J):
        W[j] = np.eye(d, dtype=np.complex128)
        
    V = np.zeros((J, N, d), dtype=np.complex128)
    for j in range(J):
        vv = np.sqrt(0.5) * (np.random.randn(N, d) + 1j * np.random.randn(N, d))
        P_vv = np.sum(np.abs(vv)**2)
        V[j] = np.sqrt(P_max / (J * P_vv)) * vv

    for _ in range(max_wmmse_iter):
        old_logdet_W = 0.0
        for j in range(J):
            old_logdet_W += np.log2(np.linalg.det(W[j]).real)
            
        U = calc_U(H, V, sigma_2, R)
        W = calc_W(U, H, V)
        V = calc_V(H, U, W, alphas, P_max)
        
        new_logdet_W = 0.0
        for j in range(J):
            new_logdet_W += np.log2(np.linalg.det(W[j]).real)
            
        if np.abs(new_logdet_W - old_logdet_W) <= eps:
            break
            
    return V
