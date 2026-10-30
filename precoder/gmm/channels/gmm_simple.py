import torch
import os

def generate_random_cov_mtxs(K, N, device):
    covs = torch.empty((K, N, N), dtype=torch.cfloat)
    for k in range(K):
        X = torch.randn((N,N), dtype=torch.cfloat)
        R = X @ X.conj().T
        covs[k] = (N / R.trace()) * R
    covs = covs.to(device)
    return covs, torch.linalg.cholesky(covs)

def get_cov_mtx_sqrts(L):
    U, S, _ = torch.linalg.svd(L)
    S_diag = torch.diag_embed(S).to(L.dtype)
    return U @ S_diag @ U.adjoint()

# def sample_gmm(L, J, idxs, device):
#     bs, N = idxs.shape[0], L.shape[-1]
#     z = torch.randn(bs, N, J, dtype=torch.cfloat, device=device) # (B, N, J)
#     decomps = L[idxs] # (B, N, N)
#     return torch.bmm(decomps, z)

def sample_gmm(L, idxs, device):
    # Assumes idxs shape is (bs, J) where each user gets an index
    bs, J = idxs.shape
    N = L.shape[-1]
    
    # 1. Sample standard normal noise independently for each user
    # Shape: (bs, J, N, 1) to treat each user's noise as a column vector
    z = torch.randn(bs, J, N, 1, dtype=torch.cfloat, device=device)
    
    # 2. Extract the unique covariance decompositions for each user
    # Shape: (bs, J, N, N)
    decomps = L[idxs] 
    
    # 3. Apply the transformation per user using batched multiplication
    # (bs * J, N, N) @ (bs * J, N, 1) -> (bs * J, N, 1)
    chans = torch.matmul(decomps, z)
    
    # 4. Squeeze out the trailing dimension and reshape to (bs, N, J)
    return chans.squeeze(-1).transpose(1, 2)

if __name__=='__main__':
    d='cpu'
    N=16
    K=20
    covs, decomps = generate_random_cov_mtxs(K=K, N=N, device=d)
    ROOT_DIR = f'channel_stats/N{N}'
    os.makedirs(ROOT_DIR,exist_ok=True)
    torch.save(covs, f'{ROOT_DIR}/cov.pt')
    torch.save(decomps, f'{ROOT_DIR}/decomp.pt')
    