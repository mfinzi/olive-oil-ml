

from ..lazy_matrix import Lazy, LazyMatrix
from ..lazy_types import I
from ...utils.mytqdm import tqdm

   

def lanczos(A,max_iter=None,tol=1e-10,keep_basis=True,reorthog=True,B=I):
    B_norm = lambda v: A.xp.sqrt(v.T@B@v)
    n = A.shape[-1]
    r = A.xp.randn(n) 
    alphas = A.xp.zeros(n); betas = A.xp.zeros(n)
    if keep_basis: Q = A.xp.zeros((n,max_iter))
    q_prev = 0; q = r/B_norm(r)
    for i in tqdm(range(max_iter or n)):
        Aq = A@q
        alphas[i] = q.T@B@Aq
        r = Aq - alphas[i]*q - betas[i]*q_prev
        if reorthog and not i%reorthog: r -= Q@(Q.T@(B@r))
        betas[i] = B_norm(r)
        if betas[i] < tol: break
        q, q_prev = r/betas[i], q
        if keep_basis: Q[:,i] = q
    # Construct Tridiagonal from alphas and betas
    T = A.xp.zeros((i,i))
    T[0,0] = alphas[0]
    for j in range(1,i):
        T[j,j] = alphas[j]
        T[j,j-1] = T[j-1,j] = betas[j-1]
    return T, Q if keep_basis else T