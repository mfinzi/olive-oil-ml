

from ..lazy_matrix import Lazy, LazyMatrix
from ..lazy_types import I
from ...utils.mytqdm import tqdm

no_log = lambda *args,**kwargs:None
def power_method(A,max_iter,tol=0,B=I,log=no_log):
    B_norm = lambda v: A.xp.sqrt(v.T@B@v)
    n = A.shape[-1]
    x = A.xp.new_randn(A,[n])
    rayleigh = 0
    for i in tqdm(range(max_iter)):
        v = A@x
        rayleigh, old_rayleigh = x.T@v, rayleigh
        if A.xp.abs(rayleigh - old_rayleigh) < tol: break
        x /= B_norm(v)
        log(x,rayleigh)
    return x


def lanczos(A,max_iter=None,tol=1e-10,keep_basis=True,reorthog=True,B=I):
    B_norm = lambda v: A.xp.sqrt(v.T@B@v)
    n = A.shape[-1]
    alphas = A.xp.new_zeros(A,[n]); betas = A.xp.new_zeros(A,[n])
    # Initial step
    r = A.xp.new_randn(A,[n]) 
    q_prev = 0; q = r/B_norm(r)
    if keep_basis: 
        Q = A.xp.new_zeros(A,(n,max_iter + 1))
        Q[:,0] = q
    Aq = A@q
    alphas[0] = q.T@B@Aq
    r = Aq - alphas[0]*q
    # Other steps
    for i in tqdm(range(1,max_iter)):
        Aq = A@q
        alphas[i] = q.T@B@Aq
        r = Aq - alphas[i]*q - betas[i-1]*q_prev
        #print(r)
        if reorthog and not i%reorthog: r -= Q@(Q.T@(B@r))
        betas[i+1] = B_norm(r)
        if betas[i+1] < tol: break
        q, q_prev = r/betas[i+1], q
        if keep_basis: 
            print(q)
            Q[:,i+1] = q
    # Construct Tridiagonal from alphas and betas
    T = A.xp.new_zeros(A,(i,i))
    T[0,0] = alphas[0]
    for j in range(1,i):
        T[j,j] = alphas[j]
        T[j,j-1] = T[j-1,j] = betas[j-1]
    return T, Q[:,:i+1] if keep_basis else T