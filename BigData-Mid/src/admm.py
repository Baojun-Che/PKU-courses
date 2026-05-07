import numpy as np
import time

def project_simplex(v):
    """
    投影向量 v 到单位单纯形 Delta:
    Delta = { z >= 0, sum(z) = 1 }
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w

def admm_solver(mu, Sigma, gamma=1.0, rho=1.0, max_iter=500, tol=1e-6, verbose=True):
    """
    ADMM 求解 Model A2: min_x gamma/2 x^T Sigma x - mu^T x s.t. 1^T x = 1, x >= 0
    """
    n = len(mu)
    x = np.ones(n)/n
    z = np.ones(n)/n
    u = np.zeros(n)
    
    Sigma_rho = gamma*Sigma + rho*np.eye(n)
    Sigma_rho_inv = np.linalg.inv(Sigma_rho)

    obj_history = []
    prim_resid_history = []
    dual_resid_history = []

    start_time = time.time()
    for k in range(max_iter):
        # x更新
        x = Sigma_rho_inv @ (mu + rho*(z - u))
        # z更新
        z_old = z.copy()
        z = project_simplex(x + u)
        # u更新
        u = u + x - z
        
        # 记录残差和目标值
        r_prim = np.linalg.norm(x - z)
        r_dual = rho * np.linalg.norm(z - z_old)
        obj_val = 0.5*gamma*x.T @ Sigma @ x - mu.T @ x

        prim_resid_history.append(r_prim)
        dual_resid_history.append(r_dual)
        obj_history.append(obj_val)

        if verbose and k % 50 == 0:
            print(f"Iter {k}: obj={obj_val:.6f}, r_prim={r_prim:.2e}, r_dual={r_dual:.2e}")

        if r_prim < tol and r_dual < tol:
            break

    end_time = time.time()
    return {
        "x": x,
        "obj_history": obj_history,
        "prim_resid_history": prim_resid_history,
        "dual_resid_history": dual_resid_history,
        "time": end_time - start_time,
        "iterations": k+1
    }