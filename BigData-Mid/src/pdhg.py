import numpy as np
import time

def project_nonnegative(v):
    """投影到非负正交空间"""
    return np.maximum(v, 0)

def pdhg_solver(mu, Sigma, gamma=1.0, tau=0.1, sigma=0.1, max_iter=500, tol=1e-6, verbose=True):
    """
    PDHG 求解 Model A2: min_x gamma/2 x^T Sigma x - mu^T x s.t. 1^T x = 1, x >= 0
    """
    n = len(mu)
    x = np.ones(n)/n
    x_bar = x.copy()
    p = 0.0   # 对应 1^T x = 1 的对偶
    q = np.zeros(n)  # 对应 x >= 0 的对偶

    I = np.eye(n)
    Sigma_mat = gamma * Sigma

    obj_history = []
    feas_history = []

    start_time = time.time()
    for k in range(max_iter):
        # dual update
        p = p + sigma*(np.sum(x_bar) - 1)
        q = project_nonnegative(q - sigma*x_bar)
        # primal update
        x_old = x.copy()
        x = np.linalg.solve(I + tau*Sigma_mat, x - tau*(p - q) + tau*mu)
        # extrapolation
        x_bar = x + (x - x_old)

        obj_val = 0.5*gamma*x.T @ Sigma @ x - mu.T @ x
        feas_resid = np.abs(np.sum(x) - 1) + np.linalg.norm(np.minimum(x, 0))

        obj_history.append(obj_val)
        feas_history.append(feas_resid)

        if verbose and k % 50 == 0:
            print(f"Iter {k}: obj={obj_val:.6f}, feasibility={feas_resid:.2e}")

        if feas_resid < tol:
            break
    end_time = time.time()
    return {
        "x": x,
        "obj_history": obj_history,
        "feas_history": feas_history,
        "time": end_time - start_time,
        "iterations": k+1
    }