import numpy as np
import time
import utils

def proj_group_l2_ball(Q, mu):
    """Project each row of Q onto l2-ball of radius mu."""
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    scaling = np.minimum(1.0, mu / (norms + 1e-12))
    return scaling * Q

def gl_ADMM_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts:dict = {}):
    
    m, n = A.shape
    _, l = b.shape

    x = np.copy(x0)
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    max_iter = 3000
    rho = 0.5
    tol_f = 1e-6
    tol_res = 1e-8
    window_size = 20

    iter_count = 0
    f_values = []
    M = rho * np.eye(n) + A.T @ A

    x_opt = np.zeros_like(x)
    best_obj = 0.5 * np.sum(b**2)

    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)

    for k in range(max_iter):
        
        r = A @ x - b
        obj =  0.5 * np.sum(r**2) + mu * np.sum(np.linalg.norm(x, axis=1))
        f_values.append(obj)

        if obj < best_obj:
            x_opt = x.copy()
            best_obj = obj

        temp = A.T @ b + rho * z - y
        x_new = M_inv @ temp

        Q = x_new + y / rho
        z_new = utils.prox_group_lasso(Q, mu/rho)

        iter_count += 1
        x, z = x_new, z_new

        if len(f_values) >= window_size + 1:
            recent_vals = f_values[-window_size-1:]
            if np.max(recent_vals) - np.min(recent_vals) < tol_f:
                break

        res = z - x

        if np.sum( res**2 ) < tol_res**2:
            break

        y = y - rho * res

    r = A @ x - b
    obj =  0.5 * np.sum(r**2) + mu * np.sum(np.linalg.norm(x, axis=1))
    f_values.append(obj)
    if obj < best_obj:
        x_opt = x.copy()
        best_obj = obj
    
    return x_opt, iter_count, f_values



if __name__ == "__main__":

    A = np.load("code/datas/A.npy")
    b = np.load("code/datas/b.npy")
    u = np.load("code/datas/u.npy")
    mu = 0.01

    m, n = A.shape
    _, l = b.shape

    x0 = np.zeros((n, l))
    
    start = time.time()
    x_opt, iter_count, f_values = gl_ADMM_primal(x0, A, b, mu)
    end = time.time()

    regular_x_opt = mu * np.sum(np.linalg.norm(x_opt, axis=1))
    smooth_x_opt = 0.5 * np.sum((A @ x_opt - b)**2) 
    f_opt = smooth_x_opt + regular_x_opt

    print(f"运行时间: {end - start:.6f} 秒")
    print(f"迭代次数: {iter_count}")
    print(f"求得目标函数最小值: {f_opt:.6f}")
    print(f"正则项: {regular_x_opt:.6f}, 光滑项: {f_opt - regular_x_opt:.6f}")
    print(f"解的非零元比例: {utils.compute_nonzero_ratio(x_opt)}")

    utils.plot_relative_error(f_values, "ADMM_primal", 0.6705752210556729)