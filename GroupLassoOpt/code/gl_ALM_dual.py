import numpy as np
import utils, time

def proj_group_l2_ball(Q, mu):
    """Project each row of Q onto l2-ball of radius mu."""
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    scaling = np.minimum(1.0, mu / (norms + 1e-12))
    return scaling * Q

def ALM_dual_opts_init():
    opts = {
        'sigma' : 10.0,        # 罚参数
        'max_outer' : 20,      # 最大外层迭代
        'max_inner' : 200,     # 每轮最大内层迭代
        'tol_inner' : 1e-6,     # 内层收敛容差
        'window_size' : 2,
        'rho' : 2.0,
        'tol_f' : 1e-7,
    }
    return opts

def gl_ALM_dual(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts:dict = {}):

    x = x0.copy()
    m, n = A.shape
    _, l = b.shape
    z = np.zeros((n, l))

    if not opts:
        opts = ALM_dual_opts_init()
        
    sigma = opts['sigma']     
    max_outer = opts['max_outer']      # 最大外层迭代
    max_inner = opts['max_inner']     # 每轮最大内层迭代
    tol_inner = opts['tol_inner']    # 内层收敛容差
    window_size = opts['window_size']
    rho = opts['rho']
    tol_f = opts['tol_f']


    x_opt = x.copy()
    best_obj = 0.5 * np.sum( (A@x - b)**2 )+ mu * np.sum(np.linalg.norm(x, axis=1))

    f_values = [best_obj]

    for k_out in range(max_outer):

        M = np.eye(m) + sigma * (A @ A.T)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        z_prev = z.copy()

        cnt_inner = 0
        for k_in in range(max_inner):

            cnt_inner += 1
            temp = A @ x - sigma * A @ z - b
            y = M_inv @ temp

            Q = x / sigma - A.T @ y
           
            z = proj_group_l2_ball(Q, mu)

            if np.linalg.norm(z - z_prev, 'fro') < tol_inner:
                break
            z_prev = z.copy()

        # print(k_out, " ", cnt_inner)

        x = x - sigma * (A.T @ y + z)

        obj = 0.5 * np.sum( (A@x - b)**2 )+ mu * np.sum(np.linalg.norm(x, axis=1))
        f_values.append(obj)
        if obj< best_obj:
            x_opt = x.copy()
            best_obj = obj

        if len(f_values) >= window_size + 1:
            recent_vals = f_values[-window_size-1:]
            if np.max(recent_vals) - np.min(recent_vals) < tol_f:
                break

        sigma *= rho
        tol_inner /= rho


    return x_opt, len(f_values)-1, f_values


if __name__ == "__main__":

    A = np.load("code/datas/A.npy")
    b = np.load("code/datas/b.npy")
    u = np.load("code/datas/u.npy")
    mu = 0.01

    m, n = A.shape
    _, l = b.shape

    x0 = np.zeros((n, l))
    
    start = time.time()
    x_opt, iter_count, f_values = gl_ALM_dual(x0, A, b, mu)
    end = time.time()

    f_opt = min(f_values)
    regular_x_opt = mu * np.sum(np.linalg.norm(x_opt, axis=1))

    print(f"运行时间: {end - start:.6f} 秒")
    print(f"迭代次数: {iter_count}")
    print(f"求得目标函数最小值: {f_opt:.6f}")
    print(f"正则项: {regular_x_opt:.6f}, 光滑项: {f_opt - regular_x_opt:.6f}")
    print(f"解的非零元比例: {utils.compute_nonzero_ratio(x_opt)}")
