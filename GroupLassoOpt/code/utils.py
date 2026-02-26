import numpy as np
import matplotlib.pyplot as plt
from gl_cvx_mosek import gl_cvx_mosek

def compute_nonzero_ratio(x, tol_factor=1e-6):
    x_flat = np.abs(x).ravel()
    max_abs = np.max(x_flat)
    
    if max_abs == 0:
        return 0.0
    
    threshold = tol_factor * max_abs
    num_nonzero = np.sum(x_flat > threshold)
    total = x_flat.size
    return num_nonzero / total

def prox_group_lasso(z, k):
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    scaling = np.maximum(0.0, 1.0 - k / (norms + 1e-8))
    return scaling * z

def test_data_init(seed = 97006855, n = 512, m = 256, l = 2, mu = 0.01, sparse = 0.1, save_data = False):
    
    np.random.seed(seed)

    k = round(n * sparse)
    A = np.random.randn(m, n)

    # 生成索引 p：随机选择前 k 个位置
    p = np.random.permutation(n)[:k]

    # 初始化 u: n x l，只在 p 对应的位置有值
    u = np.zeros((n, l))
    u[p, :] = np.random.randn(k, l)

    b = A @ u

    print(f"目标函数的全局最小值应不大于: { mu * np.sum(np.linalg.norm(u, axis=1))}")

    if save_data:
        # with open('code/datas/obj_opt.txt', 'w') as f:
        #     f.write(str(min(f_values)))

        # np.save("code/datas/opt_mosek.npy", x_opt)
        np.save("code/datas/A.npy", A)
        np.save("code/datas/b.npy", b)
        np.save("code/datas/u.npy", u)
    
    x0 = np.random.randn(n, l)
    return A, b, u, x0

def plot_relative_error(f_values_list, labels, fig_name, obj_opt = -1):

    if obj_opt < 0:
        with open('obj_opt.txt', 'r') as f:
            obj_opt = float(f.read().strip())

    assert obj_opt>0
    N = len(f_values_list)
    assert len(labels) == N
    
    plt.figure(figsize=(8, 5))

    for i in range(N):
        f_values = np.array(f_values_list[i])
        rel_err = (f_values - obj_opt) / obj_opt
        rel_err = np.maximum(rel_err, 1e-16)
        label = labels[i] + f" ({len(f_values)-1} iterations)"
        plt.semilogy(rel_err, linewidth=1.5, label = label)
    
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Error')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig('doc/figs/'+ fig_name + '.pdf', format='pdf')
    plt.close()

def recover_primal_solution(Z: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, eps = 1e-6):

    m, n = A.shape
    _, l = b.shape
    active = np.linalg.norm(Z, axis=1) >= mu - 1e-6
    x = np.zeros((n, l))

    if np.any(active):
        A_active = A[:, active]  # m x n_active
        x_active_part = np.linalg.lstsq(A_active, b, rcond=None)[0]
        x[active, :] = x_active_part
    return x

def relative_error(x_ref, x):
    return np.linalg.norm(x_ref - x, 'fro') / (1 + np.linalg.norm(x_ref - x, 'fro'))


if __name__ == "__main__":
    test_data_init()