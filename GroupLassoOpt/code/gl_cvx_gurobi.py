import numpy as np
import cvxpy as cp
import utils, time

def gl_cvx_gurobi(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts:dict = {}):

    m, n = A.shape
    _, l = b.shape

    x = cp.Variable((n, l))

    # Build group norm
    group_norm = 0
    for i in range(n):
        group_norm += cp.norm(x[i, :], 2)

    # Objective function
    obj = 0.5 * cp.sum_squares(A @ x - b) + mu * group_norm

    # Problem
    prob = cp.Problem(cp.Minimize(obj))

    # Solve with MOSEK
    try:
        prob.solve(solver=cp.GUROBI, verbose=False)
    except Exception as e:
        raise RuntimeError(f"MOSEK failed to solve the problem: {e}")

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Problem not solved to optimality. Status: {prob.status}")

    x_opt = x.value

    residual = A @ x_opt - b
    frob_sq = 0.5 * np.sum(residual ** 2)
    group_sum = np.sum(np.linalg.norm(x_opt, axis=1))
    f_final = frob_sq + mu * group_sum

    f_values = [f_final]

    return x_opt, -1, f_values


if __name__ == "__main__":

    A = np.load("code/datas/A.npy")
    b = np.load("code/datas/b.npy")
    u = np.load("code/datas/u.npy")
    mu = 0.01

    m, n = A.shape
    _, l = b.shape

    x0 = np.zeros((n, l))
    
    start = time.time()
    x_opt, iter_count, f_values = gl_cvx_gurobi(x0, A, b, mu)
    end = time.time()

    f_opt = min(f_values)
    regular_x_opt = mu * np.sum(np.linalg.norm(x_opt, axis=1))

    print(f"运行时间: {end - start:.6f} 秒")
    print(f"迭代次数: {iter_count}")
    print(f"求得目标函数最小值: {f_opt:.6f}")
    print(f"正则项: {regular_x_opt:.6f}, 光滑项: {f_opt - regular_x_opt:.6f}")
    print(f"解的非零元比例: {utils.compute_nonzero_ratio(x_opt)}")
