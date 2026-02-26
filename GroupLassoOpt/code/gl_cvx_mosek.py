import numpy as np
import cvxpy as cp

def gl_cvx_mosek(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts:dict = {}):
    
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
        prob.solve(solver=cp.MOSEK, verbose=False)
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

