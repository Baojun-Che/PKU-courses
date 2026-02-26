import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time, utils

def gl_gurobi(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts:dict = {}):
   
    m, n = A.shape
    l = b.shape[1]
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)  # 关闭输出
    
    X = model.addMVar((n, l), lb=-GRB.INFINITY)
    X.start = x0  # 设置初始解
    
    # 变量 Y (m x l) 用于残差
    Y = model.addMVar((m, l), lb=-GRB.INFINITY)
    
    # 变量 t (n) 用于组范数
    t = model.addMVar(n, lb=0.0)
    
    # 约束: Y = A X - b
    for j in range(l):
        model.addConstr(A @ X[:, j] - b[:, j] == Y[:, j])
    
    # 二阶锥约束: t_i >= ||X[i, :]||_2，等价于 t_i^2 >= X[i, :] @ X[i, :]
    for i in range(n):
        model.addConstr(X[i, :] @ X[i, :] <= t[i] * t[i])
    
    # 目标函数: 0.5 * ||Y||_F^2 + mu * sum(t)
    model.setObjective(0.5 * sum(Y[:, j] @ Y[:, j] for j in range(l)) + mu * sum(t), GRB.MINIMIZE)
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        x_opt = X.X
        obj_val = model.objVal
        return x_opt, -1, [obj_val]
    else:
        raise RuntimeError(f"Gurobi 求解失败，状态码: {model.status}")
    
if __name__ == "__main__":

    A = np.load("code/datas/A.npy")
    b = np.load("code/datas/b.npy")
    u = np.load("code/datas/u.npy")
    mu = 0.01

    m, n = A.shape
    _, l = b.shape

    x0 = np.zeros((n, l))
    
    start = time.time()
    x_opt, iter_count, f_values = gl_gurobi(x0, A, b, mu)
    end = time.time()

    f_opt = min(f_values)
    regular_x_opt = mu * np.sum(np.linalg.norm(x_opt, axis=1))

    print(f"运行时间: {end - start:.6f} 秒")
    print(f"迭代次数: {iter_count}")
    print(f"求得目标函数最小值: {f_opt:.6f}")
    print(f"正则项: {regular_x_opt:.6f}, 光滑项: {f_opt - regular_x_opt:.6f}")
    print(f"解的非零元比例: {utils.compute_nonzero_ratio(x_opt)}")

