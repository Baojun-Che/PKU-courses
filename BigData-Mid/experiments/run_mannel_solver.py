import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.admm import admm_solver
from src.pdhg import pdhg_solver

data_path = "data/estimates_train.npz"
save_path = "results/manual_solver"  # 修复路径拼写错误
os.makedirs(save_path, exist_ok=True)

data = np.load(data_path)
mu = data["mu"]
Sigma = data["Sigma"]

gamma = 1.0

# 测试不同rho的ADMM
rhos = [0.1, 1.0, 10.0, 100.0]
admm_results = {}
for rho in rhos:
    res = admm_solver(mu, Sigma, gamma=gamma, rho=rho)
    admm_results[rho] = res
    np.savez(os.path.join(save_path, f"admm_rho{rho}.npz"), **res)

# 测试不同步长的PDHG
taus = [0.01, 0.05, 0.1]
sigmas = [0.01, 0.05, 0.1]
pdhg_results = {}
for tau in taus:
    for sigma in sigmas:
        if tau*sigma*(len(mu)+1) >= 1:
            continue  # 不满足稳定性条件
        res = pdhg_solver(mu, Sigma, gamma=gamma, tau=tau, sigma=sigma)
        pdhg_results[(tau, sigma)] = res
        np.savez(os.path.join(save_path, f"pdhg_tau{tau}_sigma{sigma}.npz"), **res)

# 绘图示例：ADMM目标值收敛
plt.figure()
for rho, res in admm_results.items():
    plt.plot(res["obj_history"], label=f"rho={rho}")
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("ADMM Convergence Curve")
plt.legend()
plt.savefig(os.path.join(save_path, "admm_obj_history.png"))

# PDHG 可行性收敛曲线
plt.figure()
for (tau, sigma), res in pdhg_results.items():
    plt.plot(res["feas_history"], label=f"tau={tau}, sigma={sigma}")
plt.xlabel("Iteration")
plt.ylabel("Feasibility Residuals")
plt.title("PDHG Feasibility Convergence Curve")
plt.legend()
plt.savefig(os.path.join(save_path, "pdhg_feas_history.png"))