import numpy as np
import math
import os
from scipy.stats import multivariate_normal

def func_Prob(x, y): ## desity function, ignore constant
    return math.exp(-0.5*((x-1)**2+y**2)) + math.exp(-0.5*((x+1)**2+y**2))

def func_dProb(x, y):
    dx = -(x-1) * math.exp(-0.5*((x-1)**2+y**2)) -(x+1) * math.exp(-0.5*((x+1)**2+y**2))
    dy = -y* math.exp(-0.5*((x-1)**2+y**2)) + math.exp(-0.5*((x+1)**2+y**2))
    return np.array([dx, dy])

def grad_V(X):
    x, y = X[0], X[1]
    # p1 = math.exp(-0.5*((x-1)**2+y**2))
    # p2 = math.exp(-0.5*((x+1)**2+y**2))
    dx = x - np.tanh(x)
    dy = y
    return np.array([dx, dy])

def simulate_one_path(X0, eps, dt:0.1, max_time:100):
    X = np.array(X0, dtype=float)
    t = 0.0
    prev_X = X[0]
    assert prev_X >= 0, "x分量初值小于0!"
    prev_t = t

    while t < max_time:
        drift = -grad_V(X)
        dW = np.sqrt(2 * eps * dt) * np.random.normal(size=2)
        X_new = X + drift * dt + dW
        t_new = t + dt

        # 检查是否穿过零
        if prev_X > 0 and X_new[0] <= 0:
            # 线性插值估计穿零时间
            if prev_X - X_new[0] >= 1e-6 :
                tau = prev_t + (0 - prev_X) / (X_new[0] - prev_X) * dt
            else:
                tau = t_new
            return tau

        prev_X = X_new[0]
        prev_t = t_new
        X = X_new
        t = t_new

    return None  # 未在 max_time 内穿过

def estimate_mean_stopping_time(X0, eps, dt=0.001, max_time=100.0, n_sim=1000):
    taus = []
    for i in range(n_sim):
        tau = simulate_one_path(X0, eps, dt, max_time)
        if tau is not None:
            taus.append(tau)
        else:
            pass
        # if (i+1)%(n_sim/10) == 0:
        #     print(f"已模拟{i+1}/{n_sim}次, 成功{len(taus)}次.")
    taus = np.array(taus)
    if len(taus) == 0:
        return np.nan, np.nan, 0
    mean_tau = np.mean(taus)
    std_tau = np.std(taus, ddof=1)
    success_rate = len(taus) / n_sim
    return mean_tau, std_tau, success_rate

if __name__ == "__main__":

    eps = 0.1
    dt = 0.001
    x0 = [1.0, 0.0]
    max_time_array = [10.0, 20.0, 40.0, 80.0]  
    n_sim = 500
    
    os.makedirs("data", exist_ok=True)
    with open("data/init_test.txt", "w") as file:
        file.write("Max_Time T_Value T_std Success_Rate\n")
        for max_time in max_time_array:
            print(f"Running with max time = {max_time}")
            T_mean, T_std, success_rate = estimate_mean_stopping_time(x0, eps, dt=dt, n_sim=n_sim, max_time=max_time)
            file.write(f"{max_time:.3f} & {T_mean:.3f} & {T_std:.3f} & {success_rate*100:.2f}%\n")
            
