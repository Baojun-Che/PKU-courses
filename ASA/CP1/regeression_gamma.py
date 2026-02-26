from Potts_Model_2D import mcmc_without_external_field
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def estimate_constant(T_values, y_values, T_star, arg_name, fig_name):
    
    logT = np.log(np.abs(1 - T_values / T_star))
    logy = np.log(y_values)
    
    # 移除无穷大或NaN值
    valid_indices = np.isfinite(logT) & np.isfinite(logy)
    logT = logT[valid_indices]
    logy = logy[valid_indices]
    T_values = T_values[valid_indices]
    y_values = y_values[valid_indices]
    
    # 线性回归
    slope, intercept, r_value, p_value, _ = stats.linregress(logT, logy)
    r_squared = r_value ** 2
    
    # 绘制图形
    plt.figure(figsize=(10, 4))
    
    # 子图1: y 关于 T 的图像
    plt.subplot(1, 2, 1)
    plt.plot(T_values, y_values, 'b-', linewidth=2)
    plt.axvline(x=T_star, color='r', linestyle='--', alpha=0.7, label=f'T_star = {T_star}')
    plt.xlabel('T')
    plt.ylabel(arg_name)
    
    # 子图2: ln y 关于 ln |1-T/T_star| 的散点图和回归直线
    plt.subplot(1, 2, 2)
    plt.scatter(logT, logy, alpha=0.7)
    plt.xlabel('ln epsilon')
    plt.ylabel('ln '+arg_name)
    x_fit = np.linspace(min(logT), max(logT), 100)
    y_fit = intercept + slope * x_fit
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'y = {intercept:.3f} {slope:+.3f}x')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()
    
    print("回归结果:")
    print(f"gamma 估计值: {-slope:.6f}")
    print(f"回归R²值: {r_squared:.6f}, p值: {p_value}")
    
    return -slope

# 运行分析
if __name__ == "__main__":

    q = 3
    N = 100
    T_star = 0.995
    r = 0.05

    # 在临界区域附近生成T
    T_min = T_star * (1 - r)
    T_max = T_star * (1 + r)
    temperatures = np.linspace(T_min, T_max, 20)
    temperatures = temperatures[temperatures != T_star]

    internal_energies, specific_heats = [], []
    _, _, lattice = mcmc_without_external_field(N, q, T_min, n_tempering=500, n_measure=2000, get_energy=True, mes_energy=False)
    for T in temperatures:
        results, _, lattice = mcmc_without_external_field(N, q, T, lattice = lattice, n_tempering=0, n_measure=2000, RATE=5, get_energy=True)
        specific_heats.append(results['specific_heat'])
        print(f"c={results['specific_heat']}" )
    
    data = np.column_stack((temperatures, specific_heats))
    np.savetxt('results/reg-gamma-q=3.txt', data, header='Temperature Specific_Heats', fmt='%.6f', delimiter=' ')
    
    data = np.loadtxt('results/reg-gamma-q=3.txt', skiprows=1)
    gamma_est = estimate_constant(data[:,0], data[:,1], T_star, "c", "results/reg-gamma-q=3-wrong.pdf")

    mask = data[:,0] < 1.015
    last_index = len(data[:,0]) - np.argmax(mask[::-1])
    gamma_est = estimate_constant(data[0:last_index,0], data[0:last_index,1], T_star, "c", "results/reg-gamma-q=3.pdf")

    print(gamma_est)
    



    T_star = 0.705
    q = 10
    N = 100
    r = 0.05

    # 在临界区域附近生成T
    T_min = T_star * (1 - r)
    T_max = T_star * (1 + r)
    temperatures = np.linspace(T_min, T_max, 20)
    temperatures = temperatures[temperatures != T_star]

    internal_energies, specific_heats = [], []
    _, _, lattice = mcmc_without_external_field(N, q, T_min, n_tempering=500, n_measure=2000, get_energy=True, mes_energy=False)
    for T in temperatures:
        results, _, lattice = mcmc_without_external_field(N, q, T, lattice = lattice, n_tempering=0, n_measure=4000, RATE=5, get_energy=True)
        specific_heats.append(results['specific_heat'])
        print(f"c={results['specific_heat']}" )
    
    data = np.column_stack((temperatures, specific_heats))
    np.savetxt('results/reg-gamma-q=10.txt', data, header='Temperature Specific_Heats', fmt='%.6f', delimiter=' ')
    
    data = np.loadtxt('results/reg-gamma-q=10.txt', skiprows=1)
    gamma_est = estimate_constant(data[:,0], data[:,1], T_star, "c", "results/reg-gamma-q=10-wrong.pdf")
    
    indice = np.where((data[:,1] <= 10) & (data[:,0] < T_star))
    gamma_est = estimate_constant(data[indice,0], data[indice,1], T_star, "c", "results/reg-gamma-q=10.pdf")
