from Potts_Model_2D import mcmc_without_external_field, mcmc_without_external_field_2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def plot_correlation_length(temperatures, corr_length, fig_name):
    
    if len(temperatures) != len(corr_length):
        raise ValueError("temperatures和corr_length的长度必须相同")
    
    sorted_indices = np.argsort(temperatures)
    
    sorted_temperatures = temperatures[sorted_indices]
    sorted_corr_length = corr_length[sorted_indices]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(sorted_temperatures, sorted_corr_length, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Temperature', fontsize=14)
    plt.ylabel('Correlation Length', fontsize=14)
    plt.tight_layout()

    plt.savefig(fig_name + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    return sorted_temperatures, sorted_corr_length

# 运行分析
if __name__ == "__main__":

    T_star = 0.995
    q = 3
    N = 100
    r = 0.05
    T_min = T_star * (1 - r)
    T_max = T_star * (1 + r)
    temperatures = np.linspace(T_min, T_max, 20)
    temperatures = temperatures[temperatures != T_star]

    corr_k = np.arange(10, 41, 5)
    T_list = []
    corr_length = []
    r_square = []
    A = []
    
    for T in temperatures[10:20]:
        ## 随机初始化
        corr_gamma = np.zeros(len(corr_k))
        results = mcmc_without_external_field(N, q, T, n_tempering=0, n_measure=1000, RATE=4, n_step=2, mes_energy=False, corr_k=corr_k)
        corr_gamma += np.maximum(results["corr_gamma"], 1e-8)
        print(corr_gamma)
        reg = stats.linregress(corr_k, np.log(corr_gamma))
        print(f"回归R²值: {reg.rvalue**2}, p值: {reg.pvalue}. 相干长度 = {-1/reg.slope}")
        T_list.append(T)
        corr_length.append(-1/reg.slope)
        r_square.append(reg.rvalue**2)

    
    lattice_old = np.load('lattice/N=100,q=3,T=0.95.npy')
    for iT in range(10):
        T = temperatures[iT]
        ## 状态继承初始化
        corr_gamma = np.zeros(len(corr_k))
        results,_,lattice = mcmc_without_external_field(N, q, T, n_tempering=0, n_measure=1000, RATE=4, n_step=2, mes_energy=False, corr_k=corr_k, get_energy=True, lattice=lattice_old)
        corr_gamma += np.maximum(results["corr_gamma"], 1e-8)
        lattice_old = lattice
        corr_gamma /= 1
        print(corr_gamma)
        reg = stats.linregress(corr_k, np.log(corr_gamma))
        print(f"回归R²值: {reg.rvalue**2}, p值: {reg.pvalue}. 相干长度 = {-1/reg.slope}")
        T_list.append(T)
        corr_length.append(-1/reg.slope)
        r_square.append(reg.rvalue**2)

    T_list = np.array(T_list)
    r_square = np.array(r_square)
    corr_length = np.array(corr_length)
    # indice =  np.where(r_square > 0.75)[0]

    data = np.column_stack((temperatures, corr_length, r_square))
    np.savetxt('results/q=3,corr_len.txt', data, header='Temperature Correlation_Length Reg_R_Square', fmt='%.6f', delimiter=' ')
    
    # np.savetxt("results/corr_gamma_list.txt", A)



    T_star = 0.705
    q = 10
    N = 50
    r = 0.05
    T_min = T_star * (1 - r)
    T_max = T_star * (1 + r)
    temperatures = np.linspace(T_min, T_max, 20)
    temperatures = temperatures[temperatures != T_star]

    corr_k = np.arange(10, 21, 2)
    T_list = []
    corr_length = []
    r_square = []
    A = []
    
    lattice = np.load('lattice/N=50,q=10,T=0.705.npy')
    for T in temperatures[10:20]:
        ## 随机初始化
        corr_gamma = np.zeros(len(corr_k))
        results,_,lattice = mcmc_without_external_field(N, q, T, n_tempering=0, n_measure=4000, RATE=5, n_step=2, mes_energy=False, corr_k=corr_k, get_energy=True, lattice=lattice)
        corr_gamma += np.maximum(results["corr_gamma"], 1e-8)
        print(corr_gamma)
        reg = stats.linregress(corr_k, np.log(corr_gamma))
        print(f"回归R²值: {reg.rvalue**2}, p值: {reg.pvalue}. 相干长度 = {-1/reg.slope}")
        T_list.append(T)
        corr_length.append(-1/reg.slope)
        r_square.append(reg.rvalue**2)

    
    lattice = np.load('lattice/N=50,q=10,T=0.67.npy')
    for iT in range(10):
        T = temperatures[iT]
        ## 状态继承初始化
        corr_gamma = np.zeros(len(corr_k))
        results,_,lattice = mcmc_without_external_field(N, q, T, n_tempering=0, n_measure=4000, RATE=5, n_step=2, mes_energy=False, corr_k=corr_k, get_energy=True, lattice=lattice)
        corr_gamma += np.maximum(results["corr_gamma"], 1e-8)
        print(corr_gamma)
        reg = stats.linregress(corr_k, np.log(corr_gamma))
        print(f"回归R²值: {reg.rvalue**2}, p值: {reg.pvalue}. 相干长度 = {-1/reg.slope}")
        T_list.append(T)
        corr_length.append(-1/reg.slope)
        r_square.append(reg.rvalue**2)

    T_list = np.array(T_list)
    r_square = np.array(r_square)
    corr_length = np.array(corr_length)

    data = np.column_stack((temperatures, corr_length, r_square))
    np.savetxt('results/q=10,corr_len.txt', data, header='Temperature Correlation_Length Reg_R_Square', fmt='%.6f', delimiter=' ')
    
    # np.savetxt("results/corr_gamma_list_q=10.txt", A)