from Potts_Model_2D import mcmc_without_external_field
from regeression_gamma import estimate_constant
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def plot_corr_len(tempratures, length, r_square, fig_name, c=0.75):
    # 确保输入是numpy数组
    tempratures = np.array(tempratures).flatten()
    length = np.array(length).flatten()
    r_square = np.array(r_square).flatten()

    indices = np.where(length>0)
    tempratures = tempratures[indices]
    length = length[indices]
    r_square = r_square[indices]

    # 按照tempratures从小到大排序
    sort_indices = np.argsort(tempratures)
    tempratures = tempratures[sort_indices]
    length = length[sort_indices]
    r_square = r_square[sort_indices]   

    print(tempratures)

    # 筛选出r_square > c的数据点
    mask = np.where(r_square > c)
    filtered_tempratures = tempratures[mask]
    filtered_length = length[mask]

    plt.figure(figsize=(10, 6))
    
    plt.plot(tempratures, length, color='red')
    plt.plot(filtered_tempratures, filtered_length, color='blue', linestyle='', marker='o', label='Data (R^2 >={})'.format(c))
    plt.legend()
    
    plt.xlabel('Temperature')
    plt.ylabel('Correlation Length')

    plt.savefig(fig_name)
    plt.show()
    
    return filtered_tempratures, filtered_length


# 运行分析
if __name__ == "__main__":

    data = np.loadtxt('results/q=3,corr_len.txt', skiprows=1)
    filtered_tempratures, filtered_length = plot_corr_len(data[:,0], data[:,1], data[:,2], 'results/q=3,corr_len.pdf')

    T_star = 0.995
    mask = filtered_tempratures > T_star
    estimate_constant(filtered_tempratures[mask], filtered_length[mask], T_star, 'ξ', 'results/reg-delta-q=3.pdf')


    data = np.loadtxt('results/q=10,corr_len.txt', skiprows=1)
    indice = np.where(data[:,1] > 0)
    filtered_tempratures, filtered_length = plot_corr_len(data[indice,0], data[indice,1], data[indice,2], 'results/q=10,corr_len.pdf', c=0.50)

    T_star = 0.705
    mask = filtered_tempratures > T_star
    estimate_constant(filtered_tempratures[mask], filtered_length[mask], T_star, 'ξ', 'results/reg-delta-q=10.pdf')