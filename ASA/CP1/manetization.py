from Potts_Model_2D import mcmc_without_external_field, mcmc_with_external_field
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_data_from_file(filename, fig_name):
    try:
        # 读取数据文件
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # 解析第一行：温度值
        temperatures = np.array([float(x) for x in lines[0].split()])
        N_T = len(temperatures)
        
        # 解析第二行：h指标值
        h_values = np.array([float(x) for x in lines[1].split()])
        N_h = len(h_values)
        
        # 解析矩阵数据
        data_matrix = []
        for i in range(2, len(lines)):
            row = [float(x) for x in lines[i].split()]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # 检查数据维度是否匹配
        if data_matrix.shape != (N_T, N_h):
            print(f"警告: 数据维度不匹配。期望 ({N_T}, {N_h})，实际得到 {data_matrix.shape}")
            # 尝试截断或调整数据
            if data_matrix.shape[0] > N_T:
                data_matrix = data_matrix[:N_T, :]
            if data_matrix.shape[1] > N_h:
                data_matrix = data_matrix[:, :N_h]
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制每条温度曲线
        for i, temp in enumerate(temperatures):
            plt.plot(h_values, data_matrix[i, :], 
                    marker='o', markersize=4, linewidth=2, 
                    label=f'T = {temp}')
        
        # 设置图形属性
        plt.xlabel('h', fontsize=14)
        plt.ylabel('manetization', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 保存图形
        plt.tight_layout()
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        print(f"图形已保存为: {fig_name}")
        
        # 显示图形（可选）
        plt.show()
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filename}")
    except Exception as e:
        print(f"处理文件时出错: {e}")


if __name__ == "__main__":
    N = 100
    q = 3
    tempratures = np.array([0.5, 0.8, 0.995, 1.2, 1.5])
    h_list = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    N_T, N_h = len(tempratures), len(h_list)
    data = np.zeros((N_T, N_h))
    for iT in range(N_T):
        for ih in range(N_h):
            T = tempratures[iT]
            h = h_list[ih]
            temp = mcmc_with_external_field(N,q,T,h,n_tempering=100,n_measure=500,RATE=2)
            print(f"T={T},h={h}, manetization={temp}")
            data[iT,ih] = temp

    with open('results/manetization-q=3.txt', 'w') as f:
        np.savetxt(f, tempratures.reshape(1, -1), fmt='%g')
        np.savetxt(f, h_list.reshape(1, -1), fmt='%g')
        np.savetxt(f, data, fmt='%g')


    N = 100
    q = 10
    tempratures = np.array([0.5, 0.6, 0.705, 0.8, 0.9])
    h_list = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    N_T, N_h = len(tempratures), len(h_list)
    data = np.zeros((N_T, N_h))
    for iT in range(N_T):
        for ih in range(N_h):
            T = tempratures[iT]
            h = h_list[ih] 
            temp = mcmc_with_external_field(N,q,T,h,n_tempering=200,n_measure=1000,RATE=3)
            print(f"T={T},h={h}, manetization={temp}")
            data[iT,ih] = temp

    with open('results/manetization-q=10.txt', 'w') as f:
        np.savetxt(f, tempratures.reshape(1, -1), fmt='%g')
        np.savetxt(f, h_list.reshape(1, -1), fmt='%g')
        np.savetxt(f, data, fmt='%g')

    plot_data_from_file("results/manetization-q=3.txt", "results/manetization-q=3.pdf")
    plot_data_from_file("results/manetization-q=10.txt", "results/manetization-q=10.pdf")