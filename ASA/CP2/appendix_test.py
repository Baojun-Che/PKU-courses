import numpy as np
import matplotlib.pyplot as plt
import warnings
from SDE_solver import estimate_mean_stopping_time

if __name__ == "__main__":

    ################### Test for x0 = [0.5, 0] ########################

    x0 = np.array([0.5, 0.0])
    epsilons = np.logspace(-2, 2, 50)  # 0.01 到 ~2
    T_values = []
    for eps in epsilons:
        T_val,_,success_rate = estimate_mean_stopping_time(x0, eps, dt=0.001, n_sim=100)
        if success_rate < 0.75:
            warnings.warn(f"Success rate = {success_rate} for eps = {eps} !")
        else:
            print(f"Stopping time = {T_val} for eps = {eps}")
        T_values.append(T_val)

    # 保存数据到 data.txt 文件
    with open('data/eps-T-data-x0=0.5.txt', 'w') as file:
        # 写入表头
        file.write("Epsilon\tT_Value\n")
        for epsilon, t_value in zip(epsilons, T_values):
            # 写入每一行数据
            file.write(f"{epsilon}\t{t_value}\n")

    data = np.loadtxt("data/eps-T-data-x0=0.5.txt", skiprows=1)
    epsilons = data[:,0]
    T_values = data[:,1]

    plt.rcParams['axes.labelsize'] = 14  # x、y 轴标签的字体大小

    plt.figure()
    plt.plot(epsilons, T_values, 'o-', markersize = 1)
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$\tau^{\varepsilon}$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figs/eps-T-x0=0.5.pdf')  # 保存图像

    plt.figure()
    plt.plot(np.log10(epsilons), np.log10(T_values), 'o-', markersize=1)

    # 计算拟合直线
    coefficients = np.polyfit(np.log10(epsilons), np.log10(T_values), 1)
    polynomial = np.poly1d(coefficients)
    x_fit = np.linspace(min(np.log10(epsilons)), max(np.log10(epsilons)), 100)
    y_fit = polynomial(x_fit)

    # 绘制拟合直线
    plt.plot(x_fit, y_fit, 'r--', label=f'Fit Line: $y={coefficients[0]:.2f}x+{coefficients[1]:.2f}$')

    # 添加图例、标签等
    plt.xlabel(r'$\lg \varepsilon$')
    plt.ylabel(r'$\lg \tau^{\varepsilon}$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/loglog-eps-T-x0=0.5.pdf')  # 保存图像
    plt.show()


    ################### Test for x0 = [2, 0] ########################
    x0 = np.array([2, 0.0])
    epsilons = np.logspace(-2, 2, 50)  # 0.01 到 ~2
    T_values = []
    for eps in epsilons:
        T_val,_,success_rate = estimate_mean_stopping_time(x0, eps, dt=0.001, n_sim=100)
        if success_rate < 0.75:
            warnings.warn(f"Success rate = {success_rate} for eps = {eps} !")
        else:
            print(f"Stopping time = {T_val} for eps = {eps}")
        T_values.append(T_val)

    # 保存数据到 data.txt 文件
    with open('data/eps-T-data-x0=2.txt', 'w') as file:
        # 写入表头
        file.write("Epsilon\tT_Value\n")
        for epsilon, t_value in zip(epsilons, T_values):
            # 写入每一行数据
            file.write(f"{epsilon}\t{t_value}\n")

    data = np.loadtxt("data/eps-T-data-x0=2.txt", skiprows=1)
    epsilons = data[:,0]
    T_values = data[:,1]

    plt.rcParams['axes.labelsize'] = 14  # x、y 轴标签的字体大小

    plt.figure()
    plt.plot(epsilons, T_values, 'o-', markersize = 1)
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$\tau^{\varepsilon}$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figs/eps-T-x0=2.pdf')  # 保存图像

    plt.figure()
    plt.plot(np.log10(epsilons), np.log10(T_values), 'o-', markersize=1)

    # 计算拟合直线
    coefficients = np.polyfit(np.log10(epsilons), np.log10(T_values), 1)
    polynomial = np.poly1d(coefficients)
    x_fit = np.linspace(min(np.log10(epsilons)), max(np.log10(epsilons)), 100)
    y_fit = polynomial(x_fit)

    # 绘制拟合直线
    plt.plot(x_fit, y_fit, 'r--', label=f'Fit Line: $y={coefficients[0]:.2f}x+{coefficients[1]:.2f}$')

    # 添加图例、标签等
    plt.xlabel(r'$\lg \varepsilon$')
    plt.ylabel(r'$\lg \tau^{\varepsilon}$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/loglog-eps-T-x0=2.pdf')  # 保存图像
    plt.show()

    

    