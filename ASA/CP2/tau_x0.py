import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from SDE_solver import estimate_mean_stopping_time


if __name__ == "__main__":

    N_x = 40
    xx = np.linspace(0.05, 2.0, N_x)
    eps_array = np.array([0.25, 0.5, 1.0, 2.0])

    os.makedirs("data", exist_ok=True)
    with open("data/tau-x0.txt", "w") as file:
        file.write("Epsilon\tx_0\tT_Value\n")
        for eps in eps_array:
            print(f"Running for eps = {eps}")
            for x in xx:
                x0 = np.array([x, 0.0])
                T_val, _, success_rate = estimate_mean_stopping_time(x0, eps, dt=0.001, n_sim=500)
                if success_rate < 0.75:
                    warnings.warn(f"Success rate = {success_rate} for x = {x} !")
                else:
                    print(f"Mean stopping time = {T_val} for x = {x}")
                    # 把 x, y, T_val 输出到 data/tau-x0.txt
                    file.write(f"{eps:.2f} {x:.2f} {T_val:.6f}\n")
    
    data = np.loadtxt("data/tau-x0.txt", skiprows=1)

    T = np.zeros((4, N_x))
    for i in range(4):
        T[i, :] = data[i*N_x:(i+1)*N_x, 2]

    plt.rcParams['axes.labelsize'] = 14  # x、y 轴标签的字体大小
    
    colors = ['blue', 'red', 'green', 'orange']
    
    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(xx, T[i, :], 'o-', 
                 color=colors[i], 
                 markersize = 1,
                 label=f"eps={eps_array[i]}")
    
    plt.xlabel(r'$X_0$')
    plt.ylabel(r'$\tau^{\varepsilon}$')
    
    plt.grid(True, alpha=0.3)
    
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("figs/x0-T_plot.pdf", dpi=300, bbox_inches='tight')
    plt.show()



    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(np.log(xx), np.log(T[i, :]), 'o-',
                 color=colors[i], markersize=1,
                 label=f"eps={eps_array[i]}")
        coefficients = np.polyfit(np.log(xx), np.log(T[i, :]), 1)
        polynomial = np.poly1d(coefficients)
        x_fit = np.linspace(min(np.log(xx)), max(np.log(xx)), 100)
        y_fit = polynomial(x_fit)
        plt.plot(x_fit, y_fit, '--', color=colors[i], label=f'Fit Line: $y={coefficients[0]:.2f}x+{coefficients[1]:.2f}$')

        
    plt.xlabel(r'$\lg X_0$')
    plt.ylabel(r'$\lg \tau^{\varepsilon}$')
    
    plt.grid(True, alpha=0.3)
    
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("figs/loglog-x0-T_plot.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    
