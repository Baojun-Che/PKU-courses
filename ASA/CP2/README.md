Codes for **Computer Project 2** of course 'Applied Stochastic Analysis' in Peking University. For details see:

https://www.math.pku.edu.cn/teachers/litj/notes/appl_stoch/ComputerProjects2024.pdf


# 随机微分方程首次通过时间的数值研究

## 问题描述

本项目旨在数值研究二维随机微分方程的首次通过时间问题。具体研究以下随机微分方程：

$$
dX_t = -\nabla V(X_t)dt + \sqrt{2\varepsilon}dW_t
$$

其中势函数 $V(x,y) = -\ln p(x,y)$，$p(x,y) = \frac{1}{2}[\mathcal{N}((x,y); (+1,0), I_2) + \mathcal{N}((x,y); (-1,0), I_2)]$，首次通过时间定义为 $\tau_b = \inf\{t \geq 0 : X_t = 0\}$。

## 文件结构

### 核心模块

- **`SDE_solver.py`**: 核心求解器模块
  - 定义势函数 $V(x,y)$ 及其梯度 $\nabla V$
  - 实现 Euler-Maruyama 数值离散化方法
  - 提供单次路径模拟函数 `simulate_one_path`
  - 提供平均首次通过时间估计函数 `estimate_mean_stopping_time`

- **`tau_eps.py`**: 研究问题(b) - $\varepsilon$ 对平均首次通过时间的影响
  - 固定初始位置 $x_0 = (1,0)$
  - 变化 $\varepsilon$ 从 $10^{-2}$ 到 $10^2$
  - 生成 $\varepsilon$ vs $\tau$ 的关系图

- **`tau_x0.py`**: 研究问题(c) - 初始位置 $x_0$ 对平均首次通过时间的影响
  - 固定多个 $\varepsilon$ 值
  - 变化初始位置 $x_0$ 从 0.05 到 2.0
  - 生成 $x_0$ vs $\tau$ 的关系图

- **`appendix_test.py`**: 附录测试代码
  - 对不同初始位置（如 $x_0 = (0.5,0)$, $(2,0)$）进行额外测试

### 数据与图表输出

- **`data/`**: 存储所有数值实验结果的文本文件
- **`figs/`**: 存储所有生成的图像文件


## 核心函数说明

### `estimate_mean_stopping_time` 函数: 估计给定参数下的平均首次通过时间

``` python
mean_tau, std_tau, success_rate = estimate_mean_stopping_time(X0, eps, dt=0.001, max_time=100.0, n_sim=1000):
```

**参数**:
- `X0`: 初始位置，numpy数组，形状为(2,)，例如 [1.0, 0.0]
- `eps`: 噪声强度参数 $\varepsilon > 0$
- `dt`: 时间步长，默认为 0.001
- `max_time`: 最大模拟时间，默认为 100.0
- `n_sim`: 模拟次数，默认为 1000

**返回值**:
- `mean_tau`: 平均首次通过时间的估计值
- `std_tau`: 首次通过时间的标准差
- `success_rate`: 模拟成功率（成功到达边界的模拟次数 / 总模拟次数）

**使用示例**:
```python
from SDE_solver import estimate_mean_stopping_time

x0 = [1.0, 0.0]
eps = 0.1
mean_tau, std_tau, success_rate = estimate_mean_stopping_time(x0, eps, dt=0.001, n_sim=500)
print(f"平均停时: {mean_tau:.3f}, 标准差: {std_tau:.3f}, 成功率: {success_rate:.2f}")
```





