我们来为如下二维热传导方程设计差分格式：

$$
\begin{cases}
u_t = \Delta u + p(t)\delta(x_0, y_0), & (x,y) \in \Omega, \, t > 0, \\
u(x,y,0) = u^0(x,y), & (x,y) \in \bar{\Omega}, \\
u(x,y,t) = g(x,y,t), & (x,y) \in \partial\Omega, \, t > 0,
\end{cases}
$$

其中：
- $\Delta u = u_{xx} + u_{yy}$ 是拉普拉斯算子；
- $\delta(x_0, y_0)$ 是二维狄拉克函数，表示在点 $(x_0, y_0)$ 处的点源；
- $p(t)$ 是时间依赖的源强度；
- $\Omega$ 是二维区域（如矩形区域），$\partial\Omega$ 是其边界。

---

### **1. 网格划分**

设空间区域 $\Omega = [a,b] \times [c,d]$，时间区间 $[0,T]$。

令：
- 空间步长：$h_x = \frac{b-a}{N_x}, h_y = \frac{d-c}{N_y}$；
- 时间步长：$\tau$；
- 离散点：$x_i = a + i h_x$, $y_j = c + j h_y$, $t_n = n\tau$；
- $u_{i,j}^n \approx u(x_i, y_j, t_n)$。

---

## **一、显式差分格式（Explicit Scheme）**

使用**向前差分**处理时间导数，**中心差分**处理空间二阶导数。

### **离散形式：**

$$
\frac{u_{i,j}^{n+1} - u_{i,j}^n}{\tau} = \frac{u_{i+1,j}^n - 2u_{i,j}^n + u_{i-1,j}^n}{h_x^2} + \frac{u_{i,j+1}^n - 2u_{i,j}^n + u_{i,j-1}^n}{h_y^2} + p(t_n)\delta_{i,i_0}\delta_{j,j_0}
$$

其中 $(x_0, y_0) = (x_{i_0}, y_{j_0})$，$\delta_{i,i_0}$ 是克罗内克函数，当 $i=i_0$ 时为 1，否则为 0。

### **整理得：**

$$
u_{i,j}^{n+1} = u_{i,j}^n + \tau \left[ \frac{u_{i+1,j}^n - 2u_{i,j}^n + u_{i-1,j}^n}{h_x^2} + \frac{u_{i,j+1}^n - 2u_{i,j}^n + u_{i,j-1}^n}{h_y^2} \right] + \tau p(t_n)\delta_{i,i_0}\delta_{j,j_0}
$$

### **特点：**
- 显式，计算简单；
- 稳定性条件：需满足 CFL 条件，即
  $$
  \tau \left( \frac{1}{h_x^2} + \frac{1}{h_y^2} \right) \leq \frac{1}{2}
  $$
  否则数值不稳定。

---

## **二、隐式差分格式（Implicit Scheme）**

使用**向后差分**处理时间导数，空间仍用中心差分。

### **离散形式：**

$$
\frac{u_{i,j}^{n+1} - u_{i,j}^n}{\tau} = \frac{u_{i+1,j}^{n+1} - 2u_{i,j}^{n+1} + u_{i-1,j}^{n+1}}{h_x^2} + \frac{u_{i,j+1}^{n+1} - 2u_{i,j}^{n+1} + u_{i,j-1}^{n+1}}{h_y^2} + p(t_{n+1})\delta_{i,i_0}\delta_{j,j_0}
$$

### **整理得：**

将所有 $u^{n+1}$ 移到左边：

$$
u_{i,j}^{n+1} - \tau \left( \frac{u_{i+1,j}^{n+1} - 2u_{i,j}^{n+1} + u_{i-1,j}^{n+1}}{h_x^2} + \frac{u_{i,j+1}^{n+1} - 2u_{i,j}^{n+1} + u_{i,j-1}^{n+1}}{h_y^2} \right) = u_{i,j}^n + \tau p(t_{n+1})\delta_{i,i_0}\delta_{j,j_0}
$$

### **特点：**
- 隐式，无严格稳定性限制（对任意 $\tau$ 都稳定）；
- 每一步需解一个线性方程组（通常是大型稀疏系统）；
- 更适合长时间模拟。

---

## **三、Crank-Nicolson 格式（推荐）**

结合显式和隐式的优点，使用**平均值**。

### **离散形式：**

$$
\frac{u_{i,j}^{n+1} - u_{i,j}^n}{\tau} = \frac{1}{2} \left[ \frac{u_{i+1,j}^{n+1} - 2u_{i,j}^{n+1} + u_{i-1,j}^{n+1}}{h_x^2} + \frac{u_{i,j+1}^{n+1} - 2u_{i,j}^{n+1} + u_{i,j-1}^{n+1}}{h_y^2} \right] + \frac{1}{2} \left[ \frac{u_{i+1,j}^{n} - 2u_{i,j}^{n} + u_{i-1,j}^{n}}{h_x^2} + \frac{u_{i,j+1}^{n} - 2u_{i,j}^{n} + u_{i,j-1}^{n}}{h_y^2} \right] + \frac{1}{2} \left[ p(t_{n+1}) + p(t_n) \right]\delta_{i,i_0}\delta_{j,j_0}
$$

### **特点：**
- 二阶时间精度；
- 无条件稳定；
- 需解线性系统，但比纯隐式更精确。

---

## **四、关于点源项 $\delta(x_0, y_0)$ 的处理**

由于 $\delta(x_0, y_0)$ 是奇异源，在差分中不能直接代入。常用方法是：

- 在网格点 $(i_0, j_0)$ 上，将源项 $p(t)\delta(x_0, y_0)$ 替换为：
  $$
  \text{在 } (i_0, j_0) \text{ 处加入 } p(t) \cdot \frac{1}{h_x h_y}
  $$
  即：将点源“分布”到该网格单元上，相当于在该点增加一个源强为 $p(t)/h_x h_y$ 的项。

  所以在差分格式中，应写成：
  $$
  \text{源项} = p(t_n) \cdot \frac{1}{h_x h_y} \quad \text{当 } (i,j)=(i_0,j_0)
  $$

  注意：若 $h_x = h_y = h$，则为 $p(t)/h^2$。

---

## **五、边界条件处理**

- 对于 Dirichlet 边界条件 $u(x,y,t)=g(x,y,t)$：
  - 在边界点 $(i,j)$ 上，直接令 $u_{i,j}^n = g(x_i, y_j, t_n)$；
  - 在差分公式中，边界点不参与内部迭代，直接赋值。

---

## **总结**

| 格式 | 时间精度 | 空间精度 | 稳定性 | 是否需要求解方程组 |
|------|----------|----------|--------|------------------|
| 显式 | 一阶 | 二阶 | 有条件（CFL） | 否 |
| 隐式 | 一阶 | 二阶 | 无条件 | 是 |
| Crank-Nicolson | 二阶 | 二阶 | 无条件 | 是 |

> 建议：若追求高精度和稳定性，使用 **Crank-Nicolson 格式**；若追求简单快速且时间步小，可用显式格式。

---

如果你有具体的 $\Omega$（比如矩形）、边界条件或初始条件，可以进一步优化格式设计。


我们来求解如下偏微分方程（PDE）问题：

$$
\begin{cases}
u_t = \Delta u + \rho(t)\delta(x_0, y_0), & (x,y) \in \Omega, t > 0, \\
u(x,y,0) = u^0(x,y), & (x,y) \in \Omega, \\
u(x,y,t) = g(x,y,t), & (x,y) \in \partial\Omega, t > 0,
\end{cases}
$$

其中：
- $\rho(t) = \sin t$,
- $u^0(x,y) = u_0 = 0$,
- $(x_0, y_0) = (1, 0)$,
- 区域 $\Omega = \mathbb{R}^2$（全空间），
- 因此边界条件 $g(x,y,t)$ 不适用（因为无边界），即问题为**全空间上的初值问题**。

---

### 第一步：理解问题

由于 $\Omega = \mathbb{R}^2$，没有边界，所以边界条件自动忽略。问题简化为：

$$
\begin{cases}
u_t = \Delta u + \sin t \cdot \delta(x - 1, y - 0), & (x,y) \in \mathbb{R}^2, t > 0, \\
u(x,y,0) = 0, & (x,y) \in \mathbb{R}^2.
\end{cases}
$$

这是一个**非齐次热方程**，源项是时间依赖的狄拉克函数，集中在点 $(1, 0)$。

---

### 第二步：使用格林函数法（基本解）

在全空间 $\mathbb{R}^2$ 上，热方程的基本解（Green 函数）为：

$$
G(x,y,t; x_0,y_0,\tau) = \frac{1}{4\pi (t - \tau)} \exp\left( -\frac{(x - x_0)^2 + (y - y_0)^2}{4(t - \tau)} \right), \quad t > \tau.
$$

但这里我们有一个**点源**，其强度随时间变化：$\rho(t)\delta(x - 1, y - 0) = \sin t \cdot \delta(x - 1, y)$。

我们可以利用**卷积方法**或**Duhamel 原理**来求解。

---

### 第三步：应用 Duhamel 原理

对于初值为零、有源项的热方程：

$$
u_t = \Delta u + f(x,y,t), \quad u(x,y,0) = 0,
$$

解为：

$$
u(x,y,t) = \int_0^t \int_{\mathbb{R}^2} G(x,y,t; x',y',\tau) f(x',y',\tau) \, dx' dy' d\tau,
$$

其中 $G$ 是热核。

在本题中，$f(x,y,t) = \sin t \cdot \delta(x - 1, y)$，因此：

$$
u(x,y,t) = \int_0^t \int_{\mathbb{R}^2} G(x,y,t; x',y',\tau) \cdot \sin \tau \cdot \delta(x' - 1, y') \, dx' dy' d\tau.
$$

利用 $\delta$ 函数的性质，积分变为：

$$
u(x,y,t) = \int_0^t \sin \tau \cdot G(x,y,t; 1, 0, \tau) \, d\tau.
$$

代入热核表达式：

$$
G(x,y,t; 1,0,\tau) = \frac{1}{4\pi (t - \tau)} \exp\left( -\frac{(x - 1)^2 + y^2}{4(t - \tau)} \right).
$$

因此，

$$
\boxed{
u(x,y,t) = \int_0^t \frac{\sin \tau}{4\pi (t - \tau)} \exp\left( -\frac{(x - 1)^2 + y^2}{4(t - \tau)} \right) d\tau
}
$$

这就是所求的解。

---

### 第四步：是否可以进一步简化？

这个积分一般无法用初等函数表示，但可以写成：

令 $r^2 = (x - 1)^2 + y^2$，则

$$
u(x,y,t) = \frac{1}{4\pi} \int_0^t \frac{\sin \tau}{t - \tau} \exp\left( -\frac{r^2}{4(t - \tau)} \right) d\tau
$$

令 $\sigma = t - \tau$，则当 $\tau = 0$, $\sigma = t$；$\tau = t$, $\sigma = 0$，且 $d\tau = -d\sigma$，$\sin \tau = \sin(t - \sigma)$

于是：

$$
u(x,y,t) = \frac{1}{4\pi} \int_0^t \frac{\sin(t - \sigma)}{\sigma} \exp\left( -\frac{r^2}{4\sigma} \right) d\sigma
$$

即：

$$
\boxed{
u(x,y,t) = \frac{1}{4\pi} \int_0^t \frac{\sin(t - \sigma)}{\sigma} \exp\left( -\frac{(x - 1)^2 + y^2}{4\sigma} \right) d\sigma
}
$$

这是另一种形式，有时更便于分析。

---

### 总结

给定条件下的解为：

$$
\boxed{
u(x,y,t) = \frac{1}{4\pi} \int_0^t \frac{\sin(t - \sigma)}{\sigma} \exp\left( -\frac{(x - 1)^2 + y^2}{4\sigma} \right) d\sigma
}
$$

该解满足：
- 热方程 $u_t = \Delta u + \sin t \cdot \delta(x - 1, y)$,
- 初始条件 $u(x,y,0) = 0$,
- 定义在整个 $\mathbb{R}^2 \times (0,\infty)$ 上。

---

### 补充说明

- 这个解描述的是一个在点 $(1,0)$ 处以 $\sin t$ 的强度持续“注入”的热源，初始温度为零时的响应。
- 解的形式是**时间平移卷积**的结果，体现了热扩散的弥散性和记忆性。
- 当 $t \to 0^+$，解趋于 0，符合初值条件。
- 在远处 $(x,y)$ 远离 $(1,0)$，指数衰减快，解较小。

如需数值计算，可对上述积分进行数值积分处理。