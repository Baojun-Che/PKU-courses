Group LASSO 优化算法比较项目
==========================

本项目实现了多种求解 Group LASSO 问题的优化算法，并对它们的性能进行了全面比较。

## 问题定义

求解 Group LASSO 问题：
$$\min_{x} \frac{1}{2}\|A x - b\|_F^2 + \mu \|x\|_{1,2}$$


## 文件结构与功能

### 核心求解器文件

- `gl_cvx_mosek.py`: 使用 CVXPY 调用 MOSEK 求解器
- `gl_cvx_gurobi.py`: 使用 CVXPY 调用 Gurobi 求解器  
- `gl_mosek.py`: 直接使用 MOSEK Fusion API 求解 SOCP
- `gl_gurobi.py`: 直接使用 Gurobi Python API 求解 SOCP
- `gl_SGD_primal.py`: 原始问题的次梯度下降法
- `gl_GD_primal.py`: 原始问题的光滑化梯度法
- `gl_ProxGD_primal.py`: 原始问题的近端梯度法 (ProxGD)
- `gl_FProxGD_primal.py`: 原始问题的加速近端梯度法 (FISTA/FProxGD)
- `gl_ALM_dual.py`: 对偶问题的增广拉格朗日法 (ALM)
- `gl_ADMM_dual.py`: 对偶问题的 ADMM 算法
- `gl_ADMM_primal.py`: 原始问题的 ADMM 算法

### 工具与测试文件

- `utils.py`: 包含辅助函数（数据生成、投影算子、误差计算、绘图等）
- `test_solvers.py`: 批量运行所有求解器并生成性能比较结果

## 求解器输入输出格式

所有求解器函数遵循统一的接口：
```python
x_opt, iter_count, f_values = solver_function(x0, A, b, mu, opts)
```
其中
- opts为参数字典, 可以不输入, 或者输入空字典 opts={}
- x_opt是最优解(numpy矩阵), iter_count是迭代次数, f_values是迭代中的函数值序列(列表)
- 调用求解器的算法, iter_count返回-1, f_values返回包含最优解的单元素列表

## 各组件版本


| 组件         | 版本    |
|--------------|---------|
| Python       | 3.11.4  |
| NumPy          | 1.26.4 |
| Matplotlib   | 3.7.2   |
|cvxpy  | 1.7.5   |
|Gurobi  | 13.0.0 |
|Mosek  | 11.0 |


## 运行结果
使用随机种子
```python
np.random.seed(97006855)
```
得到的数据保存在了[datas](datas)文件夹中, 下面是各算法运行的结果:

| Solver          | Fval       | Errfun   | Errfun_Exact | Time(s) | Iter | Sparsity |
|-----------------|------------|----------|--------------|---------|------|----------|
| cvx_mosek       | 0.67057522 | -        | 3.56e-04     | 1.8220  | -    | 0.1035   |
| cvx_gurobi      | 0.67057522 | 7.66e-06 | 3.61e-04     | 2.1429  | -    | 0.1055   |
| mosek_direct    | 0.67057522 | 4.33e-07 | 3.56e-04     | 0.4263  | -    | 0.1025   |
| gurobi_direct   | 0.67057522 | 1.66e-06 | 3.57e-04     | 0.9032  | -    | 0.1025   |
| SGD_primal      | 0.67059517 | 6.93e-04 | 9.66e-04     | 4.7122  | 2357 | 0.2852   |
| GD_primal       | 0.67058665 | 3.30e-04 | 6.53e-04     | 2.7558  | 1381 | 0.5156   |
| ProxGD_primal   | 0.67057599 | 1.26e-04 | 4.65e-04     | 0.1868  | 643  | 0.1357   |
| FProxGD_primal  | 0.67057528 | 3.22e-05 | 3.37e-04     | 0.1202  | 253  | 0.1025   |
| ALM_dual        | 0.67058907 | 3.71e-04 | 6.71e-04     | 0.7970  | 9    | 0.0996   |
| ADMM_dual       | 0.67058733 | 3.33e-04 | 6.44e-04     | 0.0704  | 70   | 0.0996   |
| ADMM_primal     | 0.67060069 | 1.62e-04 | 4.46e-04     | 0.2093  | 569  | 0.6055   |