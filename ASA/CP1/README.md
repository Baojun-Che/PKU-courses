Codes for **Computer Project 1** of course 'Applied Stochastic Analysis' in Peking University. For details see:

https://www.math.pku.edu.cn/teachers/litj/notes/appl_stoch/ComputerProjects2024.pdf


## 文件说明

### Potts_Model_2D.py
核心Potts模型实现文件，包含：
- `PottsModel2D`类：二维Potts模型的主要实现
- `wolff_flip`方法：Wolff簇翻转算法
- `metropolis_flip`方法：Metropolis算法
- `mcmc_without_external_field`函数：无外场下的蒙特卡洛模拟
- `mcmc_with_external_field`函数：有外场下的蒙特卡洛模拟


### critical_temperature.py
用于估计分析临界温度工具:
- 计算生成不同参数条件下的内能和比热
- 作图察比热容峰值确定临界温度
- 逐渐精细化温度步长以确定临界温度

### prime_test.py
基础测试工具：
- 用于生成不同参数下的初始晶格配置
- 保存晶格状态供后续分析使用

### regeression_gamma.py
用于估计比热临界指数γ的分析工具：
- 通过分析比热在临界温度附近的行为来估计临界指数
- 使用线性回归拟合 ln|c| ~ ln|1-T/T*| 的关系
- 生成回归分析图和结果数据文件

### corr_length.py
计算和分析关联长度的工具：
- 使用蒙特卡洛模拟计算不同温度下的关联函数
- 通过线性回归拟合 ln(关联函数) ~ 距离 的关系来提取关联长度
- 支持状态继承以提高计算效率

### regeression_delta.py
用于估计关联长度临界指数δ的分析工具：
- 通过分析关联长度在临界温度附近的行为来估计临界指数
- 过滤R²值较低的数据点以提高拟合质量
- 生成关联长度随温度变化的图表

### manetization.py
计算磁化率的工具：
- 在不同温度和外场强度下计算系统的磁化率
- 生成磁化率随外场变化的二维数据表
- 绘制磁化率-外场曲线图

### ./results
物理量、参数数据及作图

### ./lattice
一些条件下稳定态的lattice状态


## 运行MCMC模拟示例

results, energies, lattice = mcmc_without_external_field(
N=100,      # 系统大小
q=3,        # Potts模型状态数
T=0.995,    # 温度
n_tempering = 500,  # 热化步数
n_measure = 2000,   # 测量次数
n_step = 5,         # 测量之间间隔的迭代数
RATE = 3,           # 共迭代 RATE × n_step × n_measure步
get_energy = True   # 返回结束时的lattice以及测量的energies
)