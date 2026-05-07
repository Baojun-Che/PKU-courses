from matplotlib import ticker
import pandas as pd
import numpy as np
from datetime import datetime

# 加载最终数据
returns_final = pd.read_csv('data/cleaned_returns.csv', index_col=0, parse_dates=True)

# 划分第一个训练窗口（例如：前252个交易日）
train = returns_final.iloc[:252]

# 估计均值和协方差
mu = train.mean().values
Sigma = train.cov().values

# 数值稳定化
epsilon = 1e-4
Sigma_stable = Sigma + epsilon * np.eye(len(mu))

# 保存参数估计结果到numpy数组
np.save('data/mean.npy', mu)
np.save('data/cov.npy', Sigma_stable)

print(f"均值向量维度: {mu.shape}")
print(f"协方差矩阵维度: {Sigma.shape}")
print(f"协方差矩阵是否正定: {np.all(np.linalg.eigvals(Sigma_stable) > 0)}")

np.savez(
    "results/estimates_train.npz",
    mu=mu,                 # shape: (n,)
    Sigma=Sigma_stable,           # shape: (n, n)
    tickers=np.array(train.columns)  # optional, shape: (n,)
)