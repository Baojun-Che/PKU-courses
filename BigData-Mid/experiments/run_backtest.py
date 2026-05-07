import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.backtest import backtest_strategy

data_returns = "data/cleaned/cleaned_returns.csv"
estimates = "data/estimates_train.npz"
save_path = "results/backtest"
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, "figures"), exist_ok=True)
os.makedirs(os.path.join(save_path, "tables"), exist_ok=True)

# 使用 pandas 读取，避免 UTF-8 BOM 问题
returns_df = pd.read_csv(data_returns, index_col=0)
returns = returns_df.values
available_tickers = returns_df.columns.tolist()
n_assets = returns.shape[1]

est_data = np.load(estimates)
mu = est_data["mu"]
Sigma = est_data["Sigma"]

print(f"返回数据中有 {n_assets} 只股票")
print(f"可用股票: {available_tickers}")

# 简单示例：equal-weight, QP weights, SOCP weights
equal_weights = np.ones(n_assets)/n_assets

# 加载商业求解器结果（使用 pandas 读取 CSV 文件）
# 权重文件中的股票可能与返回数据不匹配，需要对齐
qp_df = pd.read_csv("results/commercial_baselines/tables/qp_weights_by_gamma.csv")
# 获取与返回数据匹配的股票列
common_tickers_qp = [t for t in available_tickers if t in qp_df.columns]
print(f"QP权重文件中匹配的股票数量: {len(common_tickers_qp)}")

if len(common_tickers_qp) > 0:
    qp_weights = qp_df[common_tickers_qp].iloc[-1].values
    # 归一化权重
    qp_weights = qp_weights / qp_weights.sum()
else:
    print("警告：QP权重文件中没有匹配的股票，使用等权重")
    qp_weights = equal_weights.copy()

socp_df = pd.read_csv("results/commercial_baselines/tables/socp_weights_by_sigma.csv")
common_tickers_socp = [t for t in available_tickers if t in socp_df.columns]
print(f"SOCP权重文件中匹配的股票数量: {len(common_tickers_socp)}")

if len(common_tickers_socp) > 0:
    socp_weights = socp_df[common_tickers_socp].iloc[-1].values
    socp_weights = socp_weights / socp_weights.sum()
else:
    print("警告：SOCP权重文件中没有匹配的股票，使用等权重")
    socp_weights = equal_weights.copy()

strategies = {
    "equal_weight": [equal_weights]*5,  # 假设rebalance 5次
    "qp": [qp_weights]*5,
    "socp": [socp_weights]*5
}

metrics_all = {}
for name, w_hist in strategies.items():
    metrics = backtest_strategy(returns, w_hist)
    metrics_all[name] = metrics
    # 绘制累计财富曲线
    plt.plot(metrics["cumulative_wealth"], label=name)

plt.xlabel("Trading Day") 
plt.ylabel("Total Wealth")
plt.title("Strategy Backtest Cumulative Wealth")
plt.legend()
plt.savefig(os.path.join(save_path, "figures/cumulative_wealth.png"))
plt.close()

# 保存指标表
import pandas as pd
metrics_df = pd.DataFrame.from_dict(metrics_all, orient='index')
metrics_df.to_csv(os.path.join(save_path, "tables/backtest_metrics.csv"))