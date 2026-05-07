import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.backtest import backtest_strategy

data_returns = "data/cleaned/cleaned_returns.csv"
estimates = "data/estimates_train.npz"
save_path = "results/sensitivity"
sens_path = save_path
fig_path = os.path.join(sens_path, "figures")
table_path = os.path.join(sens_path, "tables")
os.makedirs(fig_path, exist_ok=True)
os.makedirs(table_path, exist_ok=True)

returns_df = pd.read_csv(data_returns, index_col=0)
returns = returns_df.values
available_tickers = returns_df.columns.tolist()

est_data = np.load(estimates)
mu = est_data["mu"]
Sigma = est_data["Sigma"]
n_assets = returns.shape[1]

qp_weights_df = pd.read_csv("results/commercial_baselines/tables/qp_weights_by_gamma.csv")
socp_weights_df = pd.read_csv("results/commercial_baselines/tables/socp_weights_by_sigma.csv")

common_tickers_qp = [t for t in available_tickers if t in qp_weights_df.columns]
common_tickers_socp = [t for t in available_tickers if t in socp_weights_df.columns]

print(f"返回数据中有 {n_assets} 只股票")
print(f"QP权重文件中匹配的股票数量: {len(common_tickers_qp)}")
print(f"SOCP权重文件中匹配的股票数量: {len(common_tickers_socp)}")

gamma_list = [0.5, 1.0, 2.0, 5.0]

def generate_weights_for_gamma(gamma):
    gamma_values = qp_weights_df['gamma'].values
    idx = np.argmin(np.abs(gamma_values - gamma))
    closest_gamma = gamma_values[idx]
    print(f"  请求 gamma={gamma:.2f}, 使用最接近的 gamma={closest_gamma:.4f}")
    
    w = qp_weights_df[common_tickers_qp].iloc[idx].values
    w = np.maximum(w, 0)
    w = w / np.sum(w)
    return [w] * 5

metrics_gamma = {}
for gamma in gamma_list:
    weights_history = generate_weights_for_gamma(gamma)
    metrics = backtest_strategy(returns, weights_history)
    metrics_gamma[gamma] = metrics

# 保存累计财富曲线
plt.figure()
for gamma, m in metrics_gamma.items():
    plt.plot(m["cumulative_wealth"], label=f"γ={gamma}")
plt.xlabel("Trading Day") 
plt.ylabel("Total Wealth")
plt.title("Sensitivity Analysis: Cumulative Wealth for Different γ")
plt.legend()
plt.savefig(os.path.join(fig_path, "cumulative_wealth_gamma.png"))
plt.close()

# 保存指标表格
df_gamma = pd.DataFrame.from_dict(metrics_gamma, orient='index')
df_gamma.to_csv(os.path.join(table_path, "metrics_gamma.csv"))

# -------------------------
# 敏感性分析 - σ_max 对 SOCP
# -------------------------
sigma_max_list = [0.01, 0.02, 0.03]

def generate_weights_for_sigma(sigma_max):
    sigma_values = socp_weights_df['sigma_max'].values
    idx = np.argmin(np.abs(sigma_values - sigma_max))
    closest_sigma = sigma_values[idx]
    print(f"  请求 sigma_max={sigma_max:.2f}, 使用最接近的 sigma_max={closest_sigma:.4f}")
    
    w = socp_weights_df[common_tickers_socp].iloc[idx].values
    w = np.maximum(w, 0)
    w = w / np.sum(w)
    return [w] * 5

metrics_sigma = {}
for sigma_max in sigma_max_list:
    weights_history = generate_weights_for_sigma(sigma_max)
    metrics = backtest_strategy(returns, weights_history)
    metrics_sigma[sigma_max] = metrics

# 绘制夏普比率随 σ_max 变化
plt.figure()
sharpe_values = [metrics_sigma[s]["sharpe_ratio"] for s in sigma_max_list]
plt.plot(sigma_max_list, sharpe_values, marker="o")
plt.xlabel("σ_max")
plt.ylabel("Sharpe Ratio")
plt.title("Sensitivity Analysis: Impact of σ_max on Sharpe Ratio")
plt.savefig(os.path.join(fig_path, "sharpe_ratio_sigma.png"))
plt.close()

# 保存指标表格
df_sigma = pd.DataFrame.from_dict(metrics_sigma, orient='index')
df_sigma.to_csv(os.path.join(table_path, "metrics_sigma.csv"))

# -------------------------
# 失败案例记录
# -------------------------
failure_cases = []
for gamma, m in metrics_gamma.items():
    max_weight = np.max(generate_weights_for_gamma(gamma)[-1])
    if m["annualized_return"] < 0 or max_weight > 0.5:
        failure_cases.append({
            "参数类型": "γ",
            "参数值": gamma,
            "年化收益": m["annualized_return"],
            "最大权重": max_weight,
            "平均换手率": m["average_turnover"],
            "备注": "收益为负或权重过于集中"
        })

for sigma, m in metrics_sigma.items():
    max_weight = np.max(generate_weights_for_sigma(sigma)[-1])
    if m["annualized_return"] < 0 or max_weight > 0.5:
        failure_cases.append({
            "参数类型": "σ_max",
            "参数值": sigma,
            "年化收益": m["annualized_return"],
            "最大权重": max_weight,
            "平均换手率": m["average_turnover"],
            "备注": "收益为负或权重过于集中"
        })

df_failure = pd.DataFrame(failure_cases)
df_failure.to_csv(os.path.join(table_path, "failure_cases.csv"), index=False)