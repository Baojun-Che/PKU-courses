# 投资组合优化项目 

本项目是大数据分析课程的期中作业，完整实现了基于 S&P 500 股票的投资组合优化，包括数据获取、清洗、统计估计、商用求解器求解、一阶算法实现（ADMM 和 PDHG）、回测及敏感性分析。


## 目录结构

```

BigData-Mid/
│
├── data/                  # 原始及清洗后的数据
│   ├── raw_prices/        # 下载的原始日线价格及日志
│   ├── sp500_top80_universe.csv
│   └── cleaned/           # 清洗后的收益率、AdjClose矩阵及异常值报告
│
├── src/                   # 核心算法脚本
│   ├── admm.py
│   ├── pdhg.py
│   ├── backtest.py
│   ├── commercial_solver_baselines.py
│
├── results/               # 运行生成的结果文件
│   ├── commercial_baselines/
│   ├── manual_solver/
│   ├── backtest/
│   └── sensitivity/
│
├── experiments/            # 测试脚本
│   ├── run_manual_solver.py
│   ├── commercial_solver_baselines.py
│   ├── run_backtest.py
│   └── run_sensitivity_analysis.py
│
├── README.md
└── environment.yml        # 可选，记录依赖环境

````


## Python 依赖

建议使用 Python 3.10 或以上版本。主要依赖包如下（版本请根据实际环境修改）：

- numpy==1.26.4
- pandas==2.0.3
- matplotlib==3.7.2
- cvxpy==1.7.5
- scs==3.2.9
- mosek==11.0.30
- requests==2.31.0

## 数据准备

1. **股票池选择**：

   * 使用 `scripts/stock_selection.py` 获取 S&P 500 市值前 80 股票，生成 `data/sp500_top80_universe.csv`。
2. **下载价格数据**：

   * 运行 `scripts/download_data.py` 下载 Yahoo Finance 历史日线数据，输出至 `data/raw_prices/`。
   * 下载字段包括 Date, Open, High, Low, Close, AdjClose（调整收盘价）和 Volume。
3. **数据清洗**：

   * 运行 `scripts/clean_data.py` 进行交易日对齐、缺失值处理、异常值过滤和收益率计算。
   * 输出文件位于 `data/cleaned/`：

     * `cleaned_returns.csv`：收益率矩阵
     * `cleaned_adjclose_prices.csv`：调整收盘价矩阵
     * 异常值报告和清洗摘要 JSON

4. **参数估计**：
    * 运行 `scripts/estimate_parameters.py` 估计收益率矩阵的参数，输出至 `data/estimates_train.npz`。

## 运行命令

1. **商用求解器基线（QP 与 SOCP）**：

```bash
python experiment/run_commercial_solver.py --input data/estimates_train.npz --outdir results/commercial_baselines/ --solver MOSEK
```

2. **ADMM 与 PDHG 手写求解器**：

```bash
python experiment/run_admm_pdhg.py
```

3. **回测**：

```bash
python experiments/run_backtest.py
```

4. **敏感性分析**：

```bash
python experiments/run_sensitivity_analysis.py
```

> 建议运行顺序：数据清洗 → 参数估计 → 商用求解器 → 手写求解器 → 回测 → 敏感性分析。

## 输出结果

* `results/commercial_baselines/`：商业求解器生成的风险-收益前沿、权重矩阵和代表性组合。
* `results/manual_solver/`：ADMM 和 PDHG 结果，包括不同参数下的结果和收敛曲线。
* `results/backtest/`：回测指标表格及累计财富曲线。
* `results/sensitivity/`：敏感性分析指标及图形。
* 图表均为 PNG 格式，可直接用于报告。


## 回测与敏感性分析说明

1. **策略**：

   * 等权重基准（Equal-Weight）
   * QP 优化组合（Model A2）
   * SOCP 优化组合（Model B）

2. **指标**：

   * 累计收益、年化收益、年化波动率、Sharpe 比率
   * 最大回撤、平均换手率

3. **敏感性分析**：

   * QP：风险厌恶参数 $\gamma$ 对组合收益及集中度影响
   * SOCP：标准差上限 $\sigma_{\max}$ 对 Sharpe 比率和组合分散度影响
   * 记录失败案例，如年化收益为负或组合最大权重过高


## 贡献声明

本报告及相关代码在撰写过程中，得到 ChatGPT V5.5-pro 和 Deepseek V3.2 专家模式的辅助，帮助完成算法说明、实验分析及文档组织。作者对最终内容和结论负主要责任。
