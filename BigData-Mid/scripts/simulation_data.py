import pandas as pd
import numpy as np
from datetime import datetime

# ---------- 参数配置（参考 download_data.py）----------
CSV_PATH = "data/sp500_top80_universe.csv"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
OUTPUT_CSV = "data/stock_prices.csv"

# ---------- 1. 读取股票列表 ----------
df_stocks = pd.read_csv(CSV_PATH)
if "YahooTicker" in df_stocks.columns:
    tickers = df_stocks["YahooTicker"].dropna().unique().tolist()
else:
    tickers = df_stocks["Ticker"].dropna().unique().tolist()

print(f"待处理的股票数量: {len(tickers)}")
print(f"前10只股票: {tickers[:10]}")

# ---------- 2. 生成日期范围 ----------
dates = pd.date_range(start=START_DATE, end=END_DATE, freq='B')  # B = 工作日
print(f"\n日期范围: {dates[0].strftime('%Y-%m-%d')} 至 {dates[-1].strftime('%Y-%m-%d')}")
print(f"总交易日数: {len(dates)}")

# ---------- 3. 生成模拟价格数据 ----------
# 设置随机种子以保证可复现性
np.random.seed(42)

# 创建空的价格DataFrame
prices = pd.DataFrame(index=dates)

# 每只股票的基准参数（基于真实市场数据统计）
stock_params = {
    'AAPL': {'return': 0.18, 'vol': 0.25, 'start_price': 10},
    'MSFT': {'return': 0.20, 'vol': 0.23, 'start_price': 25},
    'GOOGL': {'return': 0.12, 'vol': 0.28, 'start_price': 500},
    'GOOG': {'return': 0.13, 'vol': 0.27, 'start_price': 50},
    'AMZN': {'return': 0.15, 'vol': 0.32, 'start_price': 300},
    'NVDA': {'return': 0.35, 'vol': 0.45, 'start_price': 20},
    'META': {'return': 0.12, 'vol': 0.38, 'start_price': 100},
    'TSLA': {'return': 0.28, 'vol': 0.55, 'start_price': 15},
}

for i, ticker in enumerate(tickers, 1):
    # 获取股票特定参数，没有的话使用默认值
    params = stock_params.get(ticker, {})
    annual_return = params.get('return', np.random.uniform(0.08, 0.22))
    annual_vol = params.get('vol', np.random.uniform(0.18, 0.38))
    start_price = params.get('start_price', np.random.uniform(50, 200))
    
    # 计算日收益率（几何布朗运动）
    daily_returns = np.random.normal(annual_return/252, annual_vol/np.sqrt(252), len(dates))
    
    # 计算价格序列
    price_series = start_price * (1 + daily_returns).cumprod()
    
    # 添加随机噪声模拟市场波动
    price_series *= (1 + np.random.normal(0, 0.002, len(dates)))
    
    # 添加到DataFrame
    prices[ticker] = price_series.round(2)
    
    if i % 10 == 0:
        print(f"已生成 {i}/{len(tickers)} 只股票的数据")

print(f"\n模拟数据生成完成")
print(f"数据形状: {prices.shape}")

# ---------- 4. 添加一些真实市场特征 ----------
# 添加年度趋势（模拟牛市/熊市周期）
for year in range(2015, 2025):
    year_mask = (prices.index.year == year)
    n_days = sum(year_mask)
    
    # 2020年疫情影响
    if year == 2020:
        factor = np.where(
            prices.index[year_mask] < '2020-04-01',
            np.linspace(1, 0.7, n_days),
            np.linspace(0.7, 1.1, n_days)
        )
        prices.loc[year_mask] = prices.loc[year_mask].mul(factor, axis=0)
    # 2022年加息影响
    elif year == 2022:
        factor = np.linspace(1, 0.92, n_days)
        prices.loc[year_mask] = prices.loc[year_mask].mul(factor, axis=0)
    # 2023-2024年复苏
    elif year >= 2023:
        factor = np.linspace(1, 1.15, n_days)
        prices.loc[year_mask] = prices.loc[year_mask].mul(factor, axis=0)

# ---------- 5. 保存数据 ----------
prices.to_csv(OUTPUT_CSV)
print(f"\n价格数据已保存至: {OUTPUT_CSV}")

# 显示统计信息
print("\n数据统计摘要:")
print(f"日期范围: {prices.index.min().strftime('%Y-%m-%d')} 至 {prices.index.max().strftime('%Y-%m-%d')}")
print(f"股票数量: {prices.shape[1]}")
print(f"非空数据比例: {(prices.count().sum() / (prices.shape[0] * prices.shape[1]) * 100):.1f}%")

# 显示前5行
print("\n前5行数据:")
print(prices.head())

# 显示部分股票的统计信息
print("\n部分股票的价格统计:")
sample_tickers = tickers[:5] if len(tickers) >= 5 else tickers
for ticker in sample_tickers:
    print(f"{ticker}: 均值={prices[ticker].mean():.2f}, 标准差={prices[ticker].std():.2f}, "
          f"最小值={prices[ticker].min():.2f}, 最大值={prices[ticker].max():.2f}")
