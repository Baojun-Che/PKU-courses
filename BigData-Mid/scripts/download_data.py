import os
import pandas as pd
import akshare as ak
import time

# 禁用系统代理
os.environ['NO_PROXY'] = '*'
# 或者临时清除 HTTP_PROXY/HTTPS_PROXY
proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
old_env = {k: os.environ.pop(k, None) for k in proxy_vars}

CSV_PATH = "data/sp500_top80_universe.csv"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
OUTPUT_CSV = "data/stock_prices.csv"

start_str = START_DATE.replace("-", "")
end_str = END_DATE.replace("-", "")

df_stocks = pd.read_csv(CSV_PATH)
tickers = df_stocks["YahooTicker"].dropna().unique().tolist()

price_dfs = []

for ticker in tickers:
    print(f"正在下载 {ticker} ...")
    try:
        df = ak.stock_us_hist(
            symbol=ticker,
            period="daily",
            start_date=start_str,
            end_date=end_str,
            adjust="qfq"
        )
        if df.empty:
            print(f"警告：{ticker} 无数据")
            continue
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        price_series = df["收盘"].rename(ticker)
        price_dfs.append(price_series)
        time.sleep(0.5)
    except Exception as e:
        print(f"下载 {ticker} 失败: {e}")

# 恢复代理环境变量（可选）
for k, v in old_env.items():
    if v is not None:
        os.environ[k] = v

if price_dfs:
    wide_prices = pd.concat(price_dfs, axis=1)
    wide_prices.sort_index(inplace=True)
    wide_prices.dropna(how='all', inplace=True)
    wide_prices.to_csv(OUTPUT_CSV)
    print(f"保存成功，形状: {wide_prices.shape}")
else:
    print("无数据")