import os
import re
import requests
import pandas as pd
from datetime import date

########### 一次性获取 S&P 500 市值表 ###########

url = "https://stockanalysis.com/list/sp-500-stocks/"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
}

response = requests.get(url, headers=headers, timeout=30)
response.raise_for_status()

tables = pd.read_html(response.text)
sp500 = tables[0]

print(sp500.head())
print(sp500.columns)

########### 解析 Market Cap ###########

def parse_market_cap(x):
    """
    把 '4.77T', '881.82B', '10.25B' 这种字符串转成数字。
    单位：美元。
    """
    if pd.isna(x):
        return None

    s = str(x).strip().replace(",", "").replace("$", "")

    if s in {"", "-", "nan", "None"}:
        return None

    multiplier = 1

    if s[-1] == "T":
        multiplier = 1_000_000_000_000
        s = s[:-1]
    elif s[-1] == "B":
        multiplier = 1_000_000_000
        s = s[:-1]
    elif s[-1] == "M":
        multiplier = 1_000_000
        s = s[:-1]
    elif s[-1] == "K":
        multiplier = 1_000
        s = s[:-1]

    try:
        return float(s) * multiplier
    except ValueError:
        return None


sp500["MarketCapNumeric"] = sp500["Market Cap"].apply(parse_market_cap)

# 去掉没有市值的数据
sp500 = sp500.dropna(subset=["MarketCapNumeric"])

# 按市值降序取前 80
top_80 = sp500.sort_values("MarketCapNumeric", ascending=False).head(80).copy()

########### 输出文件 ###########

# 注意：这里是“运行当天”的选择日期
selection_date = date.today().isoformat()

selected_stocks = pd.DataFrame({
    "Ticker": top_80["Symbol"],
    # yfinance 里 BRK.B / BF.B 通常要写成 BRK-B / BF-B
    "YahooTicker": top_80["Symbol"].astype(str).str.replace(".", "-", regex=False),
    "CompanyName": top_80["Company Name"],
    "MarketCap": top_80["MarketCapNumeric"],
    "MarketCapRaw": top_80["Market Cap"],
    "SelectionDate": selection_date,
    "Universe": "S&P500_Top80_by_current_market_cap"
})

os.makedirs("data", exist_ok=True)

output_path = "data/sp500_top80_universe.csv"
selected_stocks.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"成功选出 {len(selected_stocks)} 只股票")
print(f"已保存至 {output_path}")
print(selected_stocks.head(10))