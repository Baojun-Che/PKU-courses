import pandas as pd
import numpy as np
from datetime import datetime

# 读取数据，第一列是日期索引
prices_clean = pd.read_csv('data/stock_prices.csv', index_col=0)
prices_clean.index = pd.to_datetime(prices_clean.index)  # 确保索引是日期类型
returns = prices_clean.pct_change().dropna()

# 查看收益率的基本统计
print("\n收益率统计摘要:")
print(returns.describe().round(4))

# 保存清洗后的收益率数据
returns.to_csv('data/cleaned_returns.csv')


