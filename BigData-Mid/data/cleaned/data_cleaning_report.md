# 数据清洗报告

## 1. 数据范围与输入文件

本次清洗使用 `download_data.py` 下载得到的日频价格数据，并使用 `AdjClose` 作为价格变量。样本区间为 **2015-01-01 至 2024-12-31**。原始价格文件为：

`data/raw_prices/all_daily_prices.csv`

初始股票池文件为：

`data/sp500_top80_universe.csv`

下载日志文件为：

`data/raw_prices/download_log.csv`

## 2. 清洗前后资产数量

- 初始股票池资产数量：80
- 下载成功资产数量：79
- 进入价格矩阵的资产数量：79
- 清洗后保留资产数量：76
- 因价格缺失、历史太短、结束太早或长时间缺口被删除的资产数量：3

删除原因汇总：

- `history_too_short_or_starts_too_late`：3 只资产
- `long_consecutive_missing_gap_possible_suspension`：3 只资产
- `too_many_missing_prices`：3 只资产


被删除资产明细：

| Ticker   |   MissingPriceRatio | FirstValidDate   | LastValidDate   |   MaxConsecutiveMissingDays | DropReason                                                                                                    |
|:---------|--------------------:|:-----------------|:----------------|----------------------------:|:--------------------------------------------------------------------------------------------------------------|
| APP      |            0.628378 | 2021-04-15       | 2024-12-31      |                        1581 | too_many_missing_prices;history_too_short_or_starts_too_late;long_consecutive_missing_gap_possible_suspension |
| GEV      |            0.923291 | 2024-03-27       | 2024-12-31      |                        2323 | too_many_missing_prices;history_too_short_or_starts_too_late;long_consecutive_missing_gap_possible_suspension |
| PLTR     |            0.574722 | 2020-09-30       | 2024-12-31      |                        1446 | too_many_missing_prices;history_too_short_or_starts_too_late;long_consecutive_missing_gap_possible_suspension |

## 3. 统一交易日历

首先将所有资产的 `AdjClose` 转为宽表，行为交易日，列为资产。为了使所有资产对齐到同一个交易日历，清洗程序先计算每个日期上有非缺失价格的资产比例。若某日期至少有 **80.0%** 的下载资产存在有效调整价格，则该日期进入共同交易日历。

共同交易日历的日期数量为 **2516**。清洗后收益率矩阵的日期数量为 **2514**。收益率日期少于价格日期，原因是收益率采用向前一期收益率定义，最后一个价格日没有 `p[t+1]`，因此必须删除。

## 4. 调整价格的使用

收益率构造使用 `AdjClose`，即 adjusted close。使用 adjusted close 的目的，是尽量消除现金分红、拆股、合股等公司行为对价格序列造成的机械跳变。清洗程序会把小于等于 0 的价格视为无效价格，并设为缺失值。

## 5. 缺失值处理

缺失值处理分为资产层面和日期层面：

1. **资产层面删除**：如果某资产在共同交易日历上的价格缺失比例超过 **5.0%**，则删除该资产。
2. **历史太短资产删除**：如果某资产第一个有效价格晚于起始日超过 **31** 个自然日，则认为其历史太短，删除该资产。
3. **可能退市或停止交易资产删除**：如果某资产最后一个有效价格早于结束日超过 **10** 个自然日，则认为它在样本末端缺少可用交易记录，删除该资产。
4. **长时间停牌或数据缺口删除**：如果某资产在共同交易日历上的连续缺失天数超过 **5** 个交易日，则删除该资产。
5. **短缺口填充**：资产删除后，对保留资产的短价格缺口进行向前填充，最多填充 **2** 个交易日。不使用向后填充，避免用未来价格信息填补过去。
6. **最终收益率缺失处理**：构造收益率后，非有限值和无法计算的收益率设为缺失。默认情况下，含有任一资产缺失收益率的日期会被删除，以得到无缺失的平衡收益率面板。

最终 `cleaned_returns.csv` 中是否仍含缺失值：**False**。

## 6. 退市资产、停牌资产和历史太短资产

- **退市资产**：如果数据源无法下载到资产数据，或资产在样本结束日前很早就没有有效价格，清洗中会将其排除。若研究目标需要严格纳入历史退市收益和退市收益率，应使用 CRSP、Compustat、Refinitiv、Bloomberg 等具备退市收益字段或 survivorship-bias-free 历史成分的数据源。
- **停牌资产**：短暂停牌或短数据缺口最多只向前填充 2 个交易日；超过阈值的长缺口会触发资产删除。这样可以避免把长时间停牌期间人为构造成连续的 0 收益。
- **历史太短资产**：起始阶段缺少足够历史价格的资产会被删除，避免最终面板被 IPO 较晚或上市时间较短的资产主导。

需要注意：如果股票池来自“当前 S&P 500 市值前 80”，则样本天然存在 survivorship bias，因为它没有包含 2015-2024 期间曾经属于大市值股票、但后来被并购、退市或跌出指数的公司。

## 7. 收益率构造方法

清洗后的收益率按以下公式构造：

```text
r[t, i] = (p[t+1, i] - p[t, i]) / p[t, i]
        = p[t+1, i] / p[t, i] - 1
```

其中，`p[t, i]` 是资产 `i` 在日期 `t` 的 `AdjClose`。`cleaned_returns.csv` 中的 `Date` 表示公式中的 `t`，即持有期开始日，而不是 `t+1`。因此，每一行收益率表示从当前交易日收盘价到下一交易日收盘价的简单收益率。

## 8. 异常值处理

异常值处理分两步：

1. **硬阈值过滤**：若单日向前收益率绝对值大于 **50.0%**，则视为高度可疑的数据错误或未被调整价格完全处理的公司行为，设为缺失值。此次硬阈值过滤数量为 **1** 个资产-日期单元。
2. **分资产 Winsorization**：对每只资产的收益率分别按 **0.100%** 和 **99.900%** 分位数进行缩尾，以降低极端尾部值对组合优化、协方差估计和回归估计的影响。此次缩尾处理数量为 **456** 个资产-日期单元。

详细异常值统计见：

`data\cleaned\outlier_report.csv`

## 9. 输出文件

清洗程序生成以下主要文件：

- 清洗后收益率矩阵：`data\cleaned\cleaned_returns.csv`
- 清洗后调整收盘价矩阵：`data\cleaned\cleaned_adjclose_prices.csv`
- 资产清洗诊断表：`data\cleaned\asset_cleaning_diagnostics.csv`
- 异常值处理报告：`data\cleaned\outlier_report.csv`
- 清洗摘要 JSON：`data\cleaned\cleaning_summary.json`
- 本报告：`data\cleaned\data_cleaning_report.md`
