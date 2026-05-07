#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_data.py

Clean downloaded adjusted daily prices and construct a balanced forward-return panel.

Default input:
    data/raw_prices/all_daily_prices.csv
    data/sp500_top80_universe.csv
    data/raw_prices/download_log.csv

Default output:
    data/cleaned/cleaned_returns.csv
    data/cleaned/cleaned_adjclose_prices.csv
    data/cleaned/asset_cleaning_diagnostics.csv
    data/cleaned/outlier_report.csv
    data/cleaned/cleaning_summary.json
    data/cleaned/data_cleaning_report.md

Return definition:
    r[t, i] = (p[t+1, i] - p[t, i]) / p[t, i]
            = p[t+1, i] / p[t, i] - 1

The Date column in cleaned_returns.csv is the date t, i.e. the day on which the
position is assumed to be formed. The final price date is dropped because it does
not have p[t+1].

Example:
    python clean_data.py \
        --raw-prices-path data/raw_prices/all_daily_prices.csv \
        --universe-path data/sp500_top80_universe.csv \
        --download-log-path data/raw_prices/download_log.csv \
        --output-dir data/cleaned
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean adjusted prices and build cleaned forward returns."
    )
    parser.add_argument(
        "--raw-prices-path",
        default="data/raw_prices/all_daily_prices.csv",
        help="Path to long-format raw daily prices downloaded by download_data.py.",
    )
    parser.add_argument(
        "--universe-path",
        default="data/sp500_top80_universe.csv",
        help="Path to initial universe CSV. Used only for counting/reporting if available.",
    )
    parser.add_argument(
        "--download-log-path",
        default="data/raw_prices/download_log.csv",
        help="Path to download log CSV. Used only for reporting if available.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/cleaned",
        help="Directory for cleaned outputs.",
    )
    parser.add_argument(
        "--start-date",
        default="2015-01-01",
        help="Start date, inclusive, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        default="2024-12-31",
        help="End date, inclusive, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--price-column",
        default="AdjClose",
        help="Adjusted price column used for return construction. Default: AdjClose.",
    )
    parser.add_argument(
        "--calendar-min-asset-ratio",
        type=float,
        default=0.80,
        help=(
            "A date is kept in the common trading calendar only if at least this fraction "
            "of downloaded assets has a non-missing adjusted price on that date."
        ),
    )
    parser.add_argument(
        "--max-price-missing-ratio",
        type=float,
        default=0.05,
        help="Drop assets whose adjusted-price missing ratio on the common calendar exceeds this threshold.",
    )
    parser.add_argument(
        "--max-first-valid-lag-days",
        type=int,
        default=31,
        help=(
            "Drop assets whose first valid adjusted price is more than this many calendar "
            "days after start-date. This removes assets with too short a history."
        ),
    )
    parser.add_argument(
        "--max-last-valid-lag-days",
        type=int,
        default=10,
        help=(
            "Drop assets whose last valid adjusted price is more than this many calendar "
            "days before end-date. This removes likely delisted/stopped assets."
        ),
    )
    parser.add_argument(
        "--max-consecutive-missing-days",
        type=int,
        default=5,
        help=(
            "Drop assets with a longer consecutive missing-price run on the common trading calendar. "
            "This catches long suspensions or serious data holes."
        ),
    )
    parser.add_argument(
        "--ffill-limit",
        type=int,
        default=2,
        help=(
            "Forward-fill short adjusted-price gaps up to this many trading days after bad assets are removed. "
            "No backward fill is used."
        ),
    )
    parser.add_argument(
        "--hard-return-cap",
        type=float,
        default=0.50,
        help=(
            "Absolute one-day forward returns larger than this are treated as suspicious data errors "
            "and set to missing before final date filtering. Default: 0.50 means 50%."
        ),
    )
    parser.add_argument(
        "--winsor-lower",
        type=float,
        default=0.001,
        help="Per-asset lower winsorization quantile for returns. Use 0 to disable lower clipping.",
    )
    parser.add_argument(
        "--winsor-upper",
        type=float,
        default=0.999,
        help="Per-asset upper winsorization quantile for returns. Use 1 to disable upper clipping.",
    )
    parser.add_argument(
        "--keep-missing-returns",
        action="store_true",
        help=(
            "If set, keep missing values in the final return panel. By default, dates with any missing "
            "return are dropped to produce a balanced panel."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if pd.Timestamp(args.start_date) > pd.Timestamp(args.end_date):
        raise ValueError("start-date must be earlier than or equal to end-date.")

    ratio_args = {
        "calendar_min_asset_ratio": args.calendar_min_asset_ratio,
        "max_price_missing_ratio": args.max_price_missing_ratio,
        "winsor_lower": args.winsor_lower,
        "winsor_upper": args.winsor_upper,
    }
    for name, value in ratio_args.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]. Got {value}.")
    if args.winsor_lower >= args.winsor_upper:
        raise ValueError("winsor-lower must be smaller than winsor-upper.")
    if args.hard_return_cap <= 0:
        raise ValueError("hard-return-cap must be positive.")
    if args.ffill_limit < 0:
        raise ValueError("ffill-limit must be non-negative.")


def max_consecutive_true(values: Iterable[bool]) -> int:
    max_run = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return int(max_run)


def normalize_ticker_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(".", "-", regex=False).str.upper()


def count_initial_universe(universe_path: str) -> Tuple[Optional[int], Optional[List[str]]]:
    path = Path(universe_path)
    if not path.exists():
        return None, None

    universe = pd.read_csv(path)
    if "YahooTicker" in universe.columns:
        tickers = normalize_ticker_series(universe["YahooTicker"]).dropna().unique().tolist()
    elif "Ticker" in universe.columns:
        tickers = normalize_ticker_series(universe["Ticker"]).dropna().unique().tolist()
    else:
        return int(len(universe)), None
    return int(len(tickers)), tickers


def read_download_log(download_log_path: str) -> Optional[pd.DataFrame]:
    path = Path(download_log_path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_raw_prices(raw_prices_path: str, price_column: str, start_date: str, end_date: str) -> pd.DataFrame:
    path = Path(raw_prices_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw price file not found: {raw_prices_path}")

    df = pd.read_csv(path)
    required = {"Date", "Ticker", price_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Raw price file is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = normalize_ticker_series(df["Ticker"])
    df[price_column] = pd.to_numeric(df[price_column], errors="coerce")

    # Invalid adjusted prices cannot produce meaningful returns.
    df.loc[df[price_column] <= 0, price_column] = np.nan

    df = df.dropna(subset=["Date", "Ticker"])
    df = df[(df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))]
    df = df.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No raw price rows remain after date filtering and basic validation.")

    return df


def build_price_matrix(df: pd.DataFrame, price_column: str) -> pd.DataFrame:
    prices = df.pivot(index="Date", columns="Ticker", values=price_column)
    prices = prices.sort_index().sort_index(axis=1)
    return prices


def build_common_calendar(prices: pd.DataFrame, min_asset_ratio: float) -> Tuple[pd.DatetimeIndex, pd.Series]:
    coverage_ratio = prices.notna().sum(axis=1) / max(prices.shape[1], 1)
    calendar = coverage_ratio[coverage_ratio >= min_asset_ratio].index

    # Fallback: if threshold is too strict, use all dates with at least one available asset.
    if len(calendar) == 0:
        calendar = coverage_ratio[coverage_ratio > 0].index

    return pd.DatetimeIndex(calendar), coverage_ratio


def asset_diagnostics(
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    max_missing_ratio: float,
    max_first_valid_lag_days: int,
    max_last_valid_lag_days: int,
    max_consecutive_missing_days: int,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    first_valid_deadline = start_ts + pd.Timedelta(days=max_first_valid_lag_days)
    last_valid_deadline = end_ts - pd.Timedelta(days=max_last_valid_lag_days)

    records: List[Dict[str, Any]] = []

    for ticker in prices.columns:
        s = prices[ticker]
        valid = s.notna()
        valid_obs = int(valid.sum())
        total_obs = int(len(s))
        missing_obs = int(total_obs - valid_obs)
        missing_ratio = float(missing_obs / total_obs) if total_obs > 0 else 1.0

        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()
        max_gap = max_consecutive_true(s.isna().tolist())

        reasons: List[str] = []
        if valid_obs == 0:
            reasons.append("no_valid_adjusted_price")
        if missing_ratio > max_missing_ratio:
            reasons.append("too_many_missing_prices")
        if first_valid is None or pd.Timestamp(first_valid) > first_valid_deadline:
            reasons.append("history_too_short_or_starts_too_late")
        if last_valid is None or pd.Timestamp(last_valid) < last_valid_deadline:
            reasons.append("ends_too_early_possible_delisting_or_stopped_trading")
        if max_gap > max_consecutive_missing_days:
            reasons.append("long_consecutive_missing_gap_possible_suspension")

        records.append(
            {
                "Ticker": ticker,
                "TotalCalendarDays": total_obs,
                "ValidPriceObservations": valid_obs,
                "MissingPriceObservations": missing_obs,
                "MissingPriceRatio": missing_ratio,
                "FirstValidDate": "" if first_valid is None else str(pd.Timestamp(first_valid).date()),
                "LastValidDate": "" if last_valid is None else str(pd.Timestamp(last_valid).date()),
                "MaxConsecutiveMissingDays": max_gap,
                "KeepAsset": len(reasons) == 0,
                "DropReason": ";".join(reasons),
            }
        )

    diag = pd.DataFrame(records).sort_values(["KeepAsset", "Ticker"], ascending=[True, True])
    return diag


def forward_fill_short_gaps(prices: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit == 0:
        return prices.copy()
    return prices.ffill(limit=limit)


def construct_forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.shift(-1) / prices - 1.0
    # The last price date has no t+1 price.
    returns = returns.iloc[:-1, :]
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns


def handle_outliers(
    returns: pd.DataFrame,
    hard_return_cap: float,
    winsor_lower: float,
    winsor_upper: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    out = returns.copy()

    hard_mask = out.abs() > hard_return_cap
    hard_outlier_count = int(hard_mask.sum().sum())
    out = out.mask(hard_mask)

    records: List[Dict[str, Any]] = []
    total_lower_clipped = 0
    total_upper_clipped = 0

    for ticker in out.columns:
        s = out[ticker]
        nonmissing = s.dropna()

        if len(nonmissing) == 0:
            records.append(
                {
                    "Ticker": ticker,
                    "HardOutlierCount": int(hard_mask[ticker].sum()),
                    "WinsorLowerQuantile": winsor_lower,
                    "WinsorUpperQuantile": winsor_upper,
                    "LowerClipValue": np.nan,
                    "UpperClipValue": np.nan,
                    "LowerClippedCount": 0,
                    "UpperClippedCount": 0,
                }
            )
            continue

        lower_value = nonmissing.quantile(winsor_lower) if winsor_lower > 0 else -np.inf
        upper_value = nonmissing.quantile(winsor_upper) if winsor_upper < 1 else np.inf

        lower_count = int((s < lower_value).sum()) if np.isfinite(lower_value) else 0
        upper_count = int((s > upper_value).sum()) if np.isfinite(upper_value) else 0

        total_lower_clipped += lower_count
        total_upper_clipped += upper_count

        out[ticker] = s.clip(lower=lower_value, upper=upper_value)

        records.append(
            {
                "Ticker": ticker,
                "HardOutlierCount": int(hard_mask[ticker].sum()),
                "WinsorLowerQuantile": winsor_lower,
                "WinsorUpperQuantile": winsor_upper,
                "LowerClipValue": None if not np.isfinite(lower_value) else float(lower_value),
                "UpperClipValue": None if not np.isfinite(upper_value) else float(upper_value),
                "LowerClippedCount": lower_count,
                "UpperClippedCount": upper_count,
            }
        )

    outlier_report = pd.DataFrame(records).sort_values("Ticker")
    total_winsor_clipped = int(total_lower_clipped + total_upper_clipped)
    return out, outlier_report, hard_outlier_count, total_winsor_clipped


def make_report(summary: Dict[str, Any], dropped_assets: pd.DataFrame) -> str:
    drop_reason_counts = {}
    if not dropped_assets.empty:
        for reasons in dropped_assets["DropReason"].fillna(""):
            for reason in str(reasons).split(";"):
                if reason:
                    drop_reason_counts[reason] = drop_reason_counts.get(reason, 0) + 1

    drop_reason_lines = ""
    if drop_reason_counts:
        for reason, count in sorted(drop_reason_counts.items(), key=lambda x: x[0]):
            drop_reason_lines += f"- `{reason}`：{count} 只资产\n"
    else:
        drop_reason_lines = "- 无资产因清洗规则被删除。\n"

    dropped_list = ""
    if not dropped_assets.empty:
        cols = ["Ticker", "MissingPriceRatio", "FirstValidDate", "LastValidDate", "MaxConsecutiveMissingDays", "DropReason"]
        available_cols = [c for c in cols if c in dropped_assets.columns]
        dropped_list = dropped_assets[available_cols].to_markdown(index=False)
    else:
        dropped_list = "无。"

    return f"""# 数据清洗报告

## 1. 数据范围与输入文件

本次清洗使用 `download_data.py` 下载得到的日频价格数据，并使用 `{summary['price_column']}` 作为价格变量。样本区间为 **{summary['start_date']} 至 {summary['end_date']}**。原始价格文件为：

`{summary['raw_prices_path']}`

初始股票池文件为：

`{summary['universe_path']}`

下载日志文件为：

`{summary['download_log_path']}`

## 2. 清洗前后资产数量

- 初始股票池资产数量：{summary['initial_universe_asset_count']}
- 下载成功资产数量：{summary['download_success_asset_count']}
- 进入价格矩阵的资产数量：{summary['downloaded_price_asset_count']}
- 清洗后保留资产数量：{summary['cleaned_asset_count']}
- 因价格缺失、历史太短、结束太早或长时间缺口被删除的资产数量：{summary['dropped_asset_count']}

删除原因汇总：

{drop_reason_lines}

被删除资产明细：

{dropped_list}

## 3. 统一交易日历

首先将所有资产的 `{summary['price_column']}` 转为宽表，行为交易日，列为资产。为了使所有资产对齐到同一个交易日历，清洗程序先计算每个日期上有非缺失价格的资产比例。若某日期至少有 **{summary['calendar_min_asset_ratio']:.1%}** 的下载资产存在有效调整价格，则该日期进入共同交易日历。

共同交易日历的日期数量为 **{summary['common_calendar_days']}**。清洗后收益率矩阵的日期数量为 **{summary['cleaned_return_days']}**。收益率日期少于价格日期，原因是收益率采用向前一期收益率定义，最后一个价格日没有 `p[t+1]`，因此必须删除。

## 4. 调整价格的使用

收益率构造使用 `{summary['price_column']}`，即 adjusted close。使用 adjusted close 的目的，是尽量消除现金分红、拆股、合股等公司行为对价格序列造成的机械跳变。清洗程序会把小于等于 0 的价格视为无效价格，并设为缺失值。

## 5. 缺失值处理

缺失值处理分为资产层面和日期层面：

1. **资产层面删除**：如果某资产在共同交易日历上的价格缺失比例超过 **{summary['max_price_missing_ratio']:.1%}**，则删除该资产。
2. **历史太短资产删除**：如果某资产第一个有效价格晚于起始日超过 **{summary['max_first_valid_lag_days']}** 个自然日，则认为其历史太短，删除该资产。
3. **可能退市或停止交易资产删除**：如果某资产最后一个有效价格早于结束日超过 **{summary['max_last_valid_lag_days']}** 个自然日，则认为它在样本末端缺少可用交易记录，删除该资产。
4. **长时间停牌或数据缺口删除**：如果某资产在共同交易日历上的连续缺失天数超过 **{summary['max_consecutive_missing_days']}** 个交易日，则删除该资产。
5. **短缺口填充**：资产删除后，对保留资产的短价格缺口进行向前填充，最多填充 **{summary['ffill_limit']}** 个交易日。不使用向后填充，避免用未来价格信息填补过去。
6. **最终收益率缺失处理**：构造收益率后，非有限值和无法计算的收益率设为缺失。默认情况下，含有任一资产缺失收益率的日期会被删除，以得到无缺失的平衡收益率面板。

最终 `cleaned_returns.csv` 中是否仍含缺失值：**{summary['final_returns_have_missing']}**。

## 6. 退市资产、停牌资产和历史太短资产

- **退市资产**：如果数据源无法下载到资产数据，或资产在样本结束日前很早就没有有效价格，清洗中会将其排除。若研究目标需要严格纳入历史退市收益和退市收益率，应使用 CRSP、Compustat、Refinitiv、Bloomberg 等具备退市收益字段或 survivorship-bias-free 历史成分的数据源。
- **停牌资产**：短暂停牌或短数据缺口最多只向前填充 {summary['ffill_limit']} 个交易日；超过阈值的长缺口会触发资产删除。这样可以避免把长时间停牌期间人为构造成连续的 0 收益。
- **历史太短资产**：起始阶段缺少足够历史价格的资产会被删除，避免最终面板被 IPO 较晚或上市时间较短的资产主导。

需要注意：如果股票池来自“当前 S&P 500 市值前 80”，则样本天然存在 survivorship bias，因为它没有包含 2015-2024 期间曾经属于大市值股票、但后来被并购、退市或跌出指数的公司。

## 7. 收益率构造方法

清洗后的收益率按以下公式构造：

```text
r[t, i] = (p[t+1, i] - p[t, i]) / p[t, i]
        = p[t+1, i] / p[t, i] - 1
```

其中，`p[t, i]` 是资产 `i` 在日期 `t` 的 `{summary['price_column']}`。`cleaned_returns.csv` 中的 `Date` 表示公式中的 `t`，即持有期开始日，而不是 `t+1`。因此，每一行收益率表示从当前交易日收盘价到下一交易日收盘价的简单收益率。

## 8. 异常值处理

异常值处理分两步：

1. **硬阈值过滤**：若单日向前收益率绝对值大于 **{summary['hard_return_cap']:.1%}**，则视为高度可疑的数据错误或未被调整价格完全处理的公司行为，设为缺失值。此次硬阈值过滤数量为 **{summary['hard_outlier_count']}** 个资产-日期单元。
2. **分资产 Winsorization**：对每只资产的收益率分别按 **{summary['winsor_lower']:.3%}** 和 **{summary['winsor_upper']:.3%}** 分位数进行缩尾，以降低极端尾部值对组合优化、协方差估计和回归估计的影响。此次缩尾处理数量为 **{summary['winsor_clipped_count']}** 个资产-日期单元。

详细异常值统计见：

`{summary['outlier_report_path']}`

## 9. 输出文件

清洗程序生成以下主要文件：

- 清洗后收益率矩阵：`{summary['cleaned_returns_path']}`
- 清洗后调整收盘价矩阵：`{summary['cleaned_prices_path']}`
- 资产清洗诊断表：`{summary['asset_diagnostics_path']}`
- 异常值处理报告：`{summary['outlier_report_path']}`
- 清洗摘要 JSON：`{summary['cleaning_summary_path']}`
- 本报告：`{summary['report_path']}`
"""


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_universe_count, initial_tickers = count_initial_universe(args.universe_path)
    download_log = read_download_log(args.download_log_path)

    if download_log is not None and "Status" in download_log.columns:
        download_success_count: Any = int((download_log["Status"] == "Success").sum())
        download_failed_count: Any = int((download_log["Status"] == "Failed").sum())
    else:
        download_success_count = "unknown"
        download_failed_count = "unknown"

    raw = load_raw_prices(
        raw_prices_path=args.raw_prices_path,
        price_column=args.price_column,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    duplicate_count = int(raw.duplicated(subset=["Date", "Ticker"]).sum())
    prices = build_price_matrix(raw, args.price_column)
    downloaded_price_asset_count = int(prices.shape[1])

    common_calendar, coverage_ratio = build_common_calendar(
        prices=prices,
        min_asset_ratio=args.calendar_min_asset_ratio,
    )
    prices_aligned = prices.reindex(common_calendar).sort_index()

    diagnostics = asset_diagnostics(
        prices=prices_aligned,
        start_date=args.start_date,
        end_date=args.end_date,
        max_missing_ratio=args.max_price_missing_ratio,
        max_first_valid_lag_days=args.max_first_valid_lag_days,
        max_last_valid_lag_days=args.max_last_valid_lag_days,
        max_consecutive_missing_days=args.max_consecutive_missing_days,
    )

    kept_assets = diagnostics.loc[diagnostics["KeepAsset"], "Ticker"].tolist()
    dropped_assets = diagnostics.loc[~diagnostics["KeepAsset"]].copy()

    if not kept_assets:
        diagnostics_path = output_dir / "asset_cleaning_diagnostics.csv"
        diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
        raise RuntimeError(
            "No assets passed the cleaning filters. Consider relaxing missing/history thresholds. "
            f"Diagnostics saved to {diagnostics_path}"
        )

    prices_kept = prices_aligned[kept_assets].copy()
    prices_filled = forward_fill_short_gaps(prices_kept, args.ffill_limit)

    returns_raw = construct_forward_returns(prices_filled)
    returns_cleaned, outlier_report, hard_outlier_count, winsor_clipped_count = handle_outliers(
        returns=returns_raw,
        hard_return_cap=args.hard_return_cap,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
    )

    rows_before_missing_drop = int(len(returns_cleaned))
    missing_cells_before_final_drop = int(returns_cleaned.isna().sum().sum())

    if args.keep_missing_returns:
        final_returns = returns_cleaned.copy()
        rows_dropped_due_to_missing_returns = 0
    else:
        final_returns = returns_cleaned.dropna(axis=0, how="any")
        rows_dropped_due_to_missing_returns = int(rows_before_missing_drop - len(final_returns))

    # Match cleaned prices to dates appearing in returns. Because returns at t uses p[t] and p[t+1],
    # the cleaned price matrix saved here keeps the same row dates as final_returns.
    cleaned_prices_for_return_dates = prices_filled.reindex(final_returns.index)

    cleaned_returns_path = output_dir / "cleaned_returns.csv"
    cleaned_prices_path = output_dir / "cleaned_adjclose_prices.csv"
    diagnostics_path = output_dir / "asset_cleaning_diagnostics.csv"
    outlier_report_path = output_dir / "outlier_report.csv"
    coverage_path = output_dir / "calendar_coverage.csv"
    summary_path = output_dir / "cleaning_summary.json"
    report_path = output_dir / "data_cleaning_report.md"

    final_returns.to_csv(cleaned_returns_path, index=True, index_label="Date", encoding="utf-8-sig")
    cleaned_prices_for_return_dates.to_csv(cleaned_prices_path, index=True, index_label="Date", encoding="utf-8-sig")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    outlier_report.to_csv(outlier_report_path, index=False, encoding="utf-8-sig")
    coverage_ratio.rename("AssetCoverageRatio").to_csv(coverage_path, index=True, index_label="Date", encoding="utf-8-sig")

    summary: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_prices_path": args.raw_prices_path,
        "universe_path": args.universe_path,
        "download_log_path": args.download_log_path,
        "price_column": args.price_column,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_universe_asset_count": initial_universe_count if initial_universe_count is not None else "unknown",
        "download_success_asset_count": download_success_count,
        "download_failed_asset_count": download_failed_count,
        "downloaded_price_asset_count": downloaded_price_asset_count,
        "cleaned_asset_count": int(final_returns.shape[1]),
        "dropped_asset_count": int(len(dropped_assets)),
        "common_calendar_days": int(len(common_calendar)),
        "raw_price_rows": int(len(raw)),
        "duplicate_rows_after_basic_load": duplicate_count,
        "cleaned_return_days": int(len(final_returns)),
        "calendar_min_asset_ratio": args.calendar_min_asset_ratio,
        "max_price_missing_ratio": args.max_price_missing_ratio,
        "max_first_valid_lag_days": args.max_first_valid_lag_days,
        "max_last_valid_lag_days": args.max_last_valid_lag_days,
        "max_consecutive_missing_days": args.max_consecutive_missing_days,
        "ffill_limit": args.ffill_limit,
        "hard_return_cap": args.hard_return_cap,
        "hard_outlier_count": hard_outlier_count,
        "winsor_lower": args.winsor_lower,
        "winsor_upper": args.winsor_upper,
        "winsor_clipped_count": winsor_clipped_count,
        "rows_before_missing_drop": rows_before_missing_drop,
        "missing_cells_before_final_drop": missing_cells_before_final_drop,
        "rows_dropped_due_to_missing_returns": rows_dropped_due_to_missing_returns,
        "final_returns_have_missing": bool(final_returns.isna().any().any()),
        "cleaned_returns_path": str(cleaned_returns_path),
        "cleaned_prices_path": str(cleaned_prices_path),
        "asset_diagnostics_path": str(diagnostics_path),
        "outlier_report_path": str(outlier_report_path),
        "calendar_coverage_path": str(coverage_path),
        "cleaning_summary_path": str(summary_path),
        "report_path": str(report_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report = make_report(summary, dropped_assets)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n========== 清洗完成 ==========")
    print(f"进入价格矩阵的资产数量：{downloaded_price_asset_count}")
    print(f"清洗后保留资产数量：{final_returns.shape[1]}")
    print(f"删除资产数量：{len(dropped_assets)}")
    print(f"共同交易日历天数：{len(common_calendar)}")
    print(f"最终收益率日期数量：{len(final_returns)}")
    print(f"最终收益率是否含缺失值：{final_returns.isna().any().any()}")
    print(f"清洗后收益率矩阵：{cleaned_returns_path}")
    print(f"数据清洗报告：{report_path}")


if __name__ == "__main__":
    main()
