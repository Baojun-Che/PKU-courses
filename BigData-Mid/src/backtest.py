import numpy as np
from .metrics import cumulative_wealth, annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, average_turnover

def backtest_strategy(returns, weights_history, periods_per_year=252):
    """
    returns: T x n numpy array
    weights_history: list of portfolio weights, 每个 rebalance 时更新一次
    """
    wealth_series = []
    w_prev = weights_history[0]
    wealth = 1.0
    rebalancing_period = returns.shape[0] // len(weights_history)

    weights_record = []
    for i, w in enumerate(weights_history):
        # 本次持仓对应的子收益
        start = i*rebalancing_period
        end = (i+1)*rebalancing_period if i < len(weights_history)-1 else returns.shape[0]
        sub_ret = returns[start:end]
        for r in sub_ret:
            wealth *= (1 + r @ w)
            wealth_series.append(wealth)
        weights_record.append(w)

    weights_record = np.array(weights_record)
    cumulative_wealth_series = np.array(wealth_series)
    
    metrics = {
        "cumulative_wealth": cumulative_wealth_series,
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, weights_history[-1], periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, weights_history[-1], periods_per_year),
        "max_drawdown": max_drawdown(cumulative_wealth_series),
        "average_turnover": average_turnover(weights_record)
    }

    return metrics