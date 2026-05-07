import numpy as np

def cumulative_wealth(returns, weights):
    """计算累计财富序列"""
    wealth = np.cumprod(1 + returns @ weights)
    return wealth

def annualized_return(returns, periods_per_year=252):
    """年化收益"""
    mean_daily = np.mean(returns @ np.ones(returns.shape[1])/returns.shape[1])
    return (1 + mean_daily)**periods_per_year - 1

def annualized_volatility(returns, weights, periods_per_year=252):
    """年化波动率"""
    port_ret = returns @ weights
    return np.std(port_ret) * np.sqrt(periods_per_year)

def sharpe_ratio(returns, weights, periods_per_year=252, risk_free_rate=0.0):
    """夏普比率"""
    r_ann = annualized_return(returns, periods_per_year)
    vol_ann = annualized_volatility(returns, weights, periods_per_year)
    return (r_ann - risk_free_rate) / vol_ann

def max_drawdown(wealth):
    """最大回撤"""
    peak = np.maximum.accumulate(wealth)
    drawdown = (peak - wealth) / peak
    return np.max(drawdown)

def average_turnover(weights_history):
    """平均换手率"""
    weights_history = np.array(weights_history)
    turnovers = np.sum(np.abs(weights_history[1:] - weights_history[:-1]), axis=1)
    return np.mean(turnovers)