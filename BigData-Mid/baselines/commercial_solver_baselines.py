"""
Commercial Solver Baselines for Portfolio Optimization Mid-Term Project.

This module solves:
  - Model A2: Quadratic Utility Portfolio, swept over risk-aversion gamma.
  - Model B : Return Maximization under a standard-deviation budget, swept over sigma_max.

Default solver is MOSEK through CVXPY. GUROBI can also be selected if installed and licensed.
The code assumes that mu and Sigma have already been estimated from the training window.

If this module was adapted with LLM assistance, state that in your submission if required by
course policy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OPTIMAL_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


@dataclass
class PortfolioResult:
    model: str
    parameter_name: str
    parameter_value: float
    status: str
    objective_value: float
    expected_return: float
    variance: float
    standard_deviation: float
    annualized_return: float
    annualized_volatility: float
    solve_time_sec: float
    n_assets: int
    max_weight: float
    min_weight: float
    active_assets_1pct: int
    budget_error: float
    min_weight_violation: float


def load_estimates(input_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load mu, Sigma, and tickers from a .npz file or a directory of CSV files.

    Supported formats
    -----------------
    1) NPZ file with keys such as:
         mu / mu_hat / expected_returns
         Sigma / Sigma_hat / covariance / cov
         tickers / assets  [optional]

    2) Directory with:
         mu.csv and Sigma.csv
       or
         mu_hat.csv and Sigma_hat.csv
       Optional:
         tickers.csv

    Returns
    -------
    mu : shape (n,)
    Sigma : shape (n, n)
    tickers : list of length n
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        keys = set(data.files)

        def first_key(candidates: Iterable[str]) -> str:
            for k in candidates:
                if k in keys:
                    return k
            raise KeyError(f"None of the keys {list(candidates)} were found in {path}. Found keys: {sorted(keys)}")

        mu_key = first_key(["mu", "mu_hat", "expected_returns", "mean", "means"])
        sigma_key = first_key(["Sigma", "Sigma_hat", "covariance", "cov", "sigma", "Sigma_epsilon"])
        mu = np.asarray(data[mu_key], dtype=float).reshape(-1)
        Sigma = np.asarray(data[sigma_key], dtype=float)
        if "tickers" in keys:
            tickers = [str(x) for x in np.asarray(data["tickers"]).reshape(-1)]
        elif "assets" in keys:
            tickers = [str(x) for x in np.asarray(data["assets"]).reshape(-1)]
        else:
            tickers = [f"asset_{i:03d}" for i in range(len(mu))]
        return validate_estimates(mu, Sigma, tickers)

    if path.is_dir():
        candidates = [
            ("mu.csv", "Sigma.csv"),
            ("mu_hat.csv", "Sigma_hat.csv"),
            ("expected_returns.csv", "covariance.csv"),
        ]
        for mu_name, sigma_name in candidates:
            mu_file = path / mu_name
            sigma_file = path / sigma_name
            if mu_file.exists() and sigma_file.exists():
                mu_df = pd.read_csv(mu_file, index_col=0)
                # mu may be a one-column file or a one-row file.
                if mu_df.shape[1] == 1:
                    tickers = [str(i) for i in mu_df.index]
                    mu = mu_df.iloc[:, 0].to_numpy(dtype=float)
                else:
                    # If saved as one row with columns=tickers.
                    tickers = [str(c) for c in mu_df.columns]
                    mu = mu_df.iloc[0, :].to_numpy(dtype=float)

                Sigma_df = pd.read_csv(sigma_file, index_col=0)
                Sigma = Sigma_df.to_numpy(dtype=float)
                # Prefer covariance labels if they look complete.
                if Sigma_df.shape[0] == len(Sigma_df.columns) and len(Sigma_df.columns) == len(mu):
                    tickers = [str(c) for c in Sigma_df.columns]
                return validate_estimates(mu, Sigma, tickers)

        raise FileNotFoundError(
            f"Could not find a supported pair of estimate files under {path}. "
            "Expected mu.csv/Sigma.csv, mu_hat.csv/Sigma_hat.csv, or expected_returns.csv/covariance.csv."
        )

    raise ValueError(f"Unsupported input format: {path}. Use .npz or a directory of CSV files.")


def validate_estimates(mu: np.ndarray, Sigma: np.ndarray, tickers: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size
    if Sigma.shape != (n, n):
        raise ValueError(f"Sigma must have shape ({n}, {n}); got {Sigma.shape}.")
    if len(tickers) != n:
        tickers = [f"asset_{i:03d}" for i in range(n)]
    if not np.all(np.isfinite(mu)):
        raise ValueError("mu contains NaN or infinite values.")
    if not np.all(np.isfinite(Sigma)):
        raise ValueError("Sigma contains NaN or infinite values.")
    Sigma = 0.5 * (Sigma + Sigma.T)
    return mu, Sigma, tickers


def stabilize_covariance(Sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Symmetrize and shift covariance to be numerically PSD."""
    Sigma = 0.5 * (Sigma + Sigma.T)
    eig_min = float(np.linalg.eigvalsh(Sigma).min())
    shift = max(eps, -eig_min + eps)
    return Sigma + shift * np.eye(Sigma.shape[0])


def covariance_factor(Sigma_psd: np.ndarray, method: str = "auto") -> np.ndarray:
    """Return B such that Sigma_psd approximately equals B @ B.T.

    For Model B, the SOC risk constraint uses ||B.T @ x||_2.
    """
    Sigma_psd = 0.5 * (Sigma_psd + Sigma_psd.T)
    if method in {"auto", "cholesky"}:
        try:
            return np.linalg.cholesky(Sigma_psd)
        except np.linalg.LinAlgError:
            if method == "cholesky":
                raise
    eigvals, eigvecs = np.linalg.eigh(Sigma_psd)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals))


def choose_default_gammas(mu: np.ndarray, Sigma: np.ndarray, num: int = 30) -> np.ndarray:
    """Construct a scale-aware gamma grid for Model A2.

    If returns are daily, typical mu is around 1e-4 to 1e-3 and diagonal covariance around
    1e-4; gamma values around |mu|/var often produce non-degenerate portfolios. The grid spans
    four orders of magnitude around that center.
    """
    avg_abs_mu = float(np.mean(np.abs(mu)))
    avg_var = float(np.mean(np.diag(Sigma)))
    if avg_abs_mu <= 0 or avg_var <= 0:
        center = 1.0
    else:
        center = avg_abs_mu / avg_var
    grid = center * np.logspace(-2, 2, num=num)
    return np.unique(np.maximum(grid, 1e-10))


def clean_weights(w: Optional[np.ndarray], tol: float = 1e-10) -> Optional[np.ndarray]:
    if w is None:
        return None
    w = np.asarray(w, dtype=float).reshape(-1)
    w[np.abs(w) < tol] = 0.0
    # Small negative values may appear because of solver tolerances; keep large violations visible.
    if w.min() >= -1e-7:
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s > 0:
            w = w / s
    return w


def portfolio_metrics(
    model: str,
    parameter_name: str,
    parameter_value: float,
    status: str,
    objective_value: float,
    weights: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    solve_time_sec: float,
    periods_per_year: float,
) -> PortfolioResult:
    ret = float(mu @ weights)
    var = float(weights @ Sigma @ weights)
    var = max(var, 0.0)
    std = math.sqrt(var)
    ann_ret = periods_per_year * ret
    ann_vol = math.sqrt(periods_per_year) * std
    return PortfolioResult(
        model=model,
        parameter_name=parameter_name,
        parameter_value=float(parameter_value),
        status=status,
        objective_value=float(objective_value),
        expected_return=ret,
        variance=var,
        standard_deviation=std,
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        solve_time_sec=float(solve_time_sec),
        n_assets=len(weights),
        max_weight=float(np.max(weights)),
        min_weight=float(np.min(weights)),
        active_assets_1pct=int(np.sum(weights >= 0.01)),
        budget_error=float(abs(weights.sum() - 1.0)),
        min_weight_violation=float(max(0.0, -np.min(weights))),
    )


def solve_qp_a2(
    mu: np.ndarray,
    Sigma_psd: np.ndarray,
    gamma: float,
    solver: str = "MOSEK",
    verbose: bool = False,
    solver_options: Optional[dict] = None,
    periods_per_year: float = 252.0,
) -> tuple[PortfolioResult, np.ndarray]:
    """Solve Model A2 with CVXPY and a commercial solver."""
    n = len(mu)
    x = cp.Variable(n)
    Sigma_const = cp.psd_wrap(Sigma_psd)
    objective = cp.Minimize(0.5 * gamma * cp.quad_form(x, Sigma_const) - mu @ x)
    constraints = [cp.sum(x) == 1, x >= 0]
    problem = cp.Problem(objective, constraints)

    options = solver_options or {}
    t0 = time.perf_counter()
    problem.solve(solver=getattr(cp, solver), verbose=verbose, **options)
    solve_time = time.perf_counter() - t0

    if problem.status not in OPTIMAL_STATUSES or x.value is None:
        weights = np.full(n, np.nan)
        result = PortfolioResult(
            model="QP_A2",
            parameter_name="gamma",
            parameter_value=float(gamma),
            status=str(problem.status),
            objective_value=float(problem.value) if problem.value is not None else np.nan,
            expected_return=np.nan,
            variance=np.nan,
            standard_deviation=np.nan,
            annualized_return=np.nan,
            annualized_volatility=np.nan,
            solve_time_sec=solve_time,
            n_assets=n,
            max_weight=np.nan,
            min_weight=np.nan,
            active_assets_1pct=0,
            budget_error=np.nan,
            min_weight_violation=np.nan,
        )
        return result, weights

    weights = clean_weights(x.value)
    result = portfolio_metrics(
        model="QP_A2",
        parameter_name="gamma",
        parameter_value=gamma,
        status=str(problem.status),
        objective_value=float(problem.value),
        weights=weights,
        mu=mu,
        Sigma=Sigma_psd,
        solve_time_sec=solve_time,
        periods_per_year=periods_per_year,
    )
    return result, weights


def solve_min_variance(
    Sigma_psd: np.ndarray,
    solver: str = "MOSEK",
    verbose: bool = False,
    solver_options: Optional[dict] = None,
) -> tuple[float, np.ndarray, str]:
    """Solve the long-only minimum variance problem to set a feasible SOCP risk lower bound."""
    n = Sigma_psd.shape[0]
    x = cp.Variable(n)
    problem = cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(Sigma_psd))), [cp.sum(x) == 1, x >= 0])
    problem.solve(solver=getattr(cp, solver), verbose=verbose, **(solver_options or {}))
    if problem.status not in OPTIMAL_STATUSES or x.value is None:
        return np.nan, np.full(n, np.nan), str(problem.status)
    w = clean_weights(x.value)
    risk = math.sqrt(max(float(w @ Sigma_psd @ w), 0.0))
    return risk, w, str(problem.status)


def solve_socp_b(
    mu: np.ndarray,
    Sigma_psd: np.ndarray,
    sigma_max: float,
    solver: str = "MOSEK",
    verbose: bool = False,
    solver_options: Optional[dict] = None,
    periods_per_year: float = 252.0,
) -> tuple[PortfolioResult, np.ndarray]:
    """Solve Model B with CVXPY and a commercial conic solver."""
    n = len(mu)
    x = cp.Variable(n)
    B = covariance_factor(Sigma_psd)
    objective = cp.Maximize(mu @ x)
    constraints = [cp.norm(B.T @ x, 2) <= float(sigma_max), cp.sum(x) == 1, x >= 0]
    problem = cp.Problem(objective, constraints)

    options = solver_options or {}
    t0 = time.perf_counter()
    problem.solve(solver=getattr(cp, solver), verbose=verbose, **options)
    solve_time = time.perf_counter() - t0

    if problem.status not in OPTIMAL_STATUSES or x.value is None:
        weights = np.full(n, np.nan)
        result = PortfolioResult(
            model="SOCP_B",
            parameter_name="sigma_max",
            parameter_value=float(sigma_max),
            status=str(problem.status),
            objective_value=float(problem.value) if problem.value is not None else np.nan,
            expected_return=np.nan,
            variance=np.nan,
            standard_deviation=np.nan,
            annualized_return=np.nan,
            annualized_volatility=np.nan,
            solve_time_sec=solve_time,
            n_assets=n,
            max_weight=np.nan,
            min_weight=np.nan,
            active_assets_1pct=0,
            budget_error=np.nan,
            min_weight_violation=np.nan,
        )
        return result, weights

    weights = clean_weights(x.value)
    # CVXPY maximization returns prob.value = mu^T x. For a common summary table,
    # store objective_value as the maximized return rather than changing the sign.
    result = portfolio_metrics(
        model="SOCP_B",
        parameter_name="sigma_max",
        parameter_value=sigma_max,
        status=str(problem.status),
        objective_value=float(problem.value),
        weights=weights,
        mu=mu,
        Sigma=Sigma_psd,
        solve_time_sec=solve_time,
        periods_per_year=periods_per_year,
    )
    return result, weights


def parse_float_list(value: Optional[str]) -> Optional[np.ndarray]:
    if value is None or value.strip() == "":
        return None
    return np.array([float(v.strip()) for v in value.split(",") if v.strip() != ""], dtype=float)


def dataframe_from_results(results: list[PortfolioResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def save_weights_matrix(
    weights_by_param: dict[float, np.ndarray],
    tickers: list[str],
    out_file: Path,
    parameter_name: str,
) -> pd.DataFrame:
    rows = []
    for p, w in weights_by_param.items():
        row = {parameter_name: p}
        for ticker, weight in zip(tickers, w):
            row[ticker] = weight
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(parameter_name)
    df.to_csv(out_file, index=False)
    return df


def representative_table(
    frontier_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    parameter_name: str,
    tickers: list[str],
    out_file: Path,
    k_weights: int = 10,
) -> pd.DataFrame:
    """Select low/mid/high risk rows and report their largest holdings."""
    valid = frontier_df[frontier_df["status"].isin(["optimal", "optimal_inaccurate"])].copy()
    valid = valid.sort_values("standard_deviation")
    if valid.empty:
        out = pd.DataFrame()
        out.to_csv(out_file, index=False)
        return out

    positions = sorted(set([0, len(valid) // 2, len(valid) - 1]))
    rows = []
    for idx in positions:
        info = valid.iloc[idx].to_dict()
        p = info["parameter_value"]
        # Match by floating point tolerance.
        weight_row = weights_df.iloc[(weights_df[parameter_name] - p).abs().argmin()]
        weights = weight_row[tickers].to_numpy(dtype=float)
        top_idx = np.argsort(weights)[::-1][:k_weights]
        top_holdings = "; ".join([f"{tickers[j]}={weights[j]:.4f}" for j in top_idx if np.isfinite(weights[j])])
        rows.append(
            {
                "case": ["low_risk", "middle_risk", "high_risk"][len(rows)] if len(positions) == 3 else f"case_{len(rows)+1}",
                parameter_name: p,
                "expected_return": info["expected_return"],
                "standard_deviation": info["standard_deviation"],
                "annualized_return": info["annualized_return"],
                "annualized_volatility": info["annualized_volatility"],
                "objective_value": info["objective_value"],
                "status": info["status"],
                "solve_time_sec": info["solve_time_sec"],
                "max_weight": info["max_weight"],
                "active_assets_1pct": int(info["active_assets_1pct"]),
                f"top_{k_weights}_holdings": top_holdings,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_file, index=False)
    return out


def plot_qp_frontier(qp_df: pd.DataFrame, out_file: Path) -> None:
    valid = qp_df[qp_df["status"].isin(["optimal", "optimal_inaccurate"])].sort_values("annualized_volatility")
    plt.figure(figsize=(7.5, 5.2))
    plt.plot(valid["annualized_volatility"], valid["annualized_return"], marker="o", linewidth=1)
    plt.xlabel("Annualized volatility")
    plt.ylabel("Annualized expected return")
    plt.title("QP Model A2 Efficient Frontier")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def plot_qp_socp(qp_df: pd.DataFrame, socp_df: pd.DataFrame, out_file: Path) -> None:
    qp = qp_df[qp_df["status"].isin(["optimal", "optimal_inaccurate"])].sort_values("annualized_volatility")
    socp = socp_df[socp_df["status"].isin(["optimal", "optimal_inaccurate"])].sort_values("annualized_volatility")
    plt.figure(figsize=(7.5, 5.2))
    if not qp.empty:
        plt.plot(qp["annualized_volatility"], qp["annualized_return"], marker="o", linewidth=1, label="QP A2: gamma sweep")
    if not socp.empty:
        plt.scatter(socp["annualized_volatility"], socp["annualized_return"], marker="x", label="SOCP B: sigma sweep")
    plt.xlabel("Annualized volatility")
    plt.ylabel("Annualized expected return")
    plt.title("QP vs SOCP in the Risk-Return Plane")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def plot_weight_paths(weights_df: pd.DataFrame, tickers: list[str], parameter_name: str, out_file: Path, top_k: int = 10) -> None:
    df = weights_df.sort_values(parameter_name).copy()
    if df.empty:
        return
    W = df[tickers].to_numpy(dtype=float)
    max_weights = np.nanmax(W, axis=0)
    top_idx = np.argsort(max_weights)[::-1][: min(top_k, len(tickers))]
    selected = [tickers[i] for i in top_idx]
    plot_df = df[[parameter_name] + selected].copy()
    other = 1.0 - plot_df[selected].sum(axis=1)
    plot_df["Other"] = np.maximum(other, 0.0)

    x_values = plot_df[parameter_name].to_numpy(dtype=float)
    plt.figure(figsize=(8.0, 5.2))
    plt.stackplot(x_values, [plot_df[c].to_numpy(dtype=float) for c in selected + ["Other"]], labels=selected + ["Other"])
    plt.xscale("log" if parameter_name == "gamma" and np.all(x_values > 0) else "linear")
    plt.xlabel(parameter_name)
    plt.ylabel("Portfolio weight")
    plt.title(f"Portfolio Weights along {parameter_name} Sweep")
    plt.ylim(0, 1)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    tabdir = outdir / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    solver = args.solver.upper()
    if solver not in {"MOSEK", "GUROBI"}:
        raise ValueError("This baseline is intended for commercial solvers only. Use --solver MOSEK or --solver GUROBI.")
    installed = set(cp.installed_solvers())
    if solver not in installed:
        raise RuntimeError(
            f"Requested solver {solver} is not listed by cvxpy.installed_solvers(). Installed solvers: {sorted(installed)}. "
            f"Install/configure {solver} and its license, then rerun."
        )

    mu, Sigma_raw, tickers = load_estimates(args.input)
    Sigma = stabilize_covariance(Sigma_raw, eps=args.eps)

    # Save a small audit of the optimization input.
    eigvals = np.linalg.eigvalsh(0.5 * (Sigma_raw + Sigma_raw.T))
    input_audit = {
        "n_assets": len(mu),
        "mu_min": float(np.min(mu)),
        "mu_max": float(np.max(mu)),
        "mu_mean": float(np.mean(mu)),
        "raw_cov_min_eigenvalue": float(np.min(eigvals)),
        "raw_cov_max_eigenvalue": float(np.max(eigvals)),
        "covariance_eps_argument": float(args.eps),
        "solver": solver,
        "periods_per_year": float(args.periods_per_year),
    }
    (outdir / "input_audit.json").write_text(json.dumps(input_audit, indent=2), encoding="utf-8")

    # QP gamma sweep.
    gammas = parse_float_list(args.gammas)
    if gammas is None:
        gammas = choose_default_gammas(mu, Sigma, num=args.num_gammas)
    if len(gammas) < 20:
        print(f"WARNING: only {len(gammas)} gamma values were supplied. The project asks for at least 20 QP risk-return points.")

    qp_results: list[PortfolioResult] = []
    qp_weights: dict[float, np.ndarray] = {}
    for gamma in gammas:
        res, w = solve_qp_a2(
            mu,
            Sigma,
            gamma=float(gamma),
            solver=solver,
            verbose=args.verbose,
            periods_per_year=args.periods_per_year,
        )
        qp_results.append(res)
        qp_weights[float(gamma)] = w

    qp_df = dataframe_from_results(qp_results)
    qp_df.to_csv(tabdir / "qp_frontier.csv", index=False)
    qp_w_df = save_weights_matrix(qp_weights, tickers, tabdir / "qp_weights_by_gamma.csv", "gamma")
    representative_table(qp_df, qp_w_df, "gamma", tickers, tabdir / "qp_representative_portfolios.csv")
    plot_qp_frontier(qp_df, figdir / "qp_efficient_frontier.png")
    plot_weight_paths(qp_w_df, tickers, "gamma", figdir / "qp_weight_paths.png")

    # SOCP sigma sweep. By default, use feasible risk budgets based on min-variance risk and QP frontier range.
    sigma_values = parse_float_list(args.sigma_values)
    if sigma_values is None:
        min_risk, _, minvar_status = solve_min_variance(Sigma, solver=solver, verbose=args.verbose)
        qp_valid = qp_df[qp_df["status"].isin(["optimal", "optimal_inaccurate"])].copy()
        if not np.isfinite(min_risk):
            raise RuntimeError(f"Could not compute minimum-variance risk for SOCP grid. Status: {minvar_status}")
        if qp_valid.empty:
            upper_risk = float(np.max(np.sqrt(np.maximum(np.diag(Sigma), 0.0))))
        else:
            upper_risk = float(max(qp_valid["standard_deviation"].max(), min_risk * 1.05))
        sigma_values = np.linspace(min_risk * 1.0001, upper_risk, args.num_sigmas)

    socp_results: list[PortfolioResult] = []
    socp_weights: dict[float, np.ndarray] = {}
    for sigma_max in sigma_values:
        res, w = solve_socp_b(
            mu,
            Sigma,
            sigma_max=float(sigma_max),
            solver=solver,
            verbose=args.verbose,
            periods_per_year=args.periods_per_year,
        )
        socp_results.append(res)
        socp_weights[float(sigma_max)] = w

    socp_df = dataframe_from_results(socp_results)
    socp_df.to_csv(tabdir / "socp_frontier.csv", index=False)
    socp_w_df = save_weights_matrix(socp_weights, tickers, tabdir / "socp_weights_by_sigma.csv", "sigma_max")
    representative_table(socp_df, socp_w_df, "sigma_max", tickers, tabdir / "socp_representative_portfolios.csv")
    plot_qp_socp(qp_df, socp_df, figdir / "qp_vs_socp_risk_return.png")

    all_results = pd.concat([qp_df, socp_df], axis=0, ignore_index=True)
    all_results.to_csv(tabdir / "commercial_solver_all_runs.csv", index=False)

    # Console summary.
    print(f"Saved commercial-solver baseline outputs under: {outdir}")
    print(f"QP optimal runs:   {(qp_df['status'].isin(['optimal', 'optimal_inaccurate'])).sum()} / {len(qp_df)}")
    print(f"SOCP optimal runs: {(socp_df['status'].isin(['optimal', 'optimal_inaccurate'])).sum()} / {len(socp_df)}")
    print("Key files:")
    print(f"  {tabdir / 'qp_frontier.csv'}")
    print(f"  {tabdir / 'socp_frontier.csv'}")
    print(f"  {figdir / 'qp_efficient_frontier.png'}")
    print(f"  {figdir / 'qp_vs_socp_risk_return.png'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QP and SOCP commercial-solver baselines for portfolio optimization.")
    parser.add_argument("--input", default="data/estimates_train.npz", help="Path to estimates .npz or directory containing mu.csv and Sigma.csv.")
    parser.add_argument("--outdir", default="results/commercial_baselines/", help="Output directory for tables and figures.")
    parser.add_argument("--solver", default="MOSEK", choices=["MOSEK", "GUROBI", "mosek", "gurobi"], help="Commercial solver to call through CVXPY.")
    parser.add_argument("--eps", type=float, default=1e-8, help="PSD stabilization shift for covariance matrix.")
    parser.add_argument("--periods-per-year", type=float, default=252.0, help="Use 252 for daily estimates; set to 1 if mu/Sigma are already annualized.")
    parser.add_argument("--num-gammas", type=int, default=30, help="Number of gamma values if --gammas is not supplied. Must be at least 20 for the report.")
    parser.add_argument("--gammas", default=None, help="Comma-separated gamma values, e.g. '0.1,0.3,1,3,10'. Overrides --num-gammas.")
    parser.add_argument("--num-sigmas", type=int, default=12, help="Number of sigma_max values if --sigma-values is not supplied.")
    parser.add_argument("--sigma-values", default=None, help="Comma-separated sigma_max values in the same frequency as Sigma. Overrides --num-sigmas.")
    parser.add_argument("--verbose", action="store_true", help="Print solver logs.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
