# Commercial Solver Baselines

This folder contains a ready-to-run implementation of the required commercial-solver baseline section.
It assumes you have already estimated `mu_hat` and `Sigma_hat`.

## 1. Save your estimates

Preferred format:

```python
import numpy as np
np.savez(
    "results/estimates_train.npz",
    mu=mu_hat,                 # shape: (n,)
    Sigma=Sigma_hat,           # shape: (n, n)
    tickers=np.array(tickers)  # optional, shape: (n,)
)
```

Alternative CSV format:

```text
estimates/
├── mu.csv
└── Sigma.csv
```

`mu.csv` can be either a one-column file with tickers as index, or a one-row file with tickers as columns. `Sigma.csv` should be a square covariance matrix with tickers as rows/columns.

## 2. Install dependencies

```bash
pip install numpy pandas matplotlib cvxpy
pip install Mosek
```

You also need a valid MOSEK license. GUROBI can be used instead by setting `--solver GUROBI` if it is installed and licensed.

## 3. Run

For daily returns and daily covariance estimates:

```bash
python commercial_solver_baselines.py \
  --input results/estimates_train.npz \
  --outdir results/commercial_baselines \
  --solver MOSEK \
  --periods-per-year 252
```

If your `mu` and `Sigma` are already annualized, use:

```bash
python commercial_solver_baselines.py \
  --input results/estimates_train.npz \
  --outdir results/commercial_baselines \
  --solver MOSEK \
  --periods-per-year 1
```

## 4. Generated outputs

```text
results/commercial_baselines/
├── input_audit.json
├── figures/
│   ├── qp_efficient_frontier.png
│   ├── qp_weight_paths.png
│   └── qp_vs_socp_risk_return.png
└── tables/
    ├── qp_frontier.csv
    ├── qp_weights_by_gamma.csv
    ├── qp_representative_portfolios.csv
    ├── socp_frontier.csv
    ├── socp_weights_by_sigma.csv
    ├── socp_representative_portfolios.csv
    └── commercial_solver_all_runs.csv
```

These files directly correspond to the report requirements: QP efficient frontier, SOCP risk-return points, QP-vs-SOCP comparison, representative portfolio metrics, solver status, running time, and weights.
