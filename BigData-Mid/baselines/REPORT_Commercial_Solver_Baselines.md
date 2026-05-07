# 4. Commercial Solver Baselines

After estimating the expected return vector \(\hat\mu\) and covariance matrix \(\hat\Sigma\) from the training sample, I use commercial solvers to compute two benchmark portfolio models. The portfolio decision variable is \(x\in\mathbb{R}^n\), where \(x_i\) is the capital weight invested in asset \(i\). Throughout this section, the portfolio is long-only and fully invested:

$$
\mathbf{1}^\top x = 1,\qquad x\ge 0.
$$

To improve numerical stability, I use the symmetrized covariance matrix and, when necessary, add a small diagonal shift,

$$
\hat\Sigma_\varepsilon = \frac{1}{2}(\hat\Sigma+\hat\Sigma^\top)+\varepsilon I,
$$

where \(\varepsilon=10^{-8}\) in the baseline experiments. All optimization problems in this section are implemented in CVXPY and solved with MOSEK. GUROBI gives an equivalent commercial-solver interface for the QP baseline, but MOSEK is used as the default solver because it directly supports both quadratic and conic formulations.

## 4.1 QP baseline: Model A2

The first benchmark is the quadratic utility portfolio model:

$$
\begin{aligned}
\min_{x\in\mathbb{R}^n}\quad & \frac{\gamma}{2}x^\top \hat\Sigma_\varepsilon x - \hat\mu^\top x \\
\text{s.t.}\quad & \mathbf{1}^\top x = 1,\\
& x\ge 0.
\end{aligned}
$$

Here \(\gamma>0\) is the risk-aversion parameter. A larger \(\gamma\) penalizes variance more heavily and therefore tends to produce lower-risk, more diversified portfolios. A smaller \(\gamma\) puts more emphasis on estimated expected return and can lead to more concentrated portfolios.

To generate the efficient frontier, I solve the model for 30 values of \(\gamma\). For each optimal solution \(x^\star(\gamma)\), I report

$$
\text{expected return}=\hat\mu^\top x^\star,
\qquad
\text{variance}=(x^\star)^\top\hat\Sigma_\varepsilon x^\star,
\qquad
\text{standard deviation}=\sqrt{(x^\star)^\top\hat\Sigma_\varepsilon x^\star}.
$$

The complete frontier results are saved in `qp_frontier.csv`. Representative low-risk, middle-risk, and high-risk portfolios are reported in `qp_representative_portfolios.csv`, including expected return, standard deviation, objective value, solver status, running time, maximum portfolio weight, the number of active positions above 1%, and the largest holdings.

**Insert Figure:** `qp_efficient_frontier.png`.

**Insert Table:** `qp_representative_portfolios.csv`.

## 4.2 SOCP baseline: Model B

The second benchmark is the conic portfolio model that maximizes expected return subject to a standard-deviation budget:

$$
\begin{aligned}
\max_{x\in\mathbb{R}^n}\quad & \hat\mu^\top x \\
\text{s.t.}\quad & \|B^\top x\|_2 \le \sigma_{\max},\\
& \mathbf{1}^\top x = 1,\\
& x\ge 0,
\end{aligned}
$$

where \(B\) satisfies \(\hat\Sigma_\varepsilon = BB^\top\). In the implementation, I first attempt a Cholesky factorization. If the covariance matrix is numerically only positive semidefinite, I use an eigenvalue decomposition and truncate small negative eigenvalues to zero.

The parameter \(\sigma_{\max}\) directly controls the feasible risk budget. I choose several values between the long-only minimum-volatility portfolio risk and the upper risk level observed on the QP frontier. For each \(\sigma_{\max}\), the solver either returns the highest-return feasible portfolio or reports infeasibility. The SOCP results are saved in `socp_frontier.csv`, and representative portfolios are saved in `socp_representative_portfolios.csv`.

**Insert Figure:** `qp_vs_socp_risk_return.png`.

**Insert Table:** `socp_representative_portfolios.csv`.

## 4.3 QP and SOCP comparison

The QP and SOCP formulations encode the same economic trade-off between expected return and risk, but they parameterize it differently. Model A2 combines expected return and variance in a single scalar objective. The risk-aversion coefficient \(\gamma\) determines the marginal penalty on variance. Model B instead imposes a hard upper bound on standard deviation and maximizes expected return within that risk budget.

Computationally, Model A2 is a convex quadratic program with a quadratic objective and linear constraints. Model B is a second-order cone program because the risk constraint has the form \(\|B^\top x\|_2\le \sigma_{\max}\). In the numerical experiments, both models are solved by the same commercial solver interface, so their results are directly comparable. The QP sweep produces a smooth efficient frontier as \(\gamma\) varies, while the SOCP sweep produces risk-return points indexed by the explicit risk budget \(\sigma_{\max}\). If the risk budgets are chosen to cover the same risk range, the SOCP points should lie close to the upper envelope of the QP frontier. Differences can arise from numerical tolerances, the finite parameter grids, and the fact that the QP objective penalizes variance while the SOCP constraint controls standard deviation.

The comparison table `commercial_solver_all_runs.csv` records the final objective value, expected return, variance, standard deviation, solver status, and running time for all QP and SOCP runs. This table is used as the commercial-solver baseline for the later ADMM and PDHG comparisons.
