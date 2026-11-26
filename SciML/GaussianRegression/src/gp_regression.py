import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import cho_solve, solve_triangular

if __package__:
    from .kernel import rbf_kernel, matern_kernel
else:
    from kernel import rbf_kernel, matern_kernel

def gp_regression(x_test: np.ndarray,
                    x_data:np.ndarray,
                    y_data:np.ndarray, 
                    gp_para: np.ndarray, 
                    kernel_type: str) -> tuple:
    
    """Return the conditional GP mean (n_test,) and covariance (n_test, n_test).

    Uses the full kernel, including its white-noise term, at test points.
    """
    y_data = np.asarray(y_data, dtype=float)
    if y_data.ndim == 2 and y_data.shape[1] == 1:
        y_data = y_data[:, 0]
    if y_data.ndim != 1 or y_data.size != len(x_data):
        raise ValueError("y_data must have shape (n,) or (n, 1), matching x_data.")

    # Compute the covariance matrices.
    if kernel_type == 'rbf':
        K = rbf_kernel(x_data, length_scale=gp_para[0], sigma_f=gp_para[1], sigma=gp_para[2])
        K_s = rbf_kernel(x_data, length_scale=gp_para[0], sigma_f=gp_para[1], sigma=gp_para[2], x2=x_test)
        K_ss = rbf_kernel(x_test, length_scale=gp_para[0], sigma_f=gp_para[1], sigma=gp_para[2])
    elif kernel_type == 'matern':
        K = matern_kernel(x_data, length_scale=gp_para[0], sigma_f=gp_para[1], sigma=gp_para[2])
        K_s = matern_kernel(x_data, length_scale=gp_para[0], sigma_f=gp_para[1], sigma=gp_para[2], x2=x_test)
        K_ss = matern_kernel(x_test, length_scale=gp_para[0], sigma_f=gp_para[1], sigma=gp_para[2])
    else:
        raise ValueError("Invalid kernel type. Choose 'rbf' or 'matern'.")

    L = np.linalg.cholesky(K + 1e-10 * np.eye(len(x_data)))
    alpha = cho_solve((L, True), y_data)
    
    # Predictive mean
    posterior_mean = K_s.T @ alpha
    v = solve_triangular(L, K_s, lower=True)
    posterior_cov = K_ss - v.T @ v
    posterior_cov = (posterior_cov + posterior_cov.T) / 2
    return posterior_mean, posterior_cov

def gp_regression_plot(x_test: np.ndarray,
                        x_data:np.ndarray,
                        y_data:np.ndarray,
                        gp_para: np.ndarray, 
                        kernel_type: str,
                        fig_name: str,
                        fig_path: str = None,
                        ) -> Path:
    """Save mean and +/-2 std. to fig_path (default GaussianRegression/result)."""
    posterior_mean, posterior_cov = gp_regression(x_test, x_data, y_data, gp_para, kernel_type)
    posterior_std = np.sqrt(np.maximum(np.diag(posterior_cov), 0))

    # 画均值和两倍标准差
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        ax.plot(np.asarray(x_data).ravel(), np.asarray(y_data).ravel(), 'ro', label='Training Data')
        # Sort the grid so lines and the uncertainty band do not zigzag.
        x = np.asarray(x_test).ravel()
        order = np.argsort(x)
        x, mean, std = x[order], posterior_mean[order], posterior_std[order]
        ax.plot(x, mean, 'b-', label='Posterior Mean')
        ax.fill_between(x, mean - 2 * std, mean + 2 * std,
                        alpha=0.3, label='Mean +/- 2 std.')
        ax.set(xlabel='x', ylabel='f(x)')
        ax.legend()
        directory = (Path(__file__).resolve().parent.parent / 'result'
                     if fig_path is None else Path(fig_path))
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / fig_name
        fig.savefig(output_path, bbox_inches='tight')
    finally:
        plt.close(fig)
    return output_path
