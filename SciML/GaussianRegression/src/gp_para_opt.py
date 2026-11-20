from pathlib import Path
import numpy as np
from kernel import rbf_kernel, matern_kernel
from scipy.optimize import minimize

def log_likelihood(X: np.ndarray, y: np.ndarray, gp_para: np.ndarray, kernel_type: str) -> float:
    if kernel_type == 'rbf':
        cov_matrix = rbf_kernel(X, length_scale=gp_para[0], sigma_f=gp_para[1], sigma=gp_para[2])
    elif kernel_type == 'matern':
        cov_matrix = matern_kernel(X, length_scale=gp_para[0], sigma_f=gp_para[1], sigma=gp_para[2])
    else:
        raise ValueError("Invalid kernel type. Choose 'rbf' or 'matern'.")

    # Compute the log-likelihood, ignore contant terms
    log_likelihood = - 0.5 * np.dot(y.T, np.linalg.solve(cov_matrix, y))[0, 0] - 0.5 * np.log(np.linalg.det(cov_matrix))

    return log_likelihood

def load_data() -> tuple:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    
    # Load the input features and output targets from .npz files
    X = np.load(data_dir / "x.npz")['X']
    y = np.load(data_dir / "fx.npz")['y']
    
    return X, y

def optimize_gp_parameters(X: np.ndarray, y: np.ndarray,
        kernel_type: str, initial_gp_para: np.ndarray=np.array([1.0, 1.0, 0.1]) )-> np.ndarray:
    """
    Optimize the Gaussian Process parameters using maximum likelihood estimation.

    Parameters:
    - X: Input features (n_samples x 1).
    - y: Output targets (n_samples x 1).
    - initial_gp_para: Initial guess for the GP parameters (length_scale, sigma_f, sigma).
    - kernel_type: Type of kernel to use ('rbf' or 'matern').

    Returns:
    - optimized_gp_para: Optimized GP parameters.
    """

    # Define the objective function to minimize (negative log-likelihood)
    def objective(gp_para):
        return -log_likelihood(X, y, gp_para, kernel_type)


    ####### The setting for bounds and optimizer is suggested by GPT.
    #################################  
    bounds = np.log([
    [1e-2, 1e2],   # length_scale
    [1e-3, 1e2],   # sigma_f
    [1e-6, 1e1],   # sigma
    ])

    # Perform optimization
    result = minimize(
    objective,
    x0=np.log(initial_gp_para),
    method="L-BFGS-B",
    bounds=bounds,
    options={
        "maxiter": 1000,
        "ftol": 1e-9,
        "gtol": 1e-6,
        "maxls": 50,
    },
    )

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    optimized_gp_para = result.x
    #################################
    return optimized_gp_para