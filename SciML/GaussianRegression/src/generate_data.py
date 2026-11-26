import numpy as np
if __package__:
    from .kernel import rbf_kernel, matern_kernel
else:
    from kernel import rbf_kernel, matern_kernel
from pathlib import Path

def generate_data(n_samples: int = 20,
        sample_range: tuple = (-8, 8),
        gp_para: np.ndarray = np.array([1.0, 1.0, 0.1]),
        kernel_type: str = 'rbf',
        random_state: int = 20030107,
        ) -> tuple:
    """
    Generate synthetic data using Gaussian Process regression.

    Parameters:
    - n_samples: Number of samples to generate.
    - gp_para: Parameters for the Gaussian Process (length_scale, sigma_f, sigma).
    - kernel_type: Type of kernel to use ('rbf' or 'matern').
    - random_state: Seed for a local NumPy generator; None gives fresh samples.

    Returns:
    - X: Input features (n_samples x 1).
    - y: Output targets (n_samples,).
    """
    
    # Generate random input features
    rng = np.random.default_rng(random_state)
    X = rng.uniform(sample_range[0], sample_range[1], size=(n_samples, 1))
    
    # Compute the covariance matrix using the specified kernel
    if kernel_type == 'rbf':
        cov_matrix = rbf_kernel(X, length_scale=gp_para[0], sigma_f=gp_para[1], sigma=gp_para[2])
    elif kernel_type == 'matern':
        cov_matrix = matern_kernel(X, length_scale=gp_para[0], sigma_f=gp_para[1], sigma=gp_para[2])
    else:
        raise ValueError("Invalid kernel type. Choose 'rbf' or 'matern'.")
    
    # Sample from the multivariate normal distribution
    y = rng.multivariate_normal(mean=np.zeros(n_samples), cov=cov_matrix)

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez(data_dir / "x.npz", X=X)
    np.savez(data_dir / "fx.npz", y=y)
    return X, y
