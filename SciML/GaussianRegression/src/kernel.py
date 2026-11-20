import numpy as np
from scipy.spatial.distance import cdist

def rbf_kernel(x:np.ndarray, length_scale: float = 1.0, sigma_f: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    #  sq_dist: disctance matrix of x

    sq_dist = cdist(x, x, 'sqeuclidean')
    
    cov_matrix = sigma_f**2 * np.exp(-0.5 * sq_dist / length_scale**2) + sigma**2 * np.eye(len(x))
    
    return cov_matrix


def matern_kernel(x:np.ndarray, length_scale: float = 1.0, sigma_f: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    sq_dist = cdist(x, x, 'sqeuclidean')
    rel_dist = (np.sqrt(3) / length_scale) * np.sqrt(sq_dist)
    cov_matrix = sigma_f**2 * (1 + rel_dist) * np.exp(-rel_dist) + sigma**2 * np.eye(len(x))
    
    return cov_matrix