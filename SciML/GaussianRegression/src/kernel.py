import numpy as np
from scipy.spatial.distance import cdist

def rbf_kernel(x:np.ndarray, length_scale: float = 1.0, sigma_f: float = 1.0, sigma: float = 1.0, x2: np.ndarray = None) -> np.ndarray:
    #  sq_dist: disctance matrix of x

    if x2 is None:
        sq_dist = cdist(x, x, 'sqeuclidean')
    else:
        sq_dist = cdist(x, x2, 'sqeuclidean')

    K1 = sigma_f**2 * np.exp(-0.5 * sq_dist / length_scale**2)
    # The Kronecker delta is one only for exactly equal input points.
    K2 = sigma**2 * (sq_dist == 0)
    return K1 + K2


def matern_kernel(x:np.ndarray, length_scale: float = 1.0, sigma_f: float = 1.0, sigma: float = 1.0, x2: np.ndarray = None) -> np.ndarray:
    if x2 is None:
        sq_dist = cdist(x, x, 'sqeuclidean')
    else:
        sq_dist = cdist(x, x2, 'sqeuclidean')

    rel_dist = (np.sqrt(3) / length_scale) * np.sqrt(sq_dist)
    K1 = sigma_f**2 * (1 + rel_dist) * np.exp(-rel_dist)
    K2 = sigma**2 * (sq_dist == 0)
    return K1 + K2
