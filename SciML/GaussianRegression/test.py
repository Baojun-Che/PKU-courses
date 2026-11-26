from pathlib import Path
import numpy as np
from src import (
    generate_data,
    gp_regression,
    gp_regression_plot,
    optimize_gp_parameters_multistart,
    plot_log_likelihood_contours,
)

def load_data() -> tuple:
    data_dir = Path(__file__).resolve().parent / "data"
    
    # Load the input features and output targets from .npz files
    X = np.load(data_dir / "x.npz")['X']
    y = np.load(data_dir / "fx.npz")['y']
    
    return X, y

if __name__ == "__main__":

    generate_data()
    X, y = load_data()
    x_test = np.linspace(-8, 8, 100).reshape(-1, 1)
    gp_regression_plot(x_test, X, y, np.array([1.0, 1.0, 0.1]), 'rbf', 'Q1-p1.png')
    gp_regression_plot(x_test, X, y, np.array([0.3, 1.08, 5e-5]), 'rbf', 'Q1-p2.png')
    gp_regression_plot(x_test, X, y, np.array([3.0, 1.16, 0.89]), 'rbf', 'Q1-p3.png')

    optimized_gp_para = optimize_gp_parameters_multistart(X, y, 'rbf', n_starts=10)
    print("Optimized GP parameters for RBF kernel (length_scale, sigma_f, sigma):", optimized_gp_para)
    plot_log_likelihood_contours(X, y, optimized_gp_para, 'rbf')
    gp_regression_plot(x_test, X, y, optimized_gp_para, 'rbf', 'Q2.png')

    optimized_matern_para = optimize_gp_parameters_multistart(X, y, 'matern', n_starts=10)
    print("Optimized GP parameters for Matérn kernel (length_scale, sigma_f, sigma):", optimized_matern_para)
    plot_log_likelihood_contours(X, y, optimized_matern_para, 'matern')
    gp_regression_plot(x_test, X, y, optimized_matern_para, 'matern', 'Q3.png')
    
