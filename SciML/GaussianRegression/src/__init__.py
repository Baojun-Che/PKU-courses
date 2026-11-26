"""Public interfaces for Gaussian-process data, fitting, and prediction."""

from .generate_data import generate_data
from .kernel import rbf_kernel, matern_kernel
from .gp_para_opt import (
    DEFAULT_BOUNDS,
    log_likelihood,
    optimize_gp_parameters,
    optimize_gp_parameters_multistart,
    plot_log_likelihood_contours,
)
from .gp_regression import gp_regression, gp_regression_plot

__all__ = [
    "generate_data",
    "rbf_kernel",
    "matern_kernel",
    "DEFAULT_BOUNDS",
    "log_likelihood",
    "optimize_gp_parameters",
    "optimize_gp_parameters_multistart",
    "plot_log_likelihood_contours",
    "gp_regression",
    "gp_regression_plot",
]
