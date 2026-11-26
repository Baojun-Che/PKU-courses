from pathlib import Path
import numpy as np
if __package__:
    from .kernel import rbf_kernel, matern_kernel
else:
    from kernel import rbf_kernel, matern_kernel
from scipy.linalg import cho_solve
from scipy.optimize import minimize


# Bounds in the original coordinates: (length_scale, sigma_f, sigma).
DEFAULT_BOUNDS = np.array([[1e-2, 1e2], [1e-3, 1e2], [1e-6, 1e1]])


def log_likelihood(X: np.ndarray, y: np.ndarray, gp_para: np.ndarray, kernel_type: str) -> float:
    """Log marginal likelihood, omitting -n/2 * log(2*pi).

    With p(theta) = 1 this is also the log posterior up to a constant.
    Accepts y with shape (n,) or (n, 1).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or y.size != len(X):
        raise ValueError("y must have shape (n,) or (n, 1), matching X.")

    if kernel_type == 'rbf':
        cov_matrix = rbf_kernel(X, length_scale=gp_para[0], sigma_f=gp_para[1], sigma=gp_para[2])
    elif kernel_type == 'matern':
        cov_matrix = matern_kernel(X, length_scale=gp_para[0], sigma_f=gp_para[1], sigma=gp_para[2])
    else:
        raise ValueError("Invalid kernel type. Choose 'rbf' or 'matern'.")

    # Jitter is a numerical stabilizer, separate from the modeled noise sigma**2.
    cov_matrix = cov_matrix + 1e-10 * np.eye(len(X))
    L = np.linalg.cholesky(cov_matrix)
    alpha = cho_solve((L, True), y)
    # log(det(K)) = 2 * sum(log(diag(L))).
    return float(-0.5 * (y @ alpha) - np.log(np.diag(L)).sum())

def optimize_gp_parameters(X: np.ndarray, y: np.ndarray,
        kernel_type: str, initial_gp_para: np.ndarray=np.array([1.0, 1.0, 0.1]),
        bounds: np.ndarray = DEFAULT_BOUNDS) -> np.ndarray:
    """
    Optimize in log coordinates using L-BFGS-B, with p(theta) = 1.

    Parameters:
    - X: Input features (n_samples x 1).
    - y: Output targets (n_samples x 1).
    - initial_gp_para: Initial guess for the GP parameters (length_scale, sigma_f, sigma).
    - kernel_type: Type of kernel to use ('rbf' or 'matern').
    - bounds: Three finite, positive (lower, upper) pairs in original coordinates.

    Returns:
    - optimized_gp_para: Optimized GP parameters.
    """

    bounds = _validate_bounds(bounds)
    initial_gp_para = np.asarray(initial_gp_para, dtype=float)
    if (initial_gp_para.shape != (3,) or not np.all(np.isfinite(initial_gp_para))
            or np.any(initial_gp_para < bounds[:, 0])
            or np.any(initial_gp_para > bounds[:, 1])):
        raise ValueError("initial_gp_para must contain three parameters within bounds.")

    def objective(z):
        # Changing optimization coordinates does not change the posterior density.
        # In particular, do not add a prior or a Jacobian term.
        return -log_likelihood(X, y, np.exp(z), kernel_type)

    # Perform optimization
    result = minimize(
        objective,
        x0=np.log(initial_gp_para),
        method="L-BFGS-B",
        bounds=np.log(bounds),
        options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-6, "maxls": 50},
    )

    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Optimization failed: " + result.message)

    # Roundoff in exp(log(upper)) can otherwise place a boundary solution
    # just outside the physical bounds, preventing it from being plotted.
    return np.clip(np.exp(result.x), bounds[:, 0], bounds[:, 1])


def _validate_bounds(bounds: np.ndarray) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    if (bounds.shape != (3, 2) or not np.all(np.isfinite(bounds))
            or np.any(bounds <= 0) or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("bounds must be three finite, positive (lower, upper) pairs.")
    return bounds


def optimize_gp_parameters_multistart(
        X: np.ndarray, y: np.ndarray, kernel_type: str, n_starts: int = 10,
        initial_gp_para: np.ndarray = np.array([1.0, 1.0, 0.1]),
        bounds: np.ndarray = DEFAULT_BOUNDS,
        random_state: int = None) -> np.ndarray:
    """Return the highest-likelihood converged result from n_starts attempts.

    The first start is initial_gp_para; the remaining starts are sampled
    uniformly in log coordinates. This sampling is not a hyperparameter prior.
    Supply random_state for reproducibility. Failed runs are skipped; raise
    RuntimeError if all runs fail. The return value is (l, sigma_f, sigma),
    in original coordinates, not their logarithms.
    """
    if not isinstance(n_starts, (int, np.integer)) or n_starts < 1:
        raise ValueError("n_starts must be a positive integer.")
    bounds = _validate_bounds(bounds)
    rng = np.random.default_rng(random_state)
    starts = [initial_gp_para]
    log_bounds = np.log(bounds)
    starts.extend(np.exp(rng.uniform(log_bounds[:, 0], log_bounds[:, 1],
                                     size=(n_starts - 1, 3))))

    best_parameters = None
    best_value = -np.inf
    failures = []
    for initial in starts:
        try:
            parameters = optimize_gp_parameters(X, y, kernel_type, initial, bounds)
            value = log_likelihood(X, y, parameters, kernel_type)
        except (RuntimeError, np.linalg.LinAlgError) as error:
            failures.append(str(error))
            continue
        if np.isfinite(value) and value > best_value:
            best_parameters, best_value = parameters, value

    if best_parameters is None:
        detail = failures[-1] if failures else "No finite likelihood found."
        raise RuntimeError(f"All {n_starts} optimization attempts failed. {detail}")
    return best_parameters


def plot_log_likelihood_contours(
        X: np.ndarray, y: np.ndarray, optimized_gp_para: np.ndarray,
        kernel_type: str, grid_size: int = 100,
        bounds: np.ndarray = DEFAULT_BOUNDS, output_dir: Path = None) -> Path:
    """Save three 2D log-posterior slices, fixing each parameter at its optimum.

    Uses logarithmic axes over bounds (in original coordinates), and marks the
    supplied optimum with a red star. Since p(theta)=1, the plotted quantity
    is log_likelihood, i.e. log posterior up to a constant. Colors/contours
    focus on the top 30 log units of each slice to keep peaks visible.
    Defaults to GaussianRegression/result/<kernel>_log_likelihood_contours.png,
    independent of the working directory. Returns the saved absolute path.
    """
    import matplotlib.pyplot as plt

    if not isinstance(grid_size, (int, np.integer)) or grid_size < 2:
        raise ValueError("grid_size must be an integer >= 2.")
    bounds = _validate_bounds(bounds)
    parameters = np.asarray(optimized_gp_para, dtype=float)
    if (parameters.shape != (3,) or not np.all(np.isfinite(parameters))
            or np.any(parameters < bounds[:, 0]) or np.any(parameters > bounds[:, 1])):
        raise ValueError("optimized_gp_para must contain three parameters within bounds.")

    grids = [np.unique(np.append(np.geomspace(*bounds[i], grid_size), parameters[i]))
             for i in range(3)]
    labels = [r"$l$", r"$\sigma_f$", r"$\sigma$"]
    names = ["length scale", "signal std.", "noise std."]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    try:
        for ax, fixed in zip(axes, range(3)):
            a, b = [i for i in range(3) if i != fixed]
            horizontal, vertical = np.meshgrid(grids[a], grids[b])
            values = np.full(horizontal.shape, np.nan)
            for row, col in np.ndindex(values.shape):
                point = parameters.copy()
                point[a], point[b] = horizontal[row, col], vertical[row, col]
                try:
                    values[row, col] = log_likelihood(X, y, point, kernel_type)
                except np.linalg.LinAlgError:
                    pass  # Mask numerically non-positive-definite grid points.

            if not np.any(np.isfinite(values)):
                raise RuntimeError("No finite likelihood values in the contour slice.")
            peak = np.nanmax(values)
            levels = np.linspace(peak - 30, peak, 21)
            surface = np.ma.masked_invalid(values)
            filled = ax.contourf(horizontal, vertical, surface, levels=levels,
                                 cmap="viridis", extend="min")
            ax.contour(horizontal, vertical, surface, levels=levels[::4],
                       colors="white", linewidths=0.5, alpha=0.6)
            ax.scatter(parameters[a], parameters[b], marker="*", s=150,
                       c="red", edgecolors="white", linewidths=0.7,
                       label="Best found", zorder=5)
            ax.set(xscale="log", yscale="log", xlabel=labels[a], ylabel=labels[b],
                   title=f"Fixed {names[fixed]} {labels[fixed]} = {parameters[fixed]:.4g}")
            ax.legend(loc="upper right", fontsize=8)
            fig.colorbar(filled, ax=ax, label="Log likelihood (constant omitted)")

        directory = (Path(__file__).resolve().parent.parent / "result"
                     if output_dir is None else Path(output_dir).resolve())
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"{kernel_type}_log_likelihood_contours.png"
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)
    return output_path
