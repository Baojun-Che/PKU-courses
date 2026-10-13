"""Projected augmented DMD with an exactly preserved constant coordinate."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig, lstsq

from ._linalg import SnapshotBasis, snapshot_basis
from .fvm import Trajectory, validate_times


@dataclass
class DMDModel:
    basis: SnapshotBasis
    operator: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    modes: np.ndarray
    initial_state: np.ndarray
    t0: float
    snapshot_interval: float


def fit_dmd(snapshots, times, rank=10) -> DMDModel:
    """Fit [z_next;1] = K [z;1], with z = V.T @ c.

    The least-squares problem fits only K's first r rows. Its last row
    is fixed to [0,...,0,1], retaining the affine boundary forcing even
    after spatial truncation. This is projected, constrained DMD rather
    than unconstrained exact DMD of a jointly truncated [c;1] matrix.
    """
    times = validate_times(times)
    delta = times[1] - times[0]
    if not np.allclose(np.diff(times), delta, rtol=1e-9, atol=1e-12 * delta):
        raise ValueError("DMD training times must be uniformly spaced")
    basis = snapshot_basis(snapshots, rank)
    c = np.asarray(snapshots, dtype=float)
    if c.shape[1] != times.size:
        raise ValueError("times length must match the number of snapshot columns")
    z = basis.vectors.T @ c
    augmented = np.vstack([z[:, :-1], np.ones(times.size - 1)])
    # Solve X_aug.T @ coefficients.T ~= Z_next.T using SVD least squares.
    # LAPACK gelsd cannot solve a least-squares system with zero RHS columns.
    coefficients = (lstsq(augmented.T, z[:, 1:].T, lapack_driver="gelsd")[0].T
                    if basis.rank else np.empty((0, 1)))
    operator = np.zeros((basis.rank + 1, basis.rank + 1))
    operator[:-1] = coefficients
    operator[-1, -1] = 1.0
    eigenvalues, eigenvectors = eig(operator)
    modes = basis.vectors @ eigenvectors[:-1]
    return DMDModel(basis, operator, eigenvalues, eigenvectors, modes, c[:, 0].copy(),
                    float(times[0]), float(delta))


def predict_dmd(model: DMDModel, times, initial=None) -> Trajectory:
    """Predict at nonnegative integer multiples of the training interval.

    Matrix powers avoid requiring an invertible eigenvector matrix or
    choosing a complex-log branch. The optional initial state is at t0.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("prediction times must be a finite nonempty vector")
    fractional_steps = (times - model.t0) / model.snapshot_interval
    if np.any(fractional_steps < -1e-9) or not np.allclose(fractional_steps, np.rint(fractional_steps),
                                                       rtol=0, atol=1e-8):
        raise ValueError("prediction times must be t0 plus nonnegative integer snapshot intervals")
    c0 = model.initial_state if initial is None else np.asarray(initial, dtype=float)
    if c0.shape != model.initial_state.shape or not np.all(np.isfinite(c0)):
        raise ValueError("initial must be a finite vector of length ndof")
    augmented_initial = np.append(model.basis.vectors.T @ c0, 1.0)
    steps = np.rint(fractional_steps).astype(int)
    coordinates = np.column_stack([np.linalg.matrix_power(model.operator, int(k)) @ augmented_initial
                                    for k in steps])
    return Trajectory(times.copy(), model.basis.vectors @ coordinates[:-1])
