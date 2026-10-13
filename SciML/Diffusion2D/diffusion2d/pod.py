"""Uncentred POD with Galerkin projection of the affine FVM system."""
from dataclasses import dataclass

import numpy as np

from ._linalg import SnapshotBasis, snapshot_basis
from .fvm import DiscreteSystem, Trajectory, integrate_euler


@dataclass
class PODModel:
    basis: SnapshotBasis
    A: np.ndarray
    f: np.ndarray


def fit_pod(system: DiscreteSystem, snapshots, rank=10) -> PODModel:
    basis = snapshot_basis(snapshots, rank)
    v = basis.vectors
    if v.shape[0] != system.problem.ndof:
        raise ValueError("snapshot spatial dimension must match the FVM system")
    return PODModel(basis, v.T @ (system.A @ v), v.T @ system.f)


def solve_pod(model: PODModel, times, dt, initial=None) -> Trajectory:
    """Project initial data, integrate coefficients, and reconstruct c.

    Supply the full-model dt for the default like-for-like comparison.
    A reduced model is not automatically stable at a larger Euler step.
    """
    v = model.basis.vectors
    c0 = np.zeros(v.shape[0]) if initial is None else np.asarray(initial, dtype=float)
    if c0.shape != (v.shape[0],) or not np.all(np.isfinite(c0)):
        raise ValueError("initial must be a finite vector of length ndof")
    reduced = integrate_euler(model.A, model.f, v.T @ c0, times, dt)
    return Trajectory(reduced.times, v @ reduced.snapshots, reduced.step_sizes, reduced.step_counts)
