"""Shared snapshot SVD and numerical-rank handling."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd


@dataclass
class SnapshotBasis:
    vectors: np.ndarray
    singular_values: np.ndarray
    energy: np.ndarray
    requested_rank: int
    numerical_rank: int

    @property
    def rank(self):
        return self.vectors.shape[1]


def snapshot_basis(snapshots, rank) -> SnapshotBasis:
    c = np.asarray(snapshots, dtype=float)
    if c.ndim != 2 or min(c.shape) == 0 or not np.all(np.isfinite(c)):
        raise ValueError("snapshots must be a finite nonempty (ndof, nsnapshots) matrix")
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    u, s, _ = svd(c, full_matrices=False, check_finite=False)
    tolerance = np.finfo(float).eps * max(c.shape) * s[0]
    numerical_rank = int(np.count_nonzero(s > tolerance))
    effective_rank = min(rank, numerical_rank)
    # Scaling avoids overflow in squared singular values for large data.
    weights = (s / s[0]) ** 2 if s[0] > 0 else np.zeros_like(s)
    energy = np.cumsum(weights) / np.sum(weights) if np.sum(weights) > 0 else weights
    return SnapshotBasis(u[:, :effective_rank], s, energy, int(rank), numerical_rank)
