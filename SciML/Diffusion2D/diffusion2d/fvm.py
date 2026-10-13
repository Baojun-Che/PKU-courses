"""Conservative upwind advection and central diffusion, explicit Euler."""
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .problem import Problem


@dataclass
class DiscreteSystem:
    problem: Problem
    A: sparse.csr_matrix
    f: np.ndarray
    inlet: np.ndarray

    @property
    def dt_limit(self):
        """Largest Euler step with nonnegative update coefficients."""
        return 1.0 / np.max(-self.A.diagonal())


@dataclass
class Trajectory:
    times: np.ndarray
    snapshots: np.ndarray
    step_sizes: np.ndarray = field(default_factory=lambda: np.empty(0))
    step_counts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    @property
    def dt(self):
        """Largest actual integration step (zero for a DMD prediction)."""
        return float(np.max(self.step_sizes)) if self.step_sizes.size else 0.0


def assemble_system(problem: Problem, inlet_values=None) -> DiscreteSystem:
    """Assemble dc/dt = A c + f without introducing boundary unknowns.

    Optional inlet values permit physical checks with zero/uniform inflow.
    Periodic edges are accumulated in COO, including the ny=2 case where
    both y-neighbours coincide.
    """
    p = problem
    inlet = p.inlet_values() if inlet_values is None else np.asarray(inlet_values, dtype=float)
    if inlet.shape != (p.ny,) or not np.all(np.isfinite(inlet)):
        raise ValueError("inlet_values must be a finite vector of length ny")
    ax = 1 / p.dx
    bx, by = p.diffusion / p.dx**2, p.diffusion / p.dy**2
    indices = np.arange(p.ndof).reshape(p.ny, p.nx)
    diagonal = np.full((p.ny, p.nx), -ax - 2 * bx - 2 * by)
    diagonal[:, 0] -= bx       # Left face is dx/2 from the cell centre.
    diagonal[:, -1] += bx     # Right diffusive flux is zero.
    rows = [indices.ravel()]
    cols = [indices.ravel()]
    values = [diagonal.ravel()]

    def add(row, col, coefficient):
        rows.append(row.ravel())
        cols.append(col.ravel())
        values.append(np.full(row.size, coefficient))

    add(indices[:, 1:], indices[:, :-1], ax + bx)
    add(indices[:, :-1], indices[:, 1:], bx)
    add(indices, np.roll(indices, 1, axis=0), by)
    add(indices, np.roll(indices, -1, axis=0), by)
    A = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(p.ndof, p.ndof)).tocsr()
    f = np.zeros((p.ny, p.nx))
    f[:, 0] = (ax + 2 * bx) * inlet
    return DiscreteSystem(p, A, f.ravel(), inlet.copy())


def positive_finite(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def validate_times(times):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("times must be a finite, strictly increasing vector with >= 2 entries")
    return times


def integrate_euler(A, f, initial, times, max_dt) -> Trajectory:
    """Shared Euler integrator: subdivide every output interval exactly.

    max_dt is a cap, not a demand for a fixed step. This also handles a
    final short interval without shifting the requested output times.
    """
    positive_finite(max_dt, "dt")
    times = validate_times(times)
    state = np.asarray(initial, dtype=float).copy()
    f = np.asarray(f, dtype=float)
    if state.shape != f.shape or state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError("initial must be a finite vector matching the system dimension")
    snapshots = np.empty((state.size, times.size))
    snapshots[:, 0] = state
    counts = np.maximum(1, np.ceil(np.diff(times) / max_dt - 1e-12).astype(int))
    steps = np.diff(times) / counts
    for k, (nsteps, dt) in enumerate(zip(counts, steps)):
        for _ in range(nsteps):
            state += dt * (A @ state + f)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("Euler integration produced non-finite states; reduce dt")
        snapshots[:, k + 1] = state
    return Trajectory(times.copy(), snapshots, steps, counts)


def solve_fvm(system: DiscreteSystem, t_final=1.5, snapshot_interval=0.01,
              dt=None, initial=None) -> Trajectory:
    """Use a default cap of 0.8*dt_limit; reject unstable manual caps."""
    positive_finite(t_final, "t_final")
    positive_finite(snapshot_interval, "snapshot_interval")
    max_dt = 0.8 * system.dt_limit if dt is None else dt
    positive_finite(max_dt, "dt")
    if max_dt > system.dt_limit * (1 + 1e-12):
        raise ValueError(f"dt={max_dt:g} exceeds the stable Euler limit {system.dt_limit:g}")
    n = int(np.floor(t_final / snapshot_interval + 1e-12))
    times = np.arange(n + 1, dtype=float) * snapshot_interval
    if np.isclose(times[-1], t_final, rtol=1e-12, atol=1e-14):
        times[-1] = t_final
    else:
        times = np.append(times, t_final)
    state = np.zeros(system.problem.ndof) if initial is None else np.asarray(initial, dtype=float)
    if state.shape != (system.problem.ndof,):
        raise ValueError("initial must have shape (ndof,)")
    return integrate_euler(system.A, system.f, state, times, max_dt)
