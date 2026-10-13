"""Physical flux checks independent of the sparse matrix assembly."""
import numpy as np
import pytest

from diffusion2d.problem import Problem
from diffusion2d.fvm import assemble_system, solve_fvm


def face_fluxes(problem, c, inlet):
    """Positive flux points in increasing x/y; the domain has unit depth."""
    ny, nx = c.shape
    fx = np.empty((ny, nx + 1))
    fx[:, 0] = inlet - problem.diffusion * (c[:, 0] - inlet) / (problem.dx / 2)
    fx[:, 1:nx] = c[:, :-1] - problem.diffusion * np.diff(c, axis=1) / problem.dx
    fx[:, nx] = c[:, -1]
    fy = -problem.diffusion * (np.roll(c, -1, axis=0) - c) / problem.dy
    return fx, fy


@pytest.mark.parametrize("nx,ny", [(5, 4), (3, 2), (2, 2)])
def test_sparse_operator_matches_face_flux_divergence(nx, ny):
    p = Problem(nx=nx, ny=ny)
    inlet = np.random.default_rng(9).random(p.ny)
    c = np.random.default_rng(12).random((p.ny, p.nx))
    system = assemble_system(p, inlet)
    fx, fy = face_fluxes(p, c, inlet)
    rhs = -(fx[:, 1:] - fx[:, :-1]) / p.dx - (fy - np.roll(fy, 1, axis=0)) / p.dy
    np.testing.assert_allclose((system.A @ c.ravel() + system.f).reshape(c.shape), rhs)


def test_half_cell_inlet_outflow_and_periodic_connections():
    p = Problem(nx=4, ny=5)
    s = assemble_system(p, np.ones(p.ny))
    # dx=1/4, dy=1/5; diffusion_x=0.16, diffusion_y=0.25.
    assert s.A[0, 0] == pytest.approx(-4.98)
    assert s.f[0] == pytest.approx(4.32)
    assert s.A[3, 3] == pytest.approx(-4.66)
    assert s.A[0, 16] == pytest.approx(0.25)
    assert s.A[0, 1] == pytest.approx(0.16)
    assert s.A[1, 0] == pytest.approx(4.16)


def test_mass_change_equals_boundary_flux():
    p = Problem(nx=7, ny=6)
    s = assemble_system(p)
    c = np.random.default_rng(4).random((p.ny, p.nx))
    fx, _ = face_fluxes(p, c, s.inlet)
    mass_rate = np.sum(s.A @ c.ravel() + s.f) * p.dx * p.dy
    assert mass_rate == pytest.approx(np.sum(fx[:, 0] - fx[:, -1]) * p.dy)


def test_uniform_state_is_steady_with_uniform_inlet():
    p = Problem(nx=6, ny=5)
    s = assemble_system(p, np.ones(p.ny))
    result = solve_fvm(s, t_final=0.2, initial=np.ones(p.ndof))
    np.testing.assert_allclose(result.snapshots, 1, atol=2e-14)


def test_zero_inlet_and_initial_state_remain_zero():
    p = Problem(nx=5, ny=4)
    result = solve_fvm(assemble_system(p, np.zeros(p.ny)), t_final=0.1)
    np.testing.assert_array_equal(result.snapshots, 0)


def test_monotone_euler_and_exact_snapshot_times():
    p = Problem(nx=20, ny=18)
    s = assemble_system(p)
    result = solve_fvm(s)
    assert result.snapshots.shape == (360, 151)
    assert np.min(result.snapshots) >= -1e-14
    assert np.max(result.snapshots) <= 1 + 1e-14
    assert result.dt <= 0.8 * s.dt_limit * (1 + 1e-12)
    for t in (0.2, 0.5, 1.0, 1.5):
        assert result.times[round(t / 0.01)] == pytest.approx(t, abs=1e-14)


def test_formal_grid_rejects_dt_point_zero_one():
    s = assemble_system(Problem(nx=200, ny=200))
    assert s.dt_limit == pytest.approx(1 / 2200)
    with pytest.raises(ValueError, match="stable"):
        solve_fvm(s, dt=0.01)


def test_strict_inlet_band():
    p = Problem(nx=3, ny=3)
    np.testing.assert_array_equal(p.inlet_values(), [0, 1, 0])


def test_final_partial_interval_and_manual_dt_cap():
    s = assemble_system(Problem(nx=4, ny=4))
    result = solve_fvm(s, t_final=0.025, snapshot_interval=0.01, dt=0.003)
    np.testing.assert_allclose(result.times, [0, 0.01, 0.02, 0.025])
    assert result.dt <= 0.003
    assert result.step_counts.tolist() == [4, 4, 2]


@pytest.mark.parametrize("kwargs", [{"nx": 1}, {"ny": 0}, {"nx": 3.5}, {"diffusion": -1}, {"diffusion": np.nan}])
def test_invalid_problem_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Problem(**kwargs)


@pytest.mark.parametrize("kwargs", [{"t_final": 0}, {"snapshot_interval": 0}, {"dt": -0.1}, {"dt": np.nan}])
def test_invalid_time_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        solve_fvm(assemble_system(Problem(nx=3, ny=3)), **kwargs)
