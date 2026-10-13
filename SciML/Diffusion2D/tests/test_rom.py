import numpy as np
import pytest

from diffusion2d.problem import Problem
from diffusion2d.fvm import assemble_system, solve_fvm
from diffusion2d.pod import fit_pod, solve_pod
from diffusion2d.dmd import fit_dmd, predict_dmd


def test_pod_orthogonality_projection_and_full_basis_equivalence():
    s = assemble_system(Problem(nx=3, ny=3))
    # An independent full-rank training matrix ensures no truncation error.
    model = fit_pod(s, np.eye(9), rank=9)
    v = model.basis.vectors
    np.testing.assert_allclose(v.T @ v, np.eye(9), atol=1e-14)
    np.testing.assert_allclose(model.A, v.T @ s.A @ v)
    np.testing.assert_allclose(model.f, v.T @ s.f)
    initial = np.linspace(0, 1, 9)
    full = solve_fvm(s, t_final=0.1, dt=0.003, initial=initial)
    reduced = solve_pod(model, full.times, dt=full.dt, initial=initial)
    np.testing.assert_allclose(reduced.snapshots, full.snapshots, atol=1e-13)


def test_pod_truncation_keeps_largest_energy_direction():
    s = assemble_system(Problem(nx=2, ny=2))
    c = np.diag([4.0, 3.0, 0.0, 0.0])
    model = fit_pod(s, c, rank=1)
    assert model.basis.rank == 1
    assert model.basis.energy[0] == pytest.approx(16 / 25)
    projected = model.basis.vectors @ (model.basis.vectors.T @ c)
    assert np.linalg.norm(c - projected) == pytest.approx(3)


def affine_training_data():
    b = np.diag([0.8, 0.5])
    force = np.array([0.2, -0.1])
    c = np.zeros((2, 21))
    for k in range(20):
        c[:, k + 1] = b @ c[:, k] + force
    return c, np.arange(21) * 0.1, b, force


def test_augmented_dmd_recovers_affine_map_spectrum_and_extrapolation():
    c, times, b, force = affine_training_data()
    model = fit_dmd(c, times, rank=2)
    np.testing.assert_array_equal(model.operator[-1], [0, 0, 1])
    np.testing.assert_allclose(np.sort(model.eigenvalues.real), [0.5, 0.8, 1], atol=1e-12)
    prediction_times = np.arange(31) * 0.1
    prediction = predict_dmd(model, prediction_times)
    expected = np.zeros((2, 31))
    for k in range(30):
        expected[:, k + 1] = b @ expected[:, k] + force
    np.testing.assert_allclose(prediction.snapshots, expected, atol=1e-12)
    assert model.modes.shape == (2, 3)
    # Physical modes obey the fitted affine augmented operator as eigenvectors.
    augmented_modes = np.vstack([model.modes, model.eigenvectors[-1:]])
    physical_operator = np.block([[b, force[:, None]], [np.zeros((1, 2)), np.ones((1, 1))]])
    np.testing.assert_allclose(physical_operator @ augmented_modes,
                               augmented_modes * model.eigenvalues, atol=1e-12)


def test_rank_deficient_snapshots_report_effective_rank():
    c = np.outer([1.0, 2.0, 0.0, 0.0], np.arange(8, dtype=float))
    times = np.arange(8) * 0.1
    system = assemble_system(Problem(nx=2, ny=2))
    pod = fit_pod(system, c, rank=20)
    dmd = fit_dmd(c, times, rank=20)
    assert pod.basis.rank == dmd.basis.rank == 1
    assert pod.basis.requested_rank == 20
    np.testing.assert_allclose(predict_dmd(dmd, times).snapshots, c, atol=1e-12)


def test_zero_snapshots_have_finite_empty_basis_models():
    c = np.zeros((4, 5))
    times = np.arange(5) * 0.1
    pod = fit_pod(assemble_system(Problem(nx=2, ny=2)), c, rank=5)
    dmd = fit_dmd(c, times, rank=5)
    assert pod.basis.rank == dmd.basis.rank == 0
    np.testing.assert_array_equal(pod.basis.energy, 0)
    np.testing.assert_array_equal(predict_dmd(dmd, times).snapshots, 0)
    np.testing.assert_array_equal(solve_pod(pod, times, dt=0.01).snapshots, 0)


@pytest.mark.parametrize("times", [[0, 0.1, 0.25], [0, 0, 0.1], [0, 0.1, np.nan]])
def test_dmd_rejects_nonuniform_or_invalid_training_times(times):
    with pytest.raises(ValueError):
        fit_dmd(np.ones((4, 3)), np.array(times), rank=2)


def test_dmd_rejects_fractional_and_backward_predictions():
    c, times, _, _ = affine_training_data()
    model = fit_dmd(c, times, rank=2)
    for query in ([0, 0.15], [-0.1, 0]):
        with pytest.raises(ValueError):
            predict_dmd(model, np.array(query))


@pytest.mark.parametrize("rank", [0, -1, 1.5])
def test_invalid_rank_is_rejected(rank):
    c, times, _, _ = affine_training_data()
    with pytest.raises(ValueError):
        fit_dmd(c, times, rank=rank)
