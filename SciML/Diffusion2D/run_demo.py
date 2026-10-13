"""Run a reproducible FVM/POD/DMD comparison; see README.md for commands."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures on both desktop and headless machines.
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq

from diffusion2d import Problem, assemble_system, solve_fvm, fit_pod, solve_pod, fit_dmd, predict_dmd


REPORT_TIMES = (0.2, 0.5, 1.0, 1.5)


def build_parser():
    parser = argparse.ArgumentParser(description="2D finite-volume / POD / augmented DMD demo")
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--formal", action="store_true", help="Use the homework's 200 x 200 grid")
    parser.add_argument("--t-final", type=float, default=1.5)
    parser.add_argument("--snapshot-interval", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=None, help="Euler step cap; must satisfy the FVM stability limit")
    parser.add_argument("--ranks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    return parser


def error_series(reference, prediction, cell_area):
    difference = prediction - reference
    absolute = np.linalg.norm(difference, axis=0) * np.sqrt(cell_area)
    reference_norm = np.linalg.norm(reference, axis=0) * np.sqrt(cell_area)
    relative = np.divide(absolute, reference_norm, out=np.zeros_like(absolute), where=reference_norm > 0)
    relative[(reference_norm == 0) & (absolute > 0)] = np.inf
    return absolute, relative, np.max(np.abs(difference), axis=0)


def plot_concentrations(problem, full, results, report_times, output):
    fields = [("FVM", full.snapshots)] + [(item["label"], item["trajectory"].snapshots) for item in results]
    ncols = min(4, len(fields))
    nrows = int(np.ceil(len(fields) / ncols))
    for time in report_times:
        index = int(np.argmin(np.abs(full.times - time)))
        vmin = min(0.0, min(float(np.min(c[:, index])) for _, c in fields))
        vmax = max(1.0, max(float(np.max(c[:, index])) for _, c in fields))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows),
                                 squeeze=False, constrained_layout=True)
        for ax, (label, c) in zip(axes.ravel(), fields):
            image = ax.imshow(c[:, index].reshape(problem.ny, problem.nx), origin="lower",
                              extent=(0, 1, 0, 1), vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set(title=label, xlabel="x", ylabel="y")
        for ax in axes.ravel()[len(fields):]:
            ax.set_visible(False)
        fig.colorbar(image, ax=list(axes.ravel()[:len(fields)]), label="Concentration")
        fig.suptitle(f"t = {time:.2f}")
        fig.savefig(output / f"concentration_t{time:.2f}.png", dpi=140)
        plt.close(fig)


def plot_energy(basis, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    indices = np.arange(1, basis.singular_values.size + 1)
    axes[0].semilogy(indices, np.maximum(basis.singular_values, np.finfo(float).tiny))
    axes[0].set(xlabel="Mode number", ylabel="Singular value", title="Uncentred snapshot SVD")
    axes[1].plot(indices, basis.energy)
    axes[1].set(xlabel="Retained modes", ylabel="Cumulative energy", ylim=(0, 1.01))
    fig.savefig(output / "pod_energy.png", dpi=140)
    plt.close(fig)


def plot_dmd(model, problem, requested_rank, output):
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    angle = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(angle), np.sin(angle), "--", color="gray", label="Unit circle")
    ax.scatter(model.eigenvalues.real, model.eigenvalues.imag, marker="x", label="DMD eigenvalues")
    ax.set(xlabel="Re(lambda)", ylabel="Im(lambda)", title=f"Discrete spectrum, rank {model.basis.rank}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    fig.savefig(output / f"dmd_spectrum_r{requested_rank}.png", dpi=140)
    plt.close(fig)

    # Illustrative ordering by initial amplitude times physical mode norm.
    q0 = np.append(model.basis.vectors.T @ model.initial_state, 1)
    amplitudes = lstsq(model.eigenvectors, q0)[0]
    scores = np.abs(amplitudes) * np.linalg.norm(model.modes, axis=0)
    selected = np.argsort(scores)[::-1][:min(4, model.eigenvalues.size)]
    fig, axes = plt.subplots(len(selected), 2, figsize=(7, 2.8 * len(selected)),
                             squeeze=False, constrained_layout=True)
    for row, mode_index in enumerate(selected):
        mode = model.modes[:, mode_index].reshape(problem.ny, problem.nx)
        bound = max(float(np.max(np.abs(mode.real))), float(np.max(np.abs(mode.imag))), 1e-15)
        for ax, values, component in zip(axes[row], (mode.real, mode.imag), ("Real", "Imag")):
            image = ax.imshow(values, origin="lower", extent=(0, 1, 0, 1),
                              cmap="RdBu_r", vmin=-bound, vmax=bound)
            eigenvalue = model.eigenvalues[mode_index]
            ax.set(title=f"{component}: lambda={eigenvalue.real:.4f}{eigenvalue.imag:+.4f}j", xlabel="x", ylabel="y")
            fig.colorbar(image, ax=ax)
    fig.savefig(output / f"dmd_modes_r{requested_rank}.png", dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.ranks or any(rank <= 0 for rank in args.ranks):
            raise ValueError("ranks must be positive integers")
        if not np.isfinite(args.t_final) or args.t_final <= 0:
            raise ValueError("t-final must be finite and positive")
        if not np.isfinite(args.snapshot_interval) or args.snapshot_interval <= 0:
            raise ValueError("snapshot-interval must be finite and positive")
        if args.t_final < args.snapshot_interval:
            raise ValueError("t-final must include at least one snapshot interval")
        report_times = [t for t in REPORT_TIMES if t <= args.t_final + 1e-12]
        if not report_times:
            report_times = [args.t_final]
        # The solver allows partial intervals; this DMD demo needs uniform data.
        for time in [args.t_final, *report_times]:
            ratio = time / args.snapshot_interval
            if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
                raise ValueError("t-final and report times must be integer multiples of snapshot-interval for DMD")
        problem = Problem(nx=200 if args.formal else args.nx, ny=200 if args.formal else args.ny)
        system = assemble_system(problem)
        full = solve_fvm(system, args.t_final, args.snapshot_interval, args.dt)
    except ValueError as error:
        parser.error(str(error))

    print(f"FVM: {problem.nx}x{problem.ny}, stable limit={system.dt_limit:.8g}, "
          f"actual dt<={full.dt:.8g}, steps={int(np.sum(full.step_counts))}", flush=True)
    results = []
    for rank in dict.fromkeys(args.ranks):
        pod = fit_pod(system, full.snapshots, rank)
        dmd = fit_dmd(full.snapshots, full.times, rank)
        for method, model, trajectory in (
                ("POD", pod, solve_pod(pod, full.times, full.dt)),
                ("DMD", dmd, predict_dmd(dmd, full.times))):
            if not np.all(np.isfinite(trajectory.snapshots)):
                raise FloatingPointError(f"{method} rank {rank} produced non-finite predictions")
            absolute, relative, linf = error_series(full.snapshots, trajectory.snapshots, problem.dx * problem.dy)
            results.append(dict(method=method, model=model, trajectory=trajectory, rank=rank,
                                label=f"{method} r={model.basis.rank}", absolute=absolute,
                                relative=relative, linf=linf))
            print(f"{method}: requested rank={rank}, effective rank={model.basis.rank}, "
                  f"final relative L2={relative[-1]:.6g}", flush=True)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "fvm.npz", times=full.times, snapshots=full.snapshots,
                        x=problem.x, y=problem.y, step_sizes=full.step_sizes, step_counts=full.step_counts)
    summary = dict(nx=problem.nx, ny=problem.ny, dx=problem.dx, dy=problem.dy,
                   diffusion=problem.diffusion, velocity=[1, 0], t_final=args.t_final,
                   snapshot_interval=args.snapshot_interval, dt_limit=system.dt_limit, actual_max_dt=full.dt,
                   snapshot_count=full.times.size, report_times=report_times,
                   training="full trajectory reconstruction; not an extrapolation test", models=[])
    metrics = []
    for item in results:
        model = item["model"]
        basis = model.basis
        data = dict(times=full.times, snapshots=item["trajectory"].snapshots, basis=basis.vectors,
                    singular_values=basis.singular_values, cumulative_energy=basis.energy,
                    requested_rank=item["rank"], effective_rank=basis.rank, numerical_rank=basis.numerical_rank)
        if item["method"] == "POD":
            data.update(A=model.A, f=model.f)
        else:
            data.update(operator=model.operator, eigenvalues=model.eigenvalues, eigenvectors=model.eigenvectors,
                        modes=model.modes, snapshot_interval=model.snapshot_interval, t0=model.t0)
            plot_dmd(model, problem, item["rank"], output)
        np.savez_compressed(output / f"{item['method'].lower()}_r{item['rank']}.npz", **data)
        summary["models"].append(dict(method=item["method"], requested_rank=item["rank"],
                                      effective_rank=basis.rank, final_relative_l2=float(item["relative"][-1])))
        for time in report_times:
            index = int(np.argmin(np.abs(full.times - time)))
            metrics.append(dict(method=item["method"], requested_rank=item["rank"], effective_rank=basis.rank,
                                time=time, absolute_l2=float(item["absolute"][index]),
                                relative_l2=float(item["relative"][index]), absolute_linf=float(item["linf"][index]),
                                concentration_min=float(np.min(item["trajectory"].snapshots[:, index])),
                                concentration_max=float(np.max(item["trajectory"].snapshots[:, index]))))
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    plot_concentrations(problem, full, results, report_times, output)
    plot_energy(results[0]["model"].basis, output)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for item in results:
        ax.semilogy(full.times[1:], np.maximum(item["relative"][1:], 1e-16), label=item["label"])
    ax.set(xlabel="t", ylabel="Relative L2 error", title="Full-trajectory reconstruction error")
    ax.legend()
    fig.savefig(output / "errors.png", dpi=140)
    plt.close(fig)
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
