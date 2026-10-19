"""Validate existing results and prepare portable LaTeX figures/tables.

No PDE solve or model fitting occurs here. Run from any working directory.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np


REPORT = Path(__file__).resolve().parent
TIMES = (0.2, 0.5, 1.0, 1.5)
RANKS = (5, 10, 20)


def scientific(value):
    """Format a number as a LaTeX scientific-notation math cell."""
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{value:.4e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPORT.parent / "outputs_formal")
    args = parser.parse_args()
    source = args.source.resolve()
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    nx, ny = summary["nx"], summary["ny"]
    if not np.isclose(summary["t_final"], 1.5):
        raise ValueError("This report expects T=1.5")
    with (source / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(row["method"], int(row["requested_rank"]), float(row["time"])): row for row in rows}
    with np.load(source / "fvm.npz") as data:
        times = data["times"].copy()
        full = data["snapshots"].copy()
        steps = int(np.sum(data["step_counts"]))
    if full.shape != (nx * ny, times.size) or not np.isfinite(full).all():
        raise ValueError("Invalid full-order snapshot data")
    indices = [int(np.argmin(abs(times - t))) for t in TIMES]
    if not np.allclose(times[indices], TIMES, atol=1e-12, rtol=0):
        raise ValueError("Missing required report times")
    cell_area = summary["dx"] * summary["dy"]
    relative_rows, pod_rows, spectra = [], [], []
    for rank in RANKS:
        for method in ("POD", "DMD"):
            with np.load(source / f"{method.lower()}_r{rank}.npz") as data:
                prediction = data["snapshots"]
                effective_rank = int(data["effective_rank"])
                if prediction.shape != full.shape or not np.isfinite(prediction).all():
                    raise ValueError(f"Invalid {method} snapshots")
                relative = np.linalg.norm(prediction[:, indices] - full[:, indices], axis=0) / np.linalg.norm(full[:, indices], axis=0)
                for t, value in zip(TIMES, relative):
                    recorded = float(lookup[(method, rank, t)]["relative_l2"])
                    if not np.isclose(value, recorded, rtol=1e-7, atol=1e-13):
                        raise ValueError(f"CSV mismatch for {method}, r={rank}, t={t}")
                relative_rows.append(f"{method} & {effective_rank} & " + " & ".join(map(scientific, relative)) + r" \\")
                if method == "POD":
                    eig = np.linalg.eigvals(data["A"])
                    if np.any(eig.real >= 0):
                        raise ValueError("Non-decaying POD spectrum; revise stability discussion")
                    spectral_limit = np.min(-2 * eig.real / np.abs(eig)**2)
                    energy = data["cumulative_energy"][effective_rank - 1]
                    pod_rows.append(f"{effective_rank} & {energy:.14f} & {spectral_limit:.8f}" + r" \\")
                else:
                    eig = data["eigenvalues"]
                    expected_row = np.zeros(effective_rank + 1)
                    expected_row[-1] = 1
                    if not np.array_equal(data["operator"][-1], expected_row):
                        raise ValueError("The DMD constant coordinate was not preserved")
                    for k, value in enumerate(sorted(eig, key=lambda z: (z.real, z.imag)), 1):
                        spectra.append(f"{effective_rank} & {k} & {value.real:.8f} & {value.imag:+.8f} & {abs(value):.8f}" + r" \\")
    fvm_rows = []
    for t, k in zip(TIMES, indices):
        c = full[:, k]
        fvm_rows.append(f"{t:.1f} & {np.sum(c)*cell_area:.8f} & {scientific(np.min(c))} & {np.max(c):.8f}" + r" \\")
    figures = [f"concentration_t{t:.2f}.png" for t in TIMES] + ["pod_energy.png", "errors.png"]
    figures += [f"dmd_{kind}_r{rank}.png" for rank in RANKS for kind in ("spectrum", "modes")]
    for name in figures:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    for folder in ("figures", "tables", "data"):
        (REPORT / folder).mkdir(parents=True, exist_ok=True)
    for name in figures:
        shutil.copy2(source / name, REPORT / "figures" / name)
    for name in ("summary.json", "metrics.csv"):
        shutil.copy2(source / name, REPORT / "data" / name)
    for name, content in (("relative_errors.tex", relative_rows), ("pod_diagnostics.tex", pod_rows),
                          ("fvm_statistics.tex", fvm_rows), ("dmd_spectrum.tex", spectra)):
        (REPORT / "tables" / name).write_text("\n".join(content) + "\n", encoding="utf-8")
    scope = ("当前数据已采用题目规定的空间网格；仍应独立核对实验与推导。" if nx == ny == 200
             else "当前数据为小网格验证结果，尚未满足题目规定的 $200\\times200$ 网格要求；不得将其冒充正式网格实验。")
    macros = {"GridX": str(nx), "GridY": str(ny), "GridDx": f"{summary['dx']:.6g}",
              "GridDy": f"{summary['dy']:.6g}", "SnapshotCount": str(times.size), "StepCount": str(steps),
              "ActualDt": f"{summary['actual_max_dt']:.8g}", "DtLimit": f"{summary['dt_limit']:.8g}",
              "SnapshotInterval": f"{summary['snapshot_interval']:.6g}", "ResolutionNote": scope}
    metadata = "\n".join(r"\newcommand{\%s}{%s}" % (name, value) for name, value in macros.items())
    (REPORT / "tables" / "metadata.tex").write_text(metadata + "\n", encoding="utf-8")
    print(f"Validated results from {source}")
    print(f"Prepared {len(figures)} figures, four numerical tables, metadata and raw CSV/JSON; no simulation rerun.")


if __name__ == "__main__":
    main()
