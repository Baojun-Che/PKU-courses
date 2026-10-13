"""Run the actual CLI and read its numerical and image artifacts."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from matplotlib import image as mpl_image


ROOT = Path(__file__).resolve().parents[1]


def test_demo_writes_readable_four_time_results_and_models(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "run_demo.py"), "--nx", "8", "--ny", "8",
         "--ranks", "2", "4", "--output", str(tmp_path)],
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["report_times"] == [0.2, 0.5, 1.0, 1.5]
    with np.load(tmp_path / "fvm.npz") as full:
        assert full["snapshots"].shape == (64, 151)
        assert np.isfinite(full["snapshots"]).all()
    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16  # Two algorithms x two ranks x four times.
    for row in rows:
        assert np.isfinite(float(row["relative_l2"]))
        assert float(row["absolute_l2"]) >= 0
    for rank in (2, 4):
        for method in ("pod", "dmd"):
            with np.load(tmp_path / f"{method}_r{rank}.npz") as data:
                assert data["snapshots"].shape == (64, 151)
                assert np.isfinite(data["snapshots"]).all()
                assert int(data["effective_rank"]) <= rank
        with np.load(tmp_path / f"dmd_r{rank}.npz") as data:
            assert np.min(np.abs(data["eigenvalues"] - 1)) < 1e-12
            np.testing.assert_array_equal(data["operator"][-1, :-1], 0)
    for time in (0.2, 0.5, 1.0, 1.5):
        image = mpl_image.imread(tmp_path / f"concentration_t{time:.2f}.png")
        assert image.shape[0] > 100 and image.shape[1] > 100
    for name in ("pod_energy.png", "errors.png", "dmd_spectrum_r2.png", "dmd_modes_r4.png"):
        assert np.isfinite(mpl_image.imread(tmp_path / name)).all()


@pytest.mark.parametrize("args", [["--snapshot-interval", "0.03"], ["--t-final", "0.255"], ["--ranks", "0"]])
def test_demo_rejects_incompatible_settings_without_partial_outputs(tmp_path, args):
    output = tmp_path / "invalid"
    result = subprocess.run([sys.executable, str(ROOT / "run_demo.py"), *args,
                             "--output", str(output)], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert not output.exists()
