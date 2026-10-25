"""The README hero plot is regenerated from the committed, measured bench JSON."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("plot_bench", ROOT / "scripts" / "plot_bench.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plot_from_committed_bench_json(tmp_path):
    pytest.importorskip("matplotlib")
    data = json.loads((ROOT / "docs" / "assets" / "bench.json").read_text())
    out = _load().plot(data, tmp_path / "bench.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (ROOT / "docs" / "assets" / "bench.png").exists()
