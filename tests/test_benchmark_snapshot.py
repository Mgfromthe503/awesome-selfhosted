import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.benchmark_snapshot import build_benchmark_snapshot


def test_benchmark_snapshot_structure():
    snap = build_benchmark_snapshot()
    assert "timestamp_utc" in snap
    assert "datasets" in snap
    assert "total_records" in snap
    assert "aggregate_sources" in snap
