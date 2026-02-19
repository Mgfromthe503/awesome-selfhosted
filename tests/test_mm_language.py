import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mm_language import parse_mm_script, run_mm_script


def test_mm_language_parse_and_execute():
    script = """
🔮 Correspondence
DETECT qubit IN qubit report
ANOMALY SERIES: 1,1,1,10,1
TAG mode=detective
"""
    parsed = parse_mm_script(script)
    assert len(parsed) == 4

    results = run_mm_script(script)
    assert len(results) == 4
    anomaly = [r for r in results if r["op"] == "ANOMALY"][0]
    assert isinstance(anomaly["result"]["anomaly_indexes"], list)
