from pathlib import Path

from sherlock_live_workflow import DNAState, Sherlock, bias_classify, bio_vector, privacy_hash, run_demo


def test_bio_vector_has_12_dimensions():
    vec = bio_vector("ATGCTCGGATCAGT")
    assert len(vec) == 12


def test_bias_classify_detects_toxicity_signal():
    result = bias_classify("You are terrible")
    assert result["toxic"] > 0
    assert round(result["toxic"] + result["non_toxic"], 4) == 1.0


def test_privacy_hash_is_sha256(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("abc", encoding="utf-8")
    digest = privacy_hash(sample)
    assert len(digest) == 64


def test_dna_principle_integration_and_truth_path():
    dna = DNAState.create()
    dna.integrate_principle({"Code": 3, "Name": "Vibration"})
    result = dna.analyze_data("ATGC")
    assert "QuantumAnalysis" in result


def test_sherlock_demo_runs_end_to_end():
    output = run_demo()
    assert "findings" in output
    assert "bias" in output
    assert "report" in output
    assert isinstance(output["findings"]["Vector"], list)


def test_sherlock_privacy_audit_wrapper(tmp_path: Path):
    sample = tmp_path / "asset.bin"
    sample.write_bytes(b"1234")
    sher = Sherlock()
    assert sher.privacy_audit(sample) == privacy_hash(sample)
