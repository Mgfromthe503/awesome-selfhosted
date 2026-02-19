from sherlock_quantum_bio import DNASystem, Sherlock, bio_vector, bias_classify, privacy_hash


def test_bio_vector_is_12_dimensional():
    vec = bio_vector("ATGCTCGGATCAGT")
    assert len(vec) == 12
    assert all(isinstance(v, float) for v in vec)


def test_principle_integration_updates_quantum_state():
    dna = DNASystem()
    dna.integrate_principle({"Code": 3, "Name": "Vibration"})
    assert dna.quantum_state.basis == "superposition"


def test_sherlock_investigate_and_report():
    sher = Sherlock()
    sher.dna.integrate_principle({"Code": 3, "Name": "Vibration"})
    findings = sher.investigate("ATGCTCGGATCAGT")
    assert "Vector" in findings
    assert findings["Principles"][0]["Name"] == "Vibration"
    report = sher.report({"Investigation": findings})
    assert "Investigation" in report


def test_bias_and_privacy_audits():
    result = bias_classify("You are a terrible person")
    assert result["toxic"] > 0
    digest = privacy_hash(b"live-data")
    assert len(digest) == 64
