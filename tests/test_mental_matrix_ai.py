from datetime import datetime, timezone

from mental_matrix_ai import AstrologyModule, MentalMatrixAI, QuantumBioinformaticsModule, QuantumChemistryModule


def test_zodiac_sign_boundaries():
    astrology = AstrologyModule()
    assert astrology.get_zodiac_sign(datetime(1988, 3, 20)) == "Pisces"
    assert astrology.get_zodiac_sign(datetime(1988, 3, 21)) == "Aries"


def test_planetary_positions_are_deterministic_for_input_datetime():
    astrology = AstrologyModule()
    now = datetime(2026, 1, 15, 5, 10, tzinfo=timezone.utc)
    positions = astrology.get_planetary_positions(now)
    assert positions == {"Mars": "Virgo", "Venus": "Aquarius", "Mercury": "Cancer"}


def test_dna_analysis():
    module = QuantumBioinformaticsModule()
    assert module.analyze_dna_sequence("ATCG") == "Length=4, GC=50.00%"
    assert module.analyze_dna_sequence("ATCX") == "Invalid nucleotides: X"


def test_molecule_analysis():
    module = QuantumChemistryModule()
    assert module.analyze_molecule("H2O") == "Atoms=3; Composition=H:2, O:1"


def test_ai_decision_contains_all_components():
    ai = MentalMatrixAI()
    output = ai.make_decision(datetime(1988, 11, 22), "ATCG", "H2O")
    assert "Sagittarius" in output
    assert "Length=4, GC=50.00%" in output
    assert "Atoms=3; Composition=H:2, O:1" in output
