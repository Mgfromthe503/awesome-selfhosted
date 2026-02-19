from datetime import datetime

import pytest

from mental_matrix_ai import (
    AstrologyModule,
    MentalMatrixAI,
    QuantumBioinformaticsModule,
    QuantumChemistryModule,
    QuantumEmotionalModel,
)


def test_zodiac_sign_boundaries():
    module = AstrologyModule()
    assert module.get_zodiac_sign(datetime(1988, 11, 22)) == "Sagittarius"
    assert module.get_zodiac_sign(datetime(1988, 1, 15)) == "Capricorn"


def test_dna_analysis():
    module = QuantumBioinformaticsModule()
    result = module.analyze_dna_sequence("ATCG")
    assert "len=4" in result
    assert "gc=0.50" in result


@pytest.mark.parametrize("invalid", ["", "ABCX", "ATUG"])
def test_invalid_dna_raises(invalid):
    module = QuantumBioinformaticsModule()
    with pytest.raises(ValueError):
        module.analyze_dna_sequence(invalid)


def test_molecule_analysis():
    module = QuantumChemistryModule()
    result = module.analyze_molecule("H2O")
    assert "total_atoms=3" in result


def test_emotional_model_updates_levels():
    model = QuantumEmotionalModel()
    reading = model.handle_input("I love this")
    assert reading.love_level >= 0
    assert pytest.approx(reading.fear_level, abs=1e-7) == -reading.love_level


def test_ai_decision_contains_parts():
    ai = MentalMatrixAI()
    decision = ai.make_decision(datetime(1988, 11, 22), "ATCG", "H2O", "I love this")
    assert "Decision based on:" in decision
    assert "emotion=" in decision
