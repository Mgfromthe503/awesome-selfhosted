from datetime import datetime

from mental_matrix_ai import MentalMatrixAI, QuantumEmotionalModel


def test_make_decision_contains_expected_sections():
    ai = MentalMatrixAI()
    decision = ai.make_decision(
        datetime.strptime("1988-11-22", "%Y-%m-%d"),
        "ATCGATCG",
        "H2O",
        "I love this",
    )

    assert "Decision based on:" in decision
    assert "Molecule 'H2O' analyzed" in decision
    assert "GC content:" in decision


def test_emotion_levels_are_opposites():
    model = QuantumEmotionalModel()
    model.handle_input("I hate this")

    assert model.love_level == -model.fear_level


def test_plot_emotions_creates_file(tmp_path):
    model = QuantumEmotionalModel()
    output = tmp_path / "plot.png"

    saved_path = model.plot_emotions(
        ["a", "b"],
        [0.1, 0.2],
        [-0.1, -0.2],
        output_path=str(output),
    )

    assert saved_path == str(output)
    assert output.exists()
