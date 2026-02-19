"""Mental Matrix AI core modules.

This module provides a small, testable implementation of the user-provided
Mental Matrix AI concept with pluggable submodules and sentiment-driven
emotion tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - fallback when matplotlib isn't installed.
    matplotlib = None
    plt = None

try:
    from textblob import TextBlob
except Exception:  # pragma: no cover - fallback when TextBlob isn't installed.
    TextBlob = None


class AstrologyModule:
    """Simple astrology helper methods."""

    def get_zodiac_sign(self, birthdate: datetime) -> str:
        month_day = (birthdate.month, birthdate.day)
        if (3, 21) <= month_day <= (4, 19):
            return "Aries"
        return "Unknown"

    def get_planetary_positions(self) -> Dict[str, str]:
        return {"Mars": "Aries", "Venus": "Taurus"}


class QuantumBioinformaticsModule:
    """Placeholder DNA analysis module."""

    def analyze_dna_sequence(self, dna_sequence: str) -> str:
        gc_content = 0.0
        if dna_sequence:
            gc = sum(1 for base in dna_sequence.upper() if base in {"G", "C"})
            gc_content = gc / len(dna_sequence)
        return f"GC content: {gc_content:.2f}"


class QuantumChemistryModule:
    """Placeholder molecule analysis module."""

    def analyze_molecule(self, molecule: str) -> str:
        return f"Molecule '{molecule}' analyzed"


@dataclass
class QuantumEmotionalModel:
    """Tracks love/fear scores and can render a plot."""

    love_level: float = 0.0
    fear_level: float = 0.0
    history: List[Tuple[str, float, float]] = field(default_factory=list)

    def handle_input(self, user_input: str) -> None:
        polarity = self._sentiment_polarity(user_input)
        self.love_level = polarity
        self.fear_level = -polarity
        self.history.append((user_input, self.love_level, self.fear_level))

    def _sentiment_polarity(self, text: str) -> float:
        if TextBlob is not None:
            return float(TextBlob(text).sentiment.polarity)

        lowered = text.lower()
        positive = sum(word in lowered for word in ["love", "amazing", "great", "good"])
        negative = sum(word in lowered for word in ["hate", "terrible", "bad", "awful"])
        score = (positive - negative) * 0.25
        return max(-1.0, min(1.0, score))

    def plot_emotions(
        self,
        inputs: Sequence[str],
        love_levels: Sequence[float],
        fear_levels: Sequence[float],
        output_path: str = "emotion_plot.png",
    ) -> str:
        if plt is None:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write("input,love,fear\n")
                for item, love, fear in zip(inputs, love_levels, fear_levels):
                    handle.write(f"{item},{love},{fear}\n")
            return output_path

        plt.figure(figsize=(8, 4))
        plt.plot(inputs, love_levels, label="Love Level")
        plt.plot(inputs, fear_levels, label="Fear Level")
        plt.xlabel("Inputs")
        plt.ylabel("Levels")
        plt.title("Love and Fear Levels Over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path


class MentalMatrixAI:
    """Core orchestrator for module outputs."""

    def __init__(self) -> None:
        self.astrology_module = AstrologyModule()
        self.quantum_bio_module = QuantumBioinformaticsModule()
        self.quantum_chem_module = QuantumChemistryModule()
        self.quantum_emotion_model = QuantumEmotionalModel()

    def make_decision(
        self,
        user_birthdate: datetime,
        dna_sequence: str,
        molecule: str,
        user_input: str,
    ) -> str:
        zodiac_sign = self.astrology_module.get_zodiac_sign(user_birthdate)
        planetary_positions = self.astrology_module.get_planetary_positions()
        dna_analysis = self.quantum_bio_module.analyze_dna_sequence(dna_sequence)
        molecule_analysis = self.quantum_chem_module.analyze_molecule(molecule)
        self.quantum_emotion_model.handle_input(user_input)

        return (
            "Decision based on: "
            f"{zodiac_sign}, {planetary_positions}, {dna_analysis}, {molecule_analysis}"
        )


def run_demo() -> Tuple[List[str], List[float], List[float]]:
    ai_core = MentalMatrixAI()
    user_birthdate = datetime.strptime("1988-11-22", "%Y-%m-%d")
    dna_sequence = "ATCGATCGATCG"
    molecule = "H2O"
    user_inputs = ["I love you", "I hate this", "You're amazing", "This is terrible"]
    love_levels: List[float] = []
    fear_levels: List[float] = []

    for user_input in user_inputs:
        decision = ai_core.make_decision(user_birthdate, dna_sequence, molecule, user_input)
        print(decision)
        love_levels.append(ai_core.quantum_emotion_model.love_level)
        fear_levels.append(ai_core.quantum_emotion_model.fear_level)

    ai_core.quantum_emotion_model.plot_emotions(user_inputs, love_levels, fear_levels)
    return user_inputs, love_levels, fear_levels


if __name__ == "__main__":
    run_demo()
