"""Mental Matrix AI core example with lightweight, testable implementations.

This module keeps optional integrations (TextBlob, matplotlib, qiskit) graceful so
it can run in constrained environments while still producing meaningful output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from textblob import TextBlob  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TextBlob = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

try:
    from qiskit import QuantumCircuit  # type: ignore
    from qiskit_aer import AerSimulator  # type: ignore

    _HAS_QISKIT = True
except Exception:  # pragma: no cover - optional dependency
    QuantumCircuit = None  # type: ignore
    AerSimulator = None  # type: ignore
    _HAS_QISKIT = False


@dataclass(frozen=True)
class EmotionReading:
    love_level: float
    fear_level: float


class AstrologyModule:
    """Simple western zodiac and placeholder planetary positions."""

    _zodiac_ranges: Sequence[Tuple[Tuple[int, int], Tuple[int, int], str]] = (
        ((3, 21), (4, 19), "Aries"),
        ((4, 20), (5, 20), "Taurus"),
        ((5, 21), (6, 20), "Gemini"),
        ((6, 21), (7, 22), "Cancer"),
        ((7, 23), (8, 22), "Leo"),
        ((8, 23), (9, 22), "Virgo"),
        ((9, 23), (10, 22), "Libra"),
        ((10, 23), (11, 21), "Scorpio"),
        ((11, 22), (12, 21), "Sagittarius"),
        ((12, 22), (1, 19), "Capricorn"),
        ((1, 20), (2, 18), "Aquarius"),
        ((2, 19), (3, 20), "Pisces"),
    )

    def get_zodiac_sign(self, birthdate: datetime) -> str:
        month_day = (birthdate.month, birthdate.day)
        for start, end, sign in self._zodiac_ranges:
            if start <= end:
                if start <= month_day <= end:
                    return sign
            elif month_day >= start or month_day <= end:
                return sign
        return "Unknown"

    def get_planetary_positions(self) -> Dict[str, str]:
        # Lightweight pseudo-live position buckets from current UTC hour.
        hour = datetime.now(timezone.utc).hour
        signs = [
            "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
            "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
        ]
        return {
            "Mars": signs[hour % 12],
            "Venus": signs[(hour + 3) % 12],
            "Mercury": signs[(hour + 6) % 12],
        }


class QuantumBioinformaticsModule:
    """Tiny DNA analyzer with optional quantum-flavored checksum bit."""

    def analyze_dna_sequence(self, dna_sequence: str) -> str:
        seq = dna_sequence.strip().upper()
        valid = set("ATCG")
        if not seq or any(base not in valid for base in seq):
            raise ValueError("DNA sequence must contain only A/T/C/G and be non-empty.")

        gc_content = (seq.count("G") + seq.count("C")) / len(seq)
        quantum_bit = self._quantum_bit(seq)
        return f"len={len(seq)}, gc={gc_content:.2f}, qbit={quantum_bit}"

    def _quantum_bit(self, seed: str) -> int:
        if _HAS_QISKIT:
            circuit = QuantumCircuit(1, 1)
            if int(sha256(seed.encode("utf-8")).hexdigest(), 16) % 2 == 0:
                circuit.h(0)
            circuit.measure(0, 0)
            simulator = AerSimulator()
            result = simulator.run(circuit, shots=1).result()
            counts = result.get_counts()
            return 1 if "1" in counts else 0

        return int(sha256(seed.encode("utf-8")).hexdigest(), 16) % 2


class QuantumChemistryModule:
    """Basic molecular parser returning composition statistics."""

    def analyze_molecule(self, molecule: str) -> str:
        text = molecule.strip()
        if not text:
            raise ValueError("Molecule string cannot be empty.")

        elements: Dict[str, int] = {}
        token = ""
        count = ""
        for ch in text:
            if ch.isupper():
                if token:
                    elements[token] = elements.get(token, 0) + int(count or 1)
                token, count = ch, ""
            elif ch.islower():
                if not token:
                    raise ValueError(f"Invalid molecule format: {molecule!r}")
                token += ch
            elif ch.isdigit():
                if not token:
                    raise ValueError(f"Invalid molecule format: {molecule!r}")
                count += ch
            else:
                raise ValueError(f"Unsupported molecule character: {ch!r}")

        if token:
            elements[token] = elements.get(token, 0) + int(count or 1)

        atoms = sum(elements.values())
        return f"elements={elements}, total_atoms={atoms}"


class QuantumEmotionalModel:
    def __init__(self) -> None:
        self.love_level = 0.0
        self.fear_level = 0.0

    def handle_input(self, user_input: str) -> EmotionReading:
        polarity = self._polarity(user_input)
        self.love_level = polarity
        self.fear_level = -polarity
        return EmotionReading(self.love_level, self.fear_level)

    def _polarity(self, text: str) -> float:
        if TextBlob is not None:
            return float(TextBlob(text).sentiment.polarity)

        positive = {"love", "amazing", "great", "good", "excellent", "happy"}
        negative = {"hate", "terrible", "bad", "awful", "sad"}
        words = set(text.lower().replace("'", "").split())
        score = len(words & positive) - len(words & negative)
        return max(-1.0, min(1.0, score / 2.0))

    def plot_emotions(self, inputs: Sequence[str], love_levels: Sequence[float], fear_levels: Sequence[float], *, output_path: Optional[str] = None) -> Optional[str]:
        if plt is None:
            return None

        fig = plt.figure(figsize=(8, 4))
        plt.plot(inputs, love_levels, marker="o", label="Love Level")
        plt.plot(inputs, fear_levels, marker="o", label="Fear Level")
        plt.xlabel("Inputs")
        plt.ylabel("Levels")
        plt.title("Love and Fear Levels Over Time")
        plt.legend()
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path)
            plt.close(fig)
            return output_path

        plt.show()
        plt.close(fig)
        return None


class MentalMatrixAI:
    def __init__(self) -> None:
        self.astrology_module = AstrologyModule()
        self.quantum_bio_module = QuantumBioinformaticsModule()
        self.quantum_chem_module = QuantumChemistryModule()
        self.quantum_emotion_model = QuantumEmotionalModel()

    def make_decision(self, user_birthdate: datetime, dna_sequence: str, molecule: str, user_input: str) -> str:
        zodiac_sign = self.astrology_module.get_zodiac_sign(user_birthdate)
        planetary_positions = self.astrology_module.get_planetary_positions()
        dna_analysis = self.quantum_bio_module.analyze_dna_sequence(dna_sequence)
        molecule_analysis = self.quantum_chem_module.analyze_molecule(molecule)
        self.quantum_emotion_model.handle_input(user_input)

        return (
            "Decision based on: "
            f"{zodiac_sign}, {planetary_positions}, {dna_analysis}, {molecule_analysis}, "
            f"emotion=({self.quantum_emotion_model.love_level:.2f},{self.quantum_emotion_model.fear_level:.2f})"
        )


def run_demo() -> List[str]:
    ai_core = MentalMatrixAI()
    user_birthdate = datetime.strptime("1988-11-22", "%Y-%m-%d")
    dna_sequence = "ATCGATCGATCG"
    molecule = "H2O"
    user_inputs = ["I love you", "I hate this", "You're amazing", "This is terrible"]

    decisions: List[str] = []
    love_levels: List[float] = []
    fear_levels: List[float] = []
    for user_input in user_inputs:
        decision = ai_core.make_decision(user_birthdate, dna_sequence, molecule, user_input)
        decisions.append(decision)
        love_levels.append(ai_core.quantum_emotion_model.love_level)
        fear_levels.append(ai_core.quantum_emotion_model.fear_level)

    ai_core.quantum_emotion_model.plot_emotions(user_inputs, love_levels, fear_levels)
    return decisions


if __name__ == "__main__":
    for line in run_demo():
        print(line)
