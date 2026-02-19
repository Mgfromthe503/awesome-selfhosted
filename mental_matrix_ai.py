from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict


@dataclass(frozen=True)
class DecisionContext:
    zodiac_sign: str
    planetary_positions: Dict[str, str]
    dna_analysis: str
    molecule_analysis: str


class AstrologyModule:
    """Astrology helpers backed by deterministic calculations."""

    _ZODIAC_WINDOWS = (
        ((1, 20), "Aquarius"),
        ((2, 19), "Pisces"),
        ((3, 21), "Aries"),
        ((4, 20), "Taurus"),
        ((5, 21), "Gemini"),
        ((6, 21), "Cancer"),
        ((7, 23), "Leo"),
        ((8, 23), "Virgo"),
        ((9, 23), "Libra"),
        ((10, 23), "Scorpio"),
        ((11, 22), "Sagittarius"),
        ((12, 22), "Capricorn"),
    )

    def get_zodiac_sign(self, birthdate: datetime) -> str:
        month_day = (birthdate.month, birthdate.day)
        current_sign = "Capricorn"
        for transition, sign in self._ZODIAC_WINDOWS:
            if month_day >= transition:
                current_sign = sign
        return current_sign

    def get_planetary_positions(self, now: datetime | None = None) -> Dict[str, str]:
        """Pseudo-live positions derived from current UTC time for reproducible tests."""
        if now is None:
            now = datetime.now(timezone.utc)

        signs = [
            "Aries",
            "Taurus",
            "Gemini",
            "Cancer",
            "Leo",
            "Virgo",
            "Libra",
            "Scorpio",
            "Sagittarius",
            "Capricorn",
            "Aquarius",
            "Pisces",
        ]
        return {
            "Mars": signs[now.hour % 12],
            "Venus": signs[now.minute % 12],
            "Mercury": signs[now.day % 12],
        }


class QuantumBioinformaticsModule:
    def analyze_dna_sequence(self, dna_sequence: str) -> str:
        if not dna_sequence:
            return "DNA sequence is empty"

        sequence = dna_sequence.upper()
        valid_nucleotides = set("ATCG")
        invalid = sorted({char for char in sequence if char not in valid_nucleotides})
        if invalid:
            return f"Invalid nucleotides: {''.join(invalid)}"

        gc_content = (sequence.count("G") + sequence.count("C")) / len(sequence)
        return f"Length={len(sequence)}, GC={gc_content:.2%}"


class QuantumChemistryModule:
    def analyze_molecule(self, molecule: str) -> str:
        if not molecule.strip():
            return "Molecule formula is empty"

        elements = {}
        token = ""
        count = ""
        for char in molecule:
            if char.isalpha() and char.isupper():
                if token:
                    elements[token] = elements.get(token, 0) + int(count or "1")
                token, count = char, ""
            elif char.isalpha() and char.islower():
                token += char
            elif char.isdigit():
                count += char
        if token:
            elements[token] = elements.get(token, 0) + int(count or "1")

        atoms = sum(elements.values())
        composition = ", ".join(f"{elem}:{num}" for elem, num in sorted(elements.items()))
        return f"Atoms={atoms}; Composition={composition}"


class MentalMatrixAI:
    def __init__(self) -> None:
        self.astrology_module = AstrologyModule()
        self.quantum_bio_module = QuantumBioinformaticsModule()
        self.quantum_chem_module = QuantumChemistryModule()

    def make_decision(self, user_birthdate: datetime, dna_sequence: str, molecule: str) -> str:
        context = DecisionContext(
            zodiac_sign=self.astrology_module.get_zodiac_sign(user_birthdate),
            planetary_positions=self.astrology_module.get_planetary_positions(),
            dna_analysis=self.quantum_bio_module.analyze_dna_sequence(dna_sequence),
            molecule_analysis=self.quantum_chem_module.analyze_molecule(molecule),
        )

        return (
            "Decision based on: "
            f"{context.zodiac_sign}, "
            f"{context.planetary_positions}, "
            f"{context.dna_analysis}, "
            f"{context.molecule_analysis}"
        )


if __name__ == "__main__":
    ai_core = MentalMatrixAI()
    user_birthdate = datetime.strptime("1988-11-22", "%Y-%m-%d")
    dna_sequence = "ATCGATCGATCG"
    molecule = "H2O"

    print(ai_core.make_decision(user_birthdate, dna_sequence, molecule))
