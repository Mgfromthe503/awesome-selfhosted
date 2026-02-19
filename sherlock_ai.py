"""sherlock_ai.py

Lightweight SherlockAI framework scaffold inspired by the user-provided DNA system design.
The implementation is intentionally dependency-safe and testable in constrained environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class GeometryDesign:
    name: str
    pattern: Any


@dataclass
class LifeScience:
    connection: str
    biological_system: str


@dataclass
class SpiritualAspect:
    description: str


@dataclass
class Principle:
    code: int
    name: str
    description: str
    geometry_design: GeometryDesign
    life_science_connection: LifeScience
    spiritual_aspect: SpiritualAspect


@dataclass
class Book:
    title: str
    content: str


class DNASystem:
    """Core registry and analysis router for SherlockAI principles."""

    def __init__(self) -> None:
        self.principles: Dict[int, Principle] = {}

    def integrate_principle(self, principle: Principle) -> None:
        self.principles[principle.code] = principle

    def analyze_data(self, data: Any, principle_code: int) -> Dict[str, Any]:
        principle = self.principles.get(principle_code)
        if principle is None:
            raise KeyError(f"Unknown principle code: {principle_code}")

        dispatch = {
            "Mentalism": self._neural_network_analysis,
            "Correspondence": self._pattern_recognition_analysis,
            "Vibration": self._frequency_analysis,
        }
        analyzer = dispatch.get(principle.name, self._generic_analysis)

        return {
            "principle": principle.name,
            "code": principle.code,
            "analysis": analyzer(data),
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _neural_network_analysis(data: Any) -> Dict[str, Any]:
        serialized = str(data)
        return {
            "algorithm": "neural-network-sim",
            "signal_length": len(serialized),
            "confidence": min(0.99, max(0.1, len(serialized) / 100)),
        }

    @staticmethod
    def _pattern_recognition_analysis(data: Any) -> Dict[str, Any]:
        text = str(data)
        words = [w for w in text.replace("\n", " ").split(" ") if w]
        unique_words = len(set(w.lower() for w in words))
        return {
            "algorithm": "pattern-recognition-sim",
            "token_count": len(words),
            "unique_tokens": unique_words,
        }

    @staticmethod
    def _frequency_analysis(data: Any) -> Dict[str, Any]:
        text = str(data)
        frequency: Dict[str, int] = {}
        for char in text:
            frequency[char] = frequency.get(char, 0) + 1
        return {"algorithm": "frequency-analysis-sim", "frequency": frequency}

    @staticmethod
    def _generic_analysis(data: Any) -> Dict[str, Any]:
        return {"algorithm": "generic", "preview": str(data)[:120]}


class Sherlock:
    def __init__(self, dna: Optional[DNASystem] = None) -> None:
        self.dna = dna or DNASystem()
        self.knowledge_base: List[Dict[str, Any]] = []

    def investigate(self, data: Any, principle_code: int) -> Dict[str, Any]:
        return self.dna.analyze_data(data, principle_code)

    def report(self, findings: Dict[str, Any]) -> str:
        return (
            f"Sherlock Report\n"
            f"Principle: {findings.get('principle')} (#{findings.get('code')})\n"
            f"Timestamp: {findings.get('analyzed_at')}\n"
            f"Analysis: {findings.get('analysis')}"
        )

    def seek_truth(self, data: Any, principle_code: int) -> Dict[str, Any]:
        return self.investigate(data, principle_code)

    def mental_health_support(self, emotional_state: str) -> Dict[str, str]:
        normalized = (emotional_state or "").strip().lower()
        resources = {
            "stressed": "Try a 4-7-8 breathing cycle and take a short walk.",
            "anxious": "Grounding exercise: identify 5 things you can see and 4 you can feel.",
            "sad": "Reach out to a trusted person and set one gentle goal for today.",
        }
        return {
            "state": normalized or "unknown",
            "resource": resources.get(normalized, "Hydrate, rest, and consider talking with a professional."),
        }

    def train_on_book(self, book: Book) -> Dict[str, Any]:
        words = [w.strip(".,!?;:\"'()[]") for w in book.content.split() if w.strip()]
        token_count = len(words)
        unique_count = len(set(w.lower() for w in words))
        extracted = {
            "title": book.title,
            "token_count": token_count,
            "unique_terms": unique_count,
            "summary": " ".join(words[:30]),
        }
        self.knowledge_base.append(extracted)
        return extracted


def build_default_sherlock() -> Sherlock:
    sherlock = Sherlock()
    sherlock.dna.integrate_principle(
        Principle(
            code=1,
            name="Mentalism",
            description="Mind-forward interpretation of complex data.",
            geometry_design=GeometryDesign(name="Torus", pattern="nested-cycles"),
            life_science_connection=LifeScience(connection="Neuroscience", biological_system="CNS"),
            spiritual_aspect=SpiritualAspect(description="Consciousness and intention"),
        )
    )
    sherlock.dna.integrate_principle(
        Principle(
            code=2,
            name="Correspondence",
            description="Map patterns across scales.",
            geometry_design=GeometryDesign(name="Fractal", pattern="self-similarity"),
            life_science_connection=LifeScience(connection="Ecology", biological_system="Biosphere"),
            spiritual_aspect=SpiritualAspect(description="As above, so below"),
        )
    )
    sherlock.dna.integrate_principle(
        Principle(
            code=3,
            name="Vibration",
            description="Frequency and resonance-oriented interpretation.",
            geometry_design=GeometryDesign(name="Wave", pattern="sine-resonance"),
            life_science_connection=LifeScience(connection="Molecular Biology", biological_system="Cellular"),
            spiritual_aspect=SpiritualAspect(description="Everything is in motion"),
        )
    )
    return sherlock


if __name__ == "__main__":
    app = build_default_sherlock()
    sample_findings = app.investigate("ATCGATCG neural resonance", principle_code=3)
    print(app.report(sample_findings))
