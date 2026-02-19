"""Sherlock workflow inspired by the user's Wolfram-style prototype.

This module intentionally uses real Python dependencies when available and
provides deterministic fallbacks so tests can run in constrained environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, SparsePauliOp

    _HAS_QISKIT = True
except Exception:  # pragma: no cover - optional dependency
    QuantumCircuit = Statevector = SparsePauliOp = None
    _HAS_QISKIT = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None
    _HAS_SKLEARN = False

PRINCIPLES: List[Dict[str, Any]] = [
    {"PrincipleID": 1, "Category": "Mentalism", "Principle": "All is mind"},
    {"PrincipleID": 2, "Category": "Correspondence", "Principle": "As above so below"},
    {"PrincipleID": 3, "Category": "Vibration", "Principle": "Everything vibrates"},
    {"PrincipleID": 4, "Category": "Polarity", "Principle": "Everything has two poles"},
    {"PrincipleID": 5, "Category": "Rhythm", "Principle": "Everything flows in cycles"},
    {"PrincipleID": 6, "Category": "Cause & Effect", "Principle": "Nothing escapes law"},
    {"PrincipleID": 7, "Category": "Gender", "Principle": "Masculine & feminine on every plane"},
    {"PrincipleID": 8, "Category": "Attraction", "Principle": "Like energy attracts like"},
    {
        "PrincipleID": 9,
        "Category": "Perpetual Transmutation",
        "Principle": "Energy constantly transforms",
    },
    {"PrincipleID": 10, "Category": "Compensation", "Principle": "Balance through equivalence"},
    {"PrincipleID": 11, "Category": "Relativity", "Principle": "Truth is comparative"},
    {"PrincipleID": 12, "Category": "Divine Oneness", "Principle": "All is connected"},
]


def _pad_or_trim(values: List[float], size: int = 12) -> List[float]:
    if len(values) >= size:
        return values[:size]
    return values + [0.0] * (size - len(values))


def bio_vector(seq: str) -> List[float]:
    """Create a 12-dimensional sequence embedding.

    - Preferred path: TF-IDF char ngrams using scikit-learn.
    - Fallback path: normalized base-count and k-mer features.
    """

    if _HAS_SKLEARN:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
        arr = vec.fit_transform([seq]).toarray()[0].tolist()
        return _pad_or_trim([float(v) for v in arr], 12)

    seq = seq.upper().strip()
    total = max(len(seq), 1)
    counts = [seq.count("A") / total, seq.count("C") / total, seq.count("G") / total, seq.count("T") / total]
    # Add simple rolling bigram hash buckets for deterministic shape.
    buckets = [0.0] * 8
    for i in range(len(seq) - 1):
        pair = seq[i : i + 2]
        buckets[hash(pair) % 8] += 1.0
    if len(seq) > 1:
        buckets = [b / (len(seq) - 1) for b in buckets]
    return counts + buckets


def bias_classify(text: str) -> Dict[str, float]:
    """Tiny lexicon-based toxicity proxy for deterministic offline use."""

    toxic_lexicon = {"terrible", "idiot", "hate", "stupid", "awful"}
    words = [w.strip(".,!?;:\"'").lower() for w in text.split() if w.strip()]
    if not words:
        return {"non_toxic": 1.0, "toxic": 0.0}
    toxic_hits = sum(1 for w in words if w in toxic_lexicon)
    toxic = min(1.0, toxic_hits / len(words))
    return {"non_toxic": round(1.0 - toxic, 4), "toxic": round(toxic, 4)}


def privacy_hash(file_path: str | Path) -> str:
    payload = Path(file_path).read_bytes()
    return sha256(payload).hexdigest()


@dataclass
class DNAState:
    principles: List[Dict[str, Any]] = field(default_factory=list)
    quantum_state: Any = field(default="|0>")

    @staticmethod
    def init_quantum_state() -> Any:
        if _HAS_QISKIT:
            return Statevector.from_label("0")
        return "|0>"

    @classmethod
    def create(cls) -> "DNAState":
        return cls(principles=[], quantum_state=cls.init_quantum_state())

    def integrate_principle(self, principle: Dict[str, Any]) -> None:
        self.principles.append(principle)
        self.quantum_state = self.quantum_update(principle)

    def quantum_update(self, principle: Dict[str, Any]) -> Any:
        code = principle.get("Code")
        if _HAS_QISKIT and code in (3, 4):
            qc = QuantumCircuit(1)
            if code == 3:
                qc.h(0)
            elif code == 4:
                qc.x(0)
            return self.quantum_state.evolve(qc)
        return self.quantum_state

    def analyze_data(self, data: str) -> Dict[str, Any]:
        return {
            "QuantumAnalysis": self.quantum_expectation_z(),
            "DataEcho": data,
        }

    def quantum_expectation_z(self) -> float:
        if _HAS_QISKIT:
            z = SparsePauliOp.from_list([("Z", 1.0)])
            return float(self.quantum_state.expectation_value(z).real)
        return 1.0 if self.quantum_state == "|0>" else 0.0


@dataclass
class Sherlock:
    dna: DNAState = field(default_factory=DNAState.create)

    def investigate(self, data: str) -> Dict[str, Any]:
        return {
            "Vector": bio_vector(data),
            "Principles": self.dna.principles,
        }

    @staticmethod
    def report(findings: Dict[str, Any]) -> str:
        return "\n".join(f"{k}\t{v}" for k, v in findings.items())

    def seek_truth(self, data: str) -> Dict[str, Any]:
        return self.dna.analyze_data(data)

    @staticmethod
    def bias_audit(text: str) -> Dict[str, float]:
        return bias_classify(text)

    @staticmethod
    def privacy_audit(file_path: str | Path) -> str:
        return privacy_hash(file_path)


def run_demo() -> Dict[str, Any]:
    sher = Sherlock()
    vib = {"Code": 3, "Name": "Vibration", "Description": "Everything vibrates"}
    sher.dna.integrate_principle(vib)

    dna_snippet = "ATGCTCGGATCAGT"
    findings = sher.investigate(dna_snippet)
    bias_result = sher.bias_audit("You are a terrible person!")

    report = sher.report(
        {
            "Investigation": findings,
            "BiasCheck": bias_result,
            "Quantum<Z>": sher.dna.quantum_state,
        }
    )

    return {
        "findings": findings,
        "bias": bias_result,
        "truth": sher.seek_truth(dna_snippet),
        "report": report,
    }
