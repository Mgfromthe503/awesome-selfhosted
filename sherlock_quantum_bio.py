"""Sherlock DNA system prototype inspired by the provided Wolfram-style pseudocode.

This module provides:
- canonical 12-principle dataset
- a minimal quantum-state abstraction with principle-driven updates
- a DNA feature vector extractor that returns a 12-dimensional embedding
- bias and privacy audit helpers
- Sherlock/DNA system constructors mirroring the user workflow
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List


PRINCIPLES: List[Dict[str, Any]] = [
    {"PrincipleID": 1, "Category": "Mentalism", "Principle": "All is mind"},
    {"PrincipleID": 2, "Category": "Correspondence", "Principle": "As above so below"},
    {"PrincipleID": 3, "Category": "Vibration", "Principle": "Everything vibrates"},
    {"PrincipleID": 4, "Category": "Polarity", "Principle": "Everything has two poles"},
    {"PrincipleID": 5, "Category": "Rhythm", "Principle": "Everything flows in cycles"},
    {"PrincipleID": 6, "Category": "Cause & Effect", "Principle": "Nothing escapes law"},
    {"PrincipleID": 7, "Category": "Gender", "Principle": "Masculine & feminine on every plane"},
    {"PrincipleID": 8, "Category": "Attraction", "Principle": "Like energy attracts like"},
    {"PrincipleID": 9, "Category": "Perpetual Transmutation", "Principle": "Energy constantly transforms"},
    {"PrincipleID": 10, "Category": "Compensation", "Principle": "Balance through equivalence"},
    {"PrincipleID": 11, "Category": "Relativity", "Principle": "Truth is comparative"},
    {"PrincipleID": 12, "Category": "Divine Oneness", "Principle": "All is connected"},
]


@dataclass
class QuantumState:
    """Tiny 1-qubit toy model.

    basis: "0", "1", or "superposition".
    """

    basis: str = "0"



def quantum_state_init() -> QuantumState:
    return QuantumState("0")



def quantum_update(qs: QuantumState, principle: Dict[str, Any]) -> QuantumState:
    principle_id = principle.get("Code", principle.get("PrincipleID"))

    if principle_id == 3:
        # Hadamard-like effect
        return QuantumState("superposition")
    if principle_id == 4:
        # Pauli-X-like bit flip
        return QuantumState("1" if qs.basis == "0" else "0")
    return qs



def bio_vector(seq: str) -> List[float]:
    """Return a 12-dimensional normalized DNA embedding."""

    seq = (seq or "").strip().upper()
    total = len(seq) or 1
    a = seq.count("A") / total
    c = seq.count("C") / total
    g = seq.count("G") / total
    t = seq.count("T") / total

    gc = (seq.count("G") + seq.count("C")) / total
    at = (seq.count("A") + seq.count("T")) / total

    dinucs = ["AA", "AT", "AG", "AC", "TA", "TT"]
    dinuc_counts = [sum(1 for i in range(max(0, len(seq) - 1)) if seq[i : i + 2] == d) / total for d in dinucs]

    vec = [a, c, g, t, gc, at, len(seq) / 100.0, sum(ch not in "ACGT" for ch in seq) / total]
    vec.extend(dinuc_counts[:4])
    return vec[:12]


TOXIC_WORDS = {"terrible", "idiot", "hate", "stupid", "awful"}


def bias_classify(text: str) -> Dict[str, float]:
    tokens = [t.strip(".,!?;:()[]{}\"'").lower() for t in (text or "").split()]
    if not tokens:
        return {"toxic": 0.0, "non_toxic": 1.0}
    toxic_hits = sum(1 for t in tokens if t in TOXIC_WORDS)
    toxic_prob = min(1.0, toxic_hits / max(1, len(tokens)))
    return {"toxic": toxic_prob, "non_toxic": 1.0 - toxic_prob}



def privacy_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass
class DNASystem:
    principles: List[Dict[str, Any]] = field(default_factory=list)
    quantum_state: QuantumState = field(default_factory=quantum_state_init)

    def integrate_principle(self, principle: Dict[str, Any]) -> None:
        self.principles.append(principle)
        self.quantum_state = quantum_update(self.quantum_state, principle)

    def analyze_data(self, data: Any) -> Dict[str, Any]:
        expectation_z = 0.0 if self.quantum_state.basis == "superposition" else (1.0 if self.quantum_state.basis == "0" else -1.0)
        return {"QuantumAnalysis": expectation_z, "DataEcho": data}


@dataclass
class Sherlock:
    dna: DNASystem = field(default_factory=DNASystem)

    def investigate(self, sequence: str) -> Dict[str, Any]:
        return {"Vector": bio_vector(sequence), "Principles": list(self.dna.principles)}

    def report(self, findings: Dict[str, Any]) -> str:
        return "\n".join(f"{k}\t{v}" for k, v in findings.items())

    def seek_truth(self, data: Any) -> Dict[str, Any]:
        return self.dna.analyze_data(data)

    def bias_audit(self, text: str) -> Dict[str, float]:
        return bias_classify(text)

    def privacy_audit(self, content: bytes) -> str:
        return privacy_hash(content)
