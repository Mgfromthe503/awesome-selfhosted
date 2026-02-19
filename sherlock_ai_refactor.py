"""Refactored SherlockAI example with zero third-party dependencies."""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

_WORD_RE = re.compile(r"[a-zA-Z']+")


@dataclass(frozen=True)
class Block:
    previous_hash: str
    transaction: Sequence[str]

    @property
    def block_hash(self) -> str:
        payload = "".join(self.transaction) + self.previous_hash
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SherlockAI:
    """Simple Naive Bayes text classifier for binary labels 0/1."""

    def __init__(self, name: str, owner: str, birthday: _dt.date):
        self.name = name
        self.owner = owner
        self.birthday = birthday
        self._is_trained = False
        self._class_counts: Counter[int] = Counter()
        self._token_counts: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}
        self._token_totals: dict[int, int] = {0: 0, 1: 0}
        self._vocab: set[str] = set()

    def get_age(self) -> int:
        return (_dt.date.today() - self.birthday).days // 365

    def _tokens(self, value: str) -> list[str]:
        return [t.lower() for t in _WORD_RE.findall(str(value))]

    def process_data(self, data: Iterable[str]) -> list[list[float]]:
        rows: list[list[float]] = []
        for value in data:
            tokens = self._tokens(value)
            token_count = len(tokens)
            unique_count = len(set(tokens))
            avg_len = float(sum(map(len, tokens)) / token_count) if token_count else 0.0
            rows.append([float(token_count), float(unique_count), avg_len])
        return rows

    def train_model(self, data: Sequence[str], labels: Sequence[int]) -> float:
        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length.")
        if len(data) < 2:
            raise ValueError("Need at least two samples to train.")

        self._class_counts.clear()
        self._token_counts = {0: Counter(), 1: Counter()}
        self._token_totals = {0: 0, 1: 0}
        self._vocab.clear()

        for text, label in zip(data, labels):
            if label not in (0, 1):
                raise ValueError("Only binary labels 0 and 1 are supported.")
            self._class_counts[label] += 1
            tokens = self._tokens(text)
            self._token_counts[label].update(tokens)
            self._token_totals[label] += len(tokens)
            self._vocab.update(tokens)

        self._is_trained = True

        predictions = self.make_decision(data)
        correct = sum(int(p == y) for p, y in zip(predictions, labels))
        return correct / len(labels)

    def _score(self, tokens: Sequence[str], label: int) -> float:
        vocab_size = max(1, len(self._vocab))
        total_docs = sum(self._class_counts.values())
        prior = (self._class_counts[label] + 1) / (total_docs + 2)
        score = math.log(prior)

        token_total = self._token_totals[label]
        for token in tokens:
            likelihood = (self._token_counts[label][token] + 1) / (token_total + vocab_size)
            score += math.log(likelihood)
        return score

    def make_decision(self, data: Sequence[str]) -> list[int]:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before inference.")

        outputs: list[int] = []
        for text in data:
            tokens = self._tokens(text)
            score_0 = self._score(tokens, 0)
            score_1 = self._score(tokens, 1)
            outputs.append(1 if score_1 >= score_0 else 0)
        return outputs


if __name__ == "__main__":
    assistant = SherlockAI("Sherlock", "Watson", _dt.date(2020, 1, 1))
    samples = [
        "alice sends 1 btc to bob",
        "urgent transfer now",
        "monthly payroll transfer approved",
        "free money click now",
    ]
    labels = [0, 1, 0, 1]
    accuracy = assistant.train_model(samples, labels)
    print(f"Training accuracy: {accuracy:.3f}")
    print("Predictions:", assistant.make_decision(["alice transfer now", "payroll processed"]))
