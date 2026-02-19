"""MM language parser/executor for Sherlock.

MM language is a lightweight command language inspired by Hermetic principles.
It is designed for deterministic data generation and detective-style analysis tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_HERMETIC = {
    "Correspondence": "As above, so below; as within, so without.",
    "Vibration": "Nothing rests; everything moves; everything vibrates.",
    "Polarity": "Everything has its opposite; opposites differ in degree.",
    "Rhythm": "Everything flows in cycles and patterns.",
    "Cause & Effect": "Every cause has its effect; every effect has its cause.",
    "Gender": "Masculine and feminine principles are present in everything.",
}


@dataclass
class MMInstruction:
    op: str
    payload: dict[str, Any]


class MMLanguageError(ValueError):
    pass


def anomaly_detection(data: list[float]) -> list[bool]:
    if not data:
        return []
    avg = sum(data) / len(data)
    var = sum((x - avg) ** 2 for x in data) / len(data)
    sigma = var ** 0.5 or 1.0
    return [abs(value - avg) > (2 * sigma) for value in data]


def _parse_detect(line: str) -> MMInstruction:
    match = re.match(r"^DETECT\s+(.+?)\s+IN\s+(.+)$", line, flags=re.IGNORECASE)
    if not match:
        raise MMLanguageError(f"Invalid DETECT syntax: {line}")
    return MMInstruction("DETECT", {"pattern": match.group(1).strip(), "text": match.group(2).strip()})


def _parse_anomaly(line: str) -> MMInstruction:
    match = re.match(r"^ANOMALY\s+SERIES\s*:\s*(.+)$", line, flags=re.IGNORECASE)
    if not match:
        raise MMLanguageError(f"Invalid ANOMALY syntax: {line}")
    raw = [p.strip() for p in match.group(1).split(",") if p.strip()]
    try:
        series = [float(x) for x in raw]
    except ValueError as exc:
        raise MMLanguageError(f"Invalid numeric value in ANOMALY series: {line}") from exc
    return MMInstruction("ANOMALY", {"series": series})


def _parse_tag(line: str) -> MMInstruction:
    match = re.match(r"^TAG\s+([A-Za-z0-9_\-]+)\s*=\s*(.+)$", line, flags=re.IGNORECASE)
    if not match:
        raise MMLanguageError(f"Invalid TAG syntax: {line}")
    return MMInstruction("TAG", {"key": match.group(1), "value": match.group(2).strip()})


def _parse_hermetic(line: str) -> MMInstruction:
    principle = line.replace("🔮", "", 1).strip()
    if principle not in _HERMETIC:
        raise MMLanguageError(f"Unknown Hermetic principle: {principle}")
    return MMInstruction("HERMETIC", {"principle": principle, "meaning": _HERMETIC[principle]})


def parse_mm_script(script: str) -> list[MMInstruction]:
    if script is None:
        raise MMLanguageError("Script cannot be None")

    instructions: list[MMInstruction] = []
    for raw in script.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("🔮"):
            instructions.append(_parse_hermetic(line))
            continue

        upper = line.upper()
        if upper.startswith("DETECT "):
            instructions.append(_parse_detect(line))
        elif upper.startswith("ANOMALY SERIES"):
            instructions.append(_parse_anomaly(line))
        elif upper.startswith("TAG "):
            instructions.append(_parse_tag(line))
        else:
            raise MMLanguageError(f"Unknown MM instruction: {line}")

    return instructions


def execute_mm_instructions(instructions: list[MMInstruction]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    tags: dict[str, str] = {}

    for ins in instructions:
        if ins.op == "HERMETIC":
            results.append({"op": ins.op, "result": ins.payload})
        elif ins.op == "DETECT":
            pattern = ins.payload["pattern"]
            text = ins.payload["text"]
            found = pattern.lower() in text.lower()
            results.append({"op": ins.op, "result": {"pattern": pattern, "found": found}})
        elif ins.op == "ANOMALY":
            series = ins.payload["series"]
            flags = anomaly_detection(series)
            indexes = [idx for idx, flag in enumerate(flags) if flag]
            results.append({"op": ins.op, "result": {"series": series, "anomaly_indexes": indexes}})
        elif ins.op == "TAG":
            tags[str(ins.payload["key"])] = str(ins.payload["value"])
            results.append({"op": ins.op, "result": {"tags": dict(tags)}})
        else:
            raise MMLanguageError(f"Unsupported op: {ins.op}")

    return results


def run_mm_script(script: str) -> list[dict[str, Any]]:
    return execute_mm_instructions(parse_mm_script(script))
