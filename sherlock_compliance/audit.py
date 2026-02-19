from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    """Append-only JSONL audit logger with hash chaining."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return "GENESIS"
        lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return "GENESIS"
        try:
            obj = json.loads(lines[-1])
        except json.JSONDecodeError:
            return "GENESIS"
        return str(obj.get("event_hash", "GENESIS"))

    def log(self, action: str, actor_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        prev_hash = self._last_hash()
        body = {
            "timestamp_utc": ts,
            "action": action,
            "actor_id": actor_id,
            "payload": payload,
            "prev_hash": prev_hash,
        }
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        event = dict(body)
        event["event_hash"] = digest

        with self.path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(event, ensure_ascii=True) + "\n")

        return event
