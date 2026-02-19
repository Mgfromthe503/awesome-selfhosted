from __future__ import annotations

import json
from pathlib import Path

from .models import CompliancePolicy


def load_policy(path: Path) -> CompliancePolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CompliancePolicy(
        allowed_purposes=tuple(payload["allowed_purposes"]),
        allowed_source_types=tuple(payload["allowed_source_types"]),
        require_case_id=bool(payload.get("require_case_id", True)),
        require_legal_basis=bool(payload.get("require_legal_basis", True)),
        require_supervisor_approval_for_sensitive=bool(
            payload.get("require_supervisor_approval_for_sensitive", True)
        ),
    )
