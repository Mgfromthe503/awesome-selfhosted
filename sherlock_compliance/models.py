from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompliancePolicy:
    allowed_purposes: tuple[str, ...]
    allowed_source_types: tuple[str, ...]
    require_case_id: bool
    require_legal_basis: bool
    require_supervisor_approval_for_sensitive: bool


@dataclass(frozen=True)
class AccessRequest:
    actor_id: str
    role: str
    purpose: str
    case_id: str | None
    legal_basis: str | None
    sensitive_action: bool
    supervisor_approved: bool


@dataclass(frozen=True)
class OsintSignal:
    lead_id: str
    source_type: str
    recency_hours: float
    confidence: float
    relevance: float


@dataclass(frozen=True)
class LeadScore:
    lead_id: str
    score: float
    explanation: str
