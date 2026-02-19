from pathlib import Path

import pytest

from sherlock_compliance import AccessDenied, AuthorizationGuard, load_policy
from sherlock_compliance.models import AccessRequest


def test_authorization_allows_valid_request():
    policy = load_policy(Path("configs/compliance_policy.json"))
    guard = AuthorizationGuard(policy)
    req = AccessRequest(
        actor_id="analyst_01",
        role="analyst",
        purpose="missing_person_search",
        case_id="CASE-2026-0001",
        legal_basis="Court order 26-1001",
        sensitive_action=True,
        supervisor_approved=True,
    )
    guard.authorize(req)


def test_authorization_denies_missing_legal_basis():
    policy = load_policy(Path("configs/compliance_policy.json"))
    guard = AuthorizationGuard(policy)
    req = AccessRequest(
        actor_id="analyst_01",
        role="analyst",
        purpose="missing_person_search",
        case_id="CASE-2026-0002",
        legal_basis="",
        sensitive_action=False,
        supervisor_approved=False,
    )
    with pytest.raises(AccessDenied):
        guard.authorize(req)
