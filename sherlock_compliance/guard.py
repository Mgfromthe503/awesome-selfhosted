from __future__ import annotations

from .models import AccessRequest, CompliancePolicy


class AccessDenied(PermissionError):
    pass


class AuthorizationGuard:
    def __init__(self, policy: CompliancePolicy):
        self.policy = policy

    def authorize(self, req: AccessRequest) -> None:
        if req.purpose not in self.policy.allowed_purposes:
            raise AccessDenied(f"purpose not allowed: {req.purpose}")

        if self.policy.require_case_id and not (req.case_id and req.case_id.strip()):
            raise AccessDenied("missing required case_id")

        if self.policy.require_legal_basis and not (req.legal_basis and req.legal_basis.strip()):
            raise AccessDenied("missing required legal_basis")

        if req.sensitive_action and self.policy.require_supervisor_approval_for_sensitive:
            if not req.supervisor_approved:
                raise AccessDenied("supervisor approval required for sensitive action")
