from pathlib import Path

from sherlock_compliance import AuditLogger


def test_audit_hash_chain(tmp_path: Path):
    log = AuditLogger(tmp_path / "audit.jsonl")
    e1 = log.log("search", "analyst_1", {"case_id": "C1"})
    e2 = log.log("rank", "analyst_1", {"case_id": "C1"})
    assert e1["event_hash"]
    assert e2["prev_hash"] == e1["event_hash"]
