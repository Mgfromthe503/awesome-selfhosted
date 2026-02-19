from __future__ import annotations

from collections import defaultdict

from .models import LeadScore, OsintSignal


def _time_decay(recency_hours: float) -> float:
    clamped = max(0.0, recency_hours)
    return 1.0 / (1.0 + clamped / 24.0)


def prioritize_leads(signals: list[OsintSignal], allowed_source_types: tuple[str, ...]) -> list[LeadScore]:
    """Rank leads from legal/approved OSINT signals only.

    This function intentionally excludes private-contact scraping, credential abuse,
    or unauthorized surveillance sources.
    """
    allowed = set(allowed_source_types)
    totals: dict[str, float] = defaultdict(float)
    reasons: dict[str, list[str]] = defaultdict(list)

    for s in signals:
        if s.source_type not in allowed:
            continue
        confidence = min(max(s.confidence, 0.0), 1.0)
        relevance = min(max(s.relevance, 0.0), 1.0)
        signal_score = confidence * relevance * _time_decay(s.recency_hours)
        totals[s.lead_id] += signal_score
        reasons[s.lead_id].append(
            f"{s.source_type}:conf={confidence:.2f},rel={relevance:.2f},age_h={s.recency_hours:.1f}"
        )

    ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    return [
        LeadScore(lead_id=lead_id, score=round(score, 6), explanation="; ".join(reasons[lead_id]))
        for lead_id, score in ranked
    ]
