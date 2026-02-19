from sherlock_compliance import prioritize_leads
from sherlock_compliance.models import OsintSignal


def test_prioritize_leads_filters_disallowed_sources_and_ranks():
    signals = [
        OsintSignal("lead_a", "public_records", 4, 0.9, 0.8),
        OsintSignal("lead_a", "open_social_posts", 2, 0.7, 0.7),
        OsintSignal("lead_b", "tip_line", 1, 0.8, 0.9),
        OsintSignal("lead_b", "private_phone_contacts", 1, 1.0, 1.0),
    ]

    out = prioritize_leads(
        signals,
        allowed_source_types=(
            "public_records",
            "open_social_posts",
            "licensed_cctv_feed",
            "news_reports",
            "tip_line",
        ),
    )

    assert len(out) == 2
    assert out[0].lead_id in {"lead_a", "lead_b"}
    assert "private_phone_contacts" not in " ".join(x.explanation for x in out)
