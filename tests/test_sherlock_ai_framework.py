from sherlock_ai_framework import (
    Book,
    Collaboration,
    ContinuousLearning,
    Contribution,
    Ethics,
    Feedback,
    Partner,
    Principle,
    Sherlock,
    Update,
    User,
)


def test_dna_principle_integration_and_truth_analysis():
    sherlock = Sherlock()
    sherlock.dna.integrate_principle(Principle(code=1, name="Correspondence", description="As above, so below."))

    truth = sherlock.seek_truth({"sample": "data"})

    assert truth["principles_loaded"] == 1
    assert truth["analysis"]["status"] == "processed"


def test_investigate_report_and_training_workflow():
    sherlock = Sherlock()
    findings = sherlock.investigate("genomics payload")
    report = sherlock.report(findings)
    training_result = sherlock.train_on_book(Book(title="Bioinformatics 101", author="A. Researcher", content="A C G T"))

    assert findings["dimensions"] == 12
    assert "Sherlock Report" in report
    assert training_result["total_books"] == 1


def test_ethics_learning_and_collaboration_integration():
    sherlock = Sherlock()
    ethics = Ethics()
    learning = ContinuousLearning(sherlock)
    collaboration = Collaboration(sherlock)

    bias_check = ethics.check_for_bias("some payload")
    privacy = ethics.ensure_privacy(User(user_id="u-1", consented=True))
    transparency = ethics.provide_transparency()

    learning.apply_update(Update(description="Tune detector"))
    learning.gather_feedback(Feedback(user="u-1", comment="Looks good", rating=5))

    result = collaboration.collaborate_with_partner(Partner(name="OpenBioLab", organization="Community"))
    collaboration.integrate_open_source_contribution(
        Contribution(source="github.com/example/repo", payload={"feature": "new-corpus"})
    )

    assert isinstance(bias_check["flagged"], bool)
    assert privacy["privacy_verified"] is True
    assert "Transparency" in transparency.title
    assert sherlock.applied_updates
    assert sherlock.feedback_log
    assert result["status"] == "collaboration_started"
