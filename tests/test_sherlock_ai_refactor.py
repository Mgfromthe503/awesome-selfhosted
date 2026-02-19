import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sherlock_ai_refactor import Block, SherlockAI


def test_block_hash_is_stable():
    block = Block(previous_hash="abc", transaction=["Alice sends 1 BTC to Bob"])
    assert len(block.block_hash) == 64
    assert block.block_hash == Block(previous_hash="abc", transaction=["Alice sends 1 BTC to Bob"]).block_hash


def test_sherlock_ai_train_and_predict():
    ai = SherlockAI("Sherlock", "Watson", dt.date(2020, 1, 1))
    samples = [
        "alice sends bitcoin",
        "click here to win",
        "salary transfer completed",
        "act now to claim",
        "invoice payment processed",
        "urgent free offer",
    ]
    labels = [0, 1, 0, 1, 0, 1]

    score = ai.train_model(samples, labels)
    assert 0.0 <= score <= 1.0

    preds = ai.make_decision(["payment approved", "win a prize now"])
    assert len(preds) == 2


def test_sherlock_age_non_negative():
    ai = SherlockAI("Sherlock", "Watson", dt.date.today())
    assert ai.get_age() >= 0
