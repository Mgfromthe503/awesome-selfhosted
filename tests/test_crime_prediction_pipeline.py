from crime_prediction_pipeline import format_predictions, predict_unsolved_cases, train_model


def _training_rows():
    return [
        {"feature1": 0, "feature2": 1, "feature3": 5, "solved": 0},
        {"feature1": 1, "feature2": 1, "feature3": 6, "solved": 1},
        {"feature1": 0, "feature2": 0, "feature3": 5, "solved": 0},
        {"feature1": 1, "feature2": 0, "feature3": 6, "solved": 1},
        {"feature1": 0, "feature2": 1, "feature3": 4, "solved": 0},
        {"feature1": 1, "feature2": 1, "feature3": 7, "solved": 1},
    ]


def test_train_model_and_predict_unsolved_cases():
    features = ["feature1", "feature2", "feature3"]
    model, report = train_model(_training_rows(), features, "solved")

    assert "training_accuracy=" in report

    unsolved = [
        {"feature1": 1, "feature2": 1, "feature3": 7},
        {"feature1": 0, "feature2": 0, "feature3": 4},
    ]
    predictions = predict_unsolved_cases(model, unsolved)
    assert predictions == [1, 0]


def test_format_predictions_messages():
    assert format_predictions([1, 0]) == [
        "Case 0 is likely to be solved.",
        "Case 1 is unlikely to be solved.",
    ]
