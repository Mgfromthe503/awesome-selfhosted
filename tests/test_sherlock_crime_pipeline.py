import csv

from sherlock_crime_pipeline import format_case_predictions, load_dataset, run_pipeline


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_run_pipeline_end_to_end(tmp_path):
    crime_rows = [
        {"feature1": 1, "feature2": 5, "feature3": 0, "solved": 0},
        {"feature1": 2, "feature2": 4, "feature3": 1, "solved": 0},
        {"feature1": 7, "feature2": 1, "feature3": 0, "solved": 1},
        {"feature1": 8, "feature2": 0, "feature3": 1, "solved": 1},
    ]
    unsolved_rows = [
        {"feature1": 2, "feature2": 4, "feature3": 1},
        {"feature1": 8, "feature2": 0, "feature3": 1},
    ]

    crime_path = tmp_path / "crime_data.csv"
    unsolved_path = tmp_path / "unsolved_crimes.csv"
    write_csv(crime_path, crime_rows)
    write_csv(unsolved_path, unsolved_rows)

    training, messages = run_pipeline(crime_path, unsolved_path, features=["feature1", "feature2", "feature3"])
    assert 0.0 <= training.test_accuracy <= 1.0
    assert len(messages) == 2
    assert all(message.startswith("Case ") for message in messages)


def test_format_predictions():
    assert format_case_predictions([1, 0]) == [
        "Case 0 is likely to be solved.",
        "Case 1 is unlikely to be solved.",
    ]


def test_load_dataset(tmp_path):
    path = tmp_path / "data.csv"
    write_csv(path, [{"feature1": 1, "solved": 0}])
    data = load_dataset(path)
    if isinstance(data, list):
        assert data[0]["feature1"] == "1"
    else:
        assert data.iloc[0]["feature1"] == 1
