"""Evaluate the trained Scania APS model on the held-out test set."""

from argparse import ArgumentParser
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from aps_failure.data import load_aps_csv, split_features_target
from aps_failure.model_io import load_model_bundle


def evaluate(model_path: str | Path, test_csv: str | Path) -> dict:
    """Return the core metrics used in the project README."""
    bundle = load_model_bundle(model_path)
    model = bundle["model"]

    test_frame = load_aps_csv(test_csv)
    x_test, y_test = split_features_target(test_frame)
    x_test = x_test.reindex(columns=bundle["feature_names"])

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model.joblib")
    parser.add_argument("--test-csv", default="data/aps_failure_test_set.csv")
    parser.add_argument("--output", default="results/metrics.json")
    args = parser.parse_args()

    metrics = evaluate(args.model, args.test_csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
