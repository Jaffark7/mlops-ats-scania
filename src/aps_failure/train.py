"""Train the Scania APS XGBoost pipeline."""

from argparse import ArgumentParser
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from aps_failure.data import load_aps_csv, split_features_target
from aps_failure.model_io import save_model_bundle


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """Create the preprocessing + classifier pipeline used by the project."""
    classifier = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", classifier),
        ]
    )


def train(training_csv: str | Path, output_path: str | Path) -> dict:
    """Fit the model and save a bundle containing the pipeline and feature names."""
    train_frame = load_aps_csv(training_csv)
    x_train, y_train = split_features_target(train_frame)

    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    scale_pos_weight = num_neg / num_pos

    pipeline = build_pipeline(scale_pos_weight=scale_pos_weight)
    cv_f1 = cross_val_score(pipeline, x_train, y_train, cv=5, scoring="f1")
    pipeline.fit(x_train, y_train)

    save_model_bundle(pipeline, x_train.columns.tolist(), output_path)
    return {
        "scale_pos_weight": scale_pos_weight,
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_folds": [float(score) for score in cv_f1],
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--training-csv", default="data/aps_failure_training_set.csv")
    parser.add_argument("--output", default="model.joblib")
    args = parser.parse_args()

    summary = train(args.training_csv, args.output)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
