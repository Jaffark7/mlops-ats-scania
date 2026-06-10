"""Model persistence and inference-frame utilities."""

from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_model_path() -> Path:
    """Resolve the model artifact in source checkouts and container runtime."""
    cwd_model = Path.cwd() / "model.joblib"
    if cwd_model.exists():
        return cwd_model
    return PROJECT_ROOT / "model.joblib"


DEFAULT_MODEL_PATH = default_model_path()


def load_model_bundle(path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Load the persisted sklearn pipeline and its training feature order."""
    bundle = joblib.load(path)
    required_keys = {"model", "feature_names"}
    missing = required_keys.difference(bundle)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Model bundle is missing required key(s): {missing_list}")
    return bundle


def save_model_bundle(model, feature_names: list[str], path: str | Path) -> None:
    """Persist a model together with the feature order expected at inference."""
    joblib.dump({"model": model, "feature_names": feature_names}, path)


def make_inference_frame(features: Mapping[str, float], feature_names: list[str]) -> pd.DataFrame:
    """Create a single-row frame ordered exactly like training data.

    Missing fields become NaN and are handled by the trained imputer inside the
    persisted pipeline.
    """
    return pd.DataFrame([features]).reindex(columns=feature_names)
