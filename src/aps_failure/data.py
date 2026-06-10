"""Dataset loading helpers for the Scania APS failure data."""

from pathlib import Path

import pandas as pd


LABEL_COLUMN = "class"
LABEL_MAPPING = {"neg": 0, "pos": 1}


def load_aps_csv(path: str | Path) -> pd.DataFrame:
    """Load a Scania APS CSV file and encode the target label.

    The original files contain 20 lines of metadata before the CSV header and
    use the string "na" for missing sensor values.
    """
    frame = pd.read_csv(path, skiprows=20, na_values=["na"])
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].map(LABEL_MAPPING)
    return frame


def split_features_target(frame: pd.DataFrame):
    """Return feature matrix and binary target vector from an APS dataframe."""
    return frame.drop(columns=[LABEL_COLUMN]), frame[LABEL_COLUMN]
