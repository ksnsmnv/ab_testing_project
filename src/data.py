from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "ab_test.csv"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else CSV_PATH
    df = pd.read_csv(csv_path)
    required = {"user_id", "age", "device", "country", "tenure_days", "prior_purchases", "group", "converted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected schema, missing columns: {sorted(missing)}")

    df = df.copy()
    df["treatment"] = (df["group"].str.lower() == "treatment").astype(int)
    df["is_mobile"] = (df["device"].str.lower() == "mobile").astype(int)
    df["is_new"] = (df["tenure_days"] < 30).astype(int)
    df["is_zero_prior"] = (df["prior_purchases"] == 0).astype(int)
    df["ones"] = 1.0
    return df


def load_ground_truth(path: Path | None = None) -> dict:
    gt_path = Path(path) if path else GROUND_TRUTH_PATH
    with gt_path.open() as f:
        return json.load(f)


def split_groups(df: pd.DataFrame, metric: str = "converted") -> tuple[np.ndarray, np.ndarray]:
    test = df.loc[df["treatment"] == 1, metric].to_numpy(dtype=float)
    control = df.loc[df["treatment"] == 0, metric].to_numpy(dtype=float)
    return test, control
