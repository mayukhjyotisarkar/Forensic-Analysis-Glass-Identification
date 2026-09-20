"""Loading and validating the Glass Identification dataset."""
import io
import urllib.request
from pathlib import Path

import pandas as pd

from .config import CLASS_NAMES, DATA_PATH, DATA_URLS, FEATURES, TARGET


def _parse_raw(text: str) -> pd.DataFrame:
    """Parse the header-less UCI format (with or without the leading Id column)."""
    df = pd.read_csv(io.StringIO(text), header=None)
    if df.shape[1] == 11:            # Id, 9 features, Type
        df = df.iloc[:, 1:]
    if df.shape[1] != 10:
        raise ValueError(f"Expected 10 or 11 columns, got {df.shape[1]}")
    df.columns = FEATURES + [TARGET]
    return df


def download_dataset(path: Path = DATA_PATH) -> Path:
    """Download the dataset and save it as a CSV *with* a header row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in DATA_URLS:
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                df = _parse_raw(resp.read().decode("utf-8"))
            df.to_csv(path, index=False)
            return path
        except Exception as exc:     # try the next mirror
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        "Could not download the dataset. Download it manually from Kaggle/UCI "
        f"and save it as {path}.\n" + "\n".join(errors)
    )


def validate(df: pd.DataFrame) -> None:
    """Fail early, with a clear message, if the data is not what we expect."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Expected header: {FEATURES + [TARGET]}"
        )
    if df[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Dataset contains missing values.")
    unknown = set(df[TARGET].unique()) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unexpected class labels: {sorted(unknown)}")


def load_data(path: Path = DATA_PATH, drop_duplicates: bool = False) -> pd.DataFrame:
    """Load the dataset (downloading it first if needed).

    Handles the Kaggle version too (which has an extra `Id` column).
    """
    path = Path(path)
    if not path.exists():
        download_dataset(path)
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ("Id", "id", "ID") if c in df.columns])
    validate(df)
    df = df[FEATURES + [TARGET]]
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df
