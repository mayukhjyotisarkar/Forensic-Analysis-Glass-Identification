"""Feature engineering and train/test splitting.

`add_features` lives in this module (rather than in a notebook) on purpose:
joblib pickles functions *by reference*, so a saved model can only be loaded
later if the function can be imported from a real module.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, RANDOM_STATE, TARGET, TEST_SIZE

# Heavily right-skewed oxides where most values are 0 -> log1p compresses them.
LOG_COLS = ["Mg", "K", "Ba", "Fe"]
EPS = 1e-6


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with engineered features added.

    Expects a DataFrame containing the 9 raw feature columns. Ratios are
    computed from the *raw* values first, then the skewed columns are log-scaled.
    """
    X = X[FEATURES].copy()
    X["Ca_Na"] = X["Ca"] / (X["Na"] + EPS)          # flux ratio
    X["Mg_Al"] = X["Mg"] / (X["Al"] + EPS)
    X["Ba_present"] = (X["Ba"] > 0).astype(int)     # Ba is mostly 0; presence is a signal
    X["Mg_zero"] = (X["Mg"] == 0).astype(int)       # Mg == 0 is typical of non-window glass
    for col in LOG_COLS:
        X[col] = np.log1p(X[col])
    return X


def feature_names() -> list:
    """Column names produced by `add_features` (useful for importance plots)."""
    return list(add_features(pd.DataFrame([[1.0] * len(FEATURES)], columns=FEATURES)).columns)


def get_X_y(df: pd.DataFrame):
    """Split a dataframe into the feature matrix and the target vector."""
    return df[FEATURES].copy(), df[TARGET].copy()


def split_data(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Stratified train/test split (keeps class proportions in both parts)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
