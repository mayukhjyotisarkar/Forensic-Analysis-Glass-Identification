"""Load the saved model and predict glass type from RI + oxide measurements.

Examples
--------
    python -m src.predict --RI 1.5161 --Na 13.5 --Mg 3.6 --Al 1.2 --Si 72.8 --K 0.4 --Ca 8.2 --Ba 0 --Fe 0
    python -m src.predict --csv new_samples.csv
"""
import argparse
import json

import joblib
import pandas as pd

from .config import CLASS_NAMES, FEATURES, METADATA_PATH, MODEL_PATH


def load_model(path=MODEL_PATH):
    """Load the trained pipeline (raises a helpful error if it is missing)."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"No model at {path}. Train one first: python -m src.train") from None


def load_metadata(path=METADATA_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def out_of_range_warnings(df: pd.DataFrame, ranges: dict) -> list:
    """Flag inputs outside the range seen in training (predictions there are extrapolation)."""
    msgs = []
    for col, (lo, hi) in ranges.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        for idx in bad.index:
            msgs.append(f"Row {idx}: {col}={df.loc[idx, col]} is outside the training range [{lo:g}, {hi:g}]")
    return msgs


def predict(model, samples) -> pd.DataFrame:
    """Predict for a dict, list of dicts, or DataFrame containing the 9 feature columns."""
    if isinstance(samples, dict):
        samples = [samples]
    X = pd.DataFrame(samples)
    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = X[FEATURES].astype(float)

    out = pd.DataFrame({"predicted_type": model.predict(X)})
    out["description"] = out["predicted_type"].map(CLASS_NAMES)
    if hasattr(model, "predict_proba"):
        proba = pd.DataFrame(model.predict_proba(X), columns=[f"P(type {c})" for c in model.classes_])
        out["confidence"] = proba.max(axis=1)
        out = pd.concat([out, proba], axis=1)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in FEATURES:
        p.add_argument(f"--{f}", type=float)
    p.add_argument("--csv", help="CSV file with the 9 feature columns (batch prediction)")
    args = p.parse_args(argv)

    if args.csv:
        samples = pd.read_csv(args.csv)
    else:
        vals = {f: getattr(args, f) for f in FEATURES}
        if any(v is None for v in vals.values()):
            p.error("Provide all 9 features (--RI ... --Fe) or use --csv")
        samples = [vals]

    result = predict(load_model(), samples)
    print(result.round(3).to_string(index=False))
    try:
        ranges = load_metadata()["feature_ranges"]
        for msg in out_of_range_warnings(pd.DataFrame(samples), ranges):
            print("WARNING:", msg)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
