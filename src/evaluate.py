"""Evaluation helpers: cross-validated comparison, test-set report, plots."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             balanced_accuracy_score, classification_report, f1_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from .config import CLASS_NAMES, CV_REPEATS, CV_SPLITS, RANDOM_STATE

SCORING = {"f1_macro": "f1_macro", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"}


def make_cv(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE):
    """Repeated stratified K-fold: more stable than one K-fold on ~170 rows."""
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


def compare_models(models: dict, X, y, cv=None) -> pd.DataFrame:
    """Cross-validate every model and return a table sorted by macro-F1."""
    cv = cv or make_cv()
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=SCORING)
        rows.append({
            "model": name,
            "f1_macro": res["test_f1_macro"].mean(),
            "f1_macro_std": res["test_f1_macro"].std(),
            "accuracy": res["test_accuracy"].mean(),
            "balanced_accuracy": res["test_balanced_accuracy"].mean(),
            "fit_time_s": res["fit_time"].mean(),
        })
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)


def plot_model_comparison(results: pd.DataFrame, save_path=None):
    """Horizontal bar chart of macro-F1 (error bars = std across folds)."""
    df = results.sort_values("f1_macro")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(df) + 1.5))
    ax.barh(df["model"], df["f1_macro"], xerr=df["f1_macro_std"], color="#4C72B0", capsize=3)
    ax.set_xlabel("Cross-validated macro-F1 (mean ± std)")
    ax.set_title("Model comparison")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def test_report(model, X_test, y_test, verbose=True) -> dict:
    """Score a fitted model on held-out data and (optionally) print the full report."""
    pred = model.predict(X_test)
    labels = list(model.classes_)
    names = [f"{c}: {CLASS_NAMES[c]}" for c in labels]
    if verbose:
        print(classification_report(y_test, pred, labels=labels, target_names=names, zero_division=0))
    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        "per_class": classification_report(y_test, pred, labels=labels, zero_division=0,
                                           output_dict=True),
    }


def plot_confusion_matrix(model, X_test, y_test, normalize=None, ax=None, save_path=None):
    """Confusion matrix; `normalize='true'` shows per-class recall instead of counts."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    ConfusionMatrixDisplay.from_predictions(
        y_test, model.predict(X_test), labels=list(model.classes_),
        normalize=normalize, cmap="Blues", ax=ax, colorbar=False,
        values_format=".2f" if normalize else "d")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def compute_permutation_importance(model, X_test, y_test, n_repeats=30) -> pd.DataFrame:
    """Drop in macro-F1 when a *raw* feature is shuffled (works with any pipeline)."""
    res = permutation_importance(model, X_test, y_test, scoring="f1_macro",
                                 n_repeats=n_repeats, random_state=RANDOM_STATE)
    return (pd.DataFrame({"feature": X_test.columns,
                          "importance": res.importances_mean,
                          "std": res.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def plot_permutation_importance(imp: pd.DataFrame, save_path=None):
    df = imp.sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(df["feature"], df["importance"], xerr=df["std"], color="#55A868", capsize=3)
    ax.set_xlabel("Drop in macro-F1 when feature is shuffled")
    ax.set_title("Permutation importance (test set)")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def _maybe_save(fig, path):
    if path is not None:
        path = str(path)
        fig.savefig(path, dpi=150, bbox_inches="tight")
