"""Train, tune, evaluate and save the glass-type classifier.

Examples
--------
    python -m src.train                          # compare all models, tune the winner
    python -m src.train --model "SVM (RBF)"      # skip the comparison, tune one model
    python -m src.train --no-tune                # faster: no grid search
    python -m src.train --refit-all              # final model trained on ALL data
"""
import matplotlib
matplotlib.use("Agg")  # no display needed when running as a script

import argparse
import json
import time
import warnings

import joblib
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config as C
from .data_loader import load_data
from .evaluate import (compare_models, compute_permutation_importance, make_cv,
                       plot_confusion_matrix, plot_model_comparison,
                       plot_permutation_importance, test_report)
from .models import get_models, get_param_grid
from .preprocess import get_X_y, split_data


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="Model name to tune (default: best in CV comparison)")
    p.add_argument("--no-tune", action="store_true", help="Skip GridSearchCV")
    p.add_argument("--engineer", action="store_true", help="Add engineered features")
    p.add_argument("--keep-duplicates", action="store_true", help="Do not drop duplicate rows")
    p.add_argument("--refit-all", action="store_true",
                   help="After evaluation, refit the final model on train+test data")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    warnings.filterwarnings("ignore")
    C.MODELS_DIR.mkdir(exist_ok=True)
    C.REPORTS_DIR.mkdir(exist_ok=True)

    # 1) data ----------------------------------------------------------
    df = load_data(drop_duplicates=not args.keep_duplicates)
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Data: {len(df)} rows | train {len(X_train)} | test {len(X_test)}")

    # 2) compare models with cross-validation (training data only) -----
    models = get_models(engineer=args.engineer)
    if args.model:
        if args.model not in models:
            raise SystemExit(f"Unknown model '{args.model}'. Choose from: {list(models)}")
        best_name = args.model
    else:
        print("\nCross-validating all models (repeated stratified 5-fold)...")
        results = compare_models(models, X_train, y_train, cv=make_cv())
        print(results.round(3).to_string(index=False))
        results.to_csv(C.REPORTS_DIR / "cv_results.csv", index=False)
        plot_model_comparison(results, C.REPORTS_DIR / "model_comparison.png")
        best_name = results[~results["model"].str.startswith("Dummy")].iloc[0]["model"]
    print(f"\nSelected model: {best_name}")

    # 3) tune ----------------------------------------------------------
    best = models[best_name]
    best_params, cv_score = {}, None
    grid = get_param_grid(best_name)
    if grid and not args.no_tune:
        print(f"Tuning over {len(grid)} hyper-parameters...")
        t0 = time.time()
        search = GridSearchCV(best, grid, scoring="f1_macro", refit=True,
                              cv=StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE))
        search.fit(X_train, y_train)
        best, best_params, cv_score = search.best_estimator_, search.best_params_, search.best_score_
        print(f"Best params: {best_params}\nCV macro-F1: {cv_score:.3f}  ({time.time() - t0:.0f}s)")
    else:
        best.fit(X_train, y_train)

    # 4) evaluate ONCE on the untouched test set ------------------------
    print("\n=== Test-set evaluation ===")
    metrics = test_report(best, X_test, y_test)
    print({k: round(v, 3) for k, v in metrics.items() if k != "per_class"})
    plot_confusion_matrix(best, X_test, y_test, save_path=C.REPORTS_DIR / "confusion_matrix.png")
    imp = compute_permutation_importance(best, X_test, y_test)
    plot_permutation_importance(imp, C.REPORTS_DIR / "permutation_importance.png")
    imp.to_csv(C.REPORTS_DIR / "permutation_importance.csv", index=False)

    # 5) save ----------------------------------------------------------
    final = best
    if args.refit_all:
        final = clone(best).fit(X, y)
        print("Refitted the final model on all data.")
    joblib.dump(final, C.MODEL_PATH)
    meta = {
        "model_name": best_name,
        "best_params": best_params,
        "cv_macro_f1_tuned": cv_score,
        "test_metrics": {k: v for k, v in metrics.items() if k != "per_class"},
        "engineered_features": args.engineer,
        "refit_on_all_data": args.refit_all,
        "features": C.FEATURES,
        "classes": [int(c) for c in final.classes_],
        "class_names": {str(k): v for k, v in C.CLASS_NAMES.items()},
        "feature_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in C.FEATURES},
        "n_rows": int(len(df)),
        "sklearn_version": sklearn.__version__,
    }
    C.METADATA_PATH.write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nSaved model -> {C.MODEL_PATH}\nSaved metadata -> {C.METADATA_PATH}")


if __name__ == "__main__":
    main()
