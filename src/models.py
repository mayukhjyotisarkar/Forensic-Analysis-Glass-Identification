"""Model zoo and hyper-parameter grids.

Every model is wrapped in the same Pipeline:
    feature engineering -> scaling -> (optional SMOTE) -> classifier
so that ALL preprocessing is fitted inside cross-validation folds (no leakage).
"""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .preprocess import add_features


def make_pipeline(clf, engineer: bool = False, smote: bool = False) -> Pipeline:
    """Wrap a classifier with the shared preprocessing steps."""
    steps = []
    if engineer:
        steps.append(("features", FunctionTransformer(add_features)))
    steps.append(("scale", StandardScaler()))
    if smote:
        # k_neighbors is small because the rarest class has only ~5 samples per CV fold
        steps.append(("smote", SMOTE(k_neighbors=2, random_state=RANDOM_STATE)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def get_models(engineer: bool = False) -> dict:
    """Return {name: pipeline} for every candidate model.

    `engineer=False` is the default: in our ablation study (notebooks/02) the
    engineered features did not beat the raw ones, so we keep the simpler model.
    """
    rs = RANDOM_STATE
    rf = lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=rs)
    return {
        "Dummy (most frequent)": make_pipeline(DummyClassifier(strategy="most_frequent"), engineer),
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=5000, class_weight="balanced"), engineer),
        "KNN": make_pipeline(KNeighborsClassifier(n_neighbors=5, weights="distance"), engineer),
        "SVM (RBF)": make_pipeline(
            SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=rs), engineer),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(class_weight="balanced", random_state=rs), engineer),
        "Random Forest": make_pipeline(rf(), engineer),
        "Extra Trees": make_pipeline(
            ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=rs), engineer),
        "Gradient Boosting": make_pipeline(GradientBoostingClassifier(random_state=rs), engineer),
        "Random Forest + SMOTE": make_pipeline(
            RandomForestClassifier(n_estimators=300, random_state=rs), engineer, smote=True),
    }


# Hyper-parameter grids (keys use the pipeline step name "clf__").
PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.1, 1, 10, 100]},
    "KNN": {"clf__n_neighbors": [1, 3, 5, 7, 9], "clf__weights": ["uniform", "distance"]},
    "SVM (RBF)": {"clf__C": [1, 5, 10, 50, 100], "clf__gamma": ["scale", 0.02, 0.05, 0.1, 0.3]},
    "Decision Tree": {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 2, 4]},
    "Random Forest": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10],
                      "clf__min_samples_leaf": [1, 2], "clf__max_features": ["sqrt", 0.5]},
    "Extra Trees": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10],
                    "clf__min_samples_leaf": [1, 2], "clf__max_features": ["sqrt", 0.5]},
    "Gradient Boosting": {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                          "clf__max_depth": [2, 3]},
}


def get_param_grid(model_name: str) -> dict:
    """Grid for a model name (the SMOTE variant reuses the plain model's grid)."""
    return PARAM_GRIDS.get(model_name.replace(" + SMOTE", ""), {})
