"""Fast sanity tests. Run with:  pytest -q"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.config import CLASS_NAMES, FEATURES
from src.data_loader import load_data, validate
from src.models import get_models, make_pipeline
from src.predict import out_of_range_warnings, predict
from src.preprocess import add_features, feature_names, get_X_y, split_data


@pytest.fixture(scope="module")
def data():
    return load_data(drop_duplicates=True)


def test_load_data(data):
    assert data.shape[1] == 10 and len(data) > 200
    assert set(data["Type"]) <= set(CLASS_NAMES)


def test_validate_rejects_bad_data(data):
    with pytest.raises(ValueError):
        validate(data.drop(columns=["Ba"]))
    bad = data.copy()
    bad.loc[0, "Type"] = 99
    with pytest.raises(ValueError):
        validate(bad)


def test_add_features_is_finite(data):
    X, _ = get_X_y(data)
    out = add_features(X)
    assert list(out.columns) == feature_names()
    assert np.isfinite(out.to_numpy()).all()


def test_split_is_stratified(data):
    X, y = get_X_y(data)
    _, _, ytr, yte = split_data(X, y)
    assert set(ytr) == set(yte) == set(y)


@pytest.mark.parametrize("engineer", [False, True])
def test_pipeline_fits_and_predicts(data, engineer):
    X, y = get_X_y(data)
    pipe = make_pipeline(RandomForestClassifier(n_estimators=20, random_state=0), engineer=engineer)
    pipe.fit(X, y)
    assert len(pipe.predict(X.head(5))) == 5


def test_all_models_are_buildable():
    assert "Random Forest" in get_models()


def test_predict_output_shape(data):
    X, y = get_X_y(data)
    pipe = make_pipeline(RandomForestClassifier(n_estimators=20, random_state=0)).fit(X, y)
    out = predict(pipe, X.iloc[0].to_dict())
    assert out.loc[0, "predicted_type"] in CLASS_NAMES
    assert 0 < out.loc[0, "confidence"] <= 1
    with pytest.raises(ValueError):
        predict(pipe, {"RI": 1.5})


def test_out_of_range_warning():
    df = pd.DataFrame([{"RI": 2.0}])
    assert out_of_range_warnings(df, {"RI": [1.51, 1.53]})
