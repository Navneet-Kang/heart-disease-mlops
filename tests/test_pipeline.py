
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAL_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]


def create_sample_data():
    """Create clinically plausible sample observations for testing."""
    return pd.DataFrame(
        {
            "age": [63, 45, 58, 52],
            "sex": [1, 0, 1, 1],
            "cp": [3, 1, 2, 0],
            "trestbps": [145, 120, 132, 128],
            "chol": [233, 210, 224, 245],
            "fbs": [1, 0, 0, 0],
            "restecg": [0, 1, 1, 0],
            "thalach": [150, 170, 165, 140],
            "exang": [0, 0, 0, 1],
            "oldpeak": [2.3, 0.2, 1.0, 1.8],
            "slope": [0, 2, 1, 1],
            "ca": [0, 0, 1, 2],
            "thal": [1, 2, 2, 3],
        }
    )


def create_pipeline():
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )


def test_sample_data_shape():
    data = create_sample_data()

    expected_columns = set(
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    )

    assert data.shape == (4, 13)
    assert set(data.columns) == expected_columns


def test_pipeline_fits_and_predicts():
    X = create_sample_data()
    y = np.array([1, 0, 1, 0])

    pipeline = create_pipeline()
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})


def test_prediction_probabilities():
    X = create_sample_data()
    y = np.array([1, 0, 1, 0])

    pipeline = create_pipeline()
    pipeline.fit(X, y)

    probabilities = pipeline.predict_proba(X)

    assert probabilities.shape == (4, 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_pipeline_handles_missing_values():
    X = create_sample_data()
    y = np.array([1, 0, 1, 0])

    X.loc[0, "chol"] = np.nan
    X.loc[1, "ca"] = np.nan
    X.loc[2, "thal"] = np.nan

    pipeline = create_pipeline()
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == 4
    assert not pd.isna(predictions).any()


def test_unknown_category_is_handled():
    X = create_sample_data()
    y = np.array([1, 0, 1, 0])

    pipeline = create_pipeline()
    pipeline.fit(X, y)

    new_patient = X.iloc[[0]].copy()
    new_patient["cp"] = 99

    prediction = pipeline.predict(new_patient)

    assert prediction[0] in [0, 1]
