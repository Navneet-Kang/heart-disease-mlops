
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/heart_disease_pipeline.joblib")


def test_saved_model_exists():
    assert MODEL_PATH.exists(), (
        "Saved model was not found. Run the model packaging section first."
    )


def test_saved_model_prediction():
    model = joblib.load(MODEL_PATH)

    patient = pd.DataFrame(
        [
            {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        ]
    )

    prediction = model.predict(patient)
    probability = model.predict_proba(patient)

    assert prediction[0] in [0, 1]
    assert probability.shape == (1, 2)
    assert 0 <= probability[0, 1] <= 1
