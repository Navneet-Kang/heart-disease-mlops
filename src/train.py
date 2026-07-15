"""Train, compare, track, and save heart-disease classification models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_processed_data,
)

RANDOM_STATE = 42

# Your local Windows project path.
DEFAULT_WINDOWS_PROJECT_ROOT = Path(
    r"C:\Users\Navneet Kang\Desktop\heart-disease-mlops\heart-disease-mlops"
)


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    """Resolve the project root safely on Windows and other environments."""
    if project_root:
        root = Path(project_root).expanduser().resolve()
    elif DEFAULT_WINDOWS_PROJECT_ROOT.exists():
        root = DEFAULT_WINDOWS_PROJECT_ROOT.resolve()
    else:
        root = Path.cwd().resolve()

    required_items = ["src", "data"]
    missing = [item for item in required_items if not (root / item).exists()]

    if missing:
        raise FileNotFoundError(
            f"Invalid project root: {root}\n"
            f"Missing required folders: {missing}\n"
            "Run this script from the project root or pass "
            "--project-root followed by the full path."
        )

    return root


def build_preprocessor() -> ColumnTransformer:
    """Create numeric and categorical preprocessing pipelines."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(classifier: Any) -> Pipeline:
    """Combine preprocessing and classifier into one reusable pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", clone(build_preprocessor())),
            ("classifier", classifier),
        ]
    )


def create_model_searches(cv: StratifiedKFold) -> dict[str, GridSearchCV]:
    """Create tuned searches for Logistic Regression and Random Forest."""
    model_configs = {
        "logistic_regression": (
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear", "lbfgs"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "classifier__n_estimators": [200, 500],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        ),
    }

    return {
        name: GridSearchCV(
            estimator=build_pipeline(estimator),
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        for name, (estimator, parameter_grid) in model_configs.items()
    }


def calculate_metrics(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calculate hold-out classification metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def train(
    project_root: str | Path | None = None,
    use_mlflow: bool = True,
) -> dict[str, Any]:
    """Train both models, select the best model, and save all artifacts."""
    root = resolve_project_root(project_root)

    print(f"Using project root: {root}")

    model_dir = root / "models"
    artifact_dir = root / "artifacts"

    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    data = load_processed_data(root)

    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    searches = create_model_searches(cv)

    mlflow_database_path = root / "mlflow.db"

    if use_mlflow:
        mlflow.set_tracking_uri(
            f"sqlite:///{mlflow_database_path.as_posix()}"
        )
        mlflow.set_experiment("heart-disease-classification")

    results: list[dict[str, Any]] = []
    fitted_searches: dict[str, GridSearchCV] = {}

    for model_name, search in searches.items():
        print(f"\nTraining {model_name}...")

        if use_mlflow:
            mlflow.start_run(run_name=model_name)

        try:
            search.fit(X_train, y_train)
            fitted_searches[model_name] = search

            metrics = calculate_metrics(
                search.best_estimator_,
                X_test,
                y_test,
            )

            result = {
                "model": model_name,
                "best_cv_roc_auc": float(search.best_score_),
                **metrics,
                "best_params": search.best_params_,
            }
            results.append(result)

            print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
            print(f"Best parameters: {search.best_params_}")
            print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")

            if use_mlflow:
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("random_state", RANDOM_STATE)
                mlflow.log_param("test_size", 0.20)
                mlflow.log_params(search.best_params_)

                mlflow.log_metric(
                    "best_cv_roc_auc",
                    result["best_cv_roc_auc"],
                )

                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                mlflow.sklearn.log_model(
                sk_model=search.best_estimator_,
                name="model",
                serialization_format="cloudpickle",
                )
        finally:
            if use_mlflow:
                mlflow.end_run()

    # Select the final model using cross-validation ROC-AUC.
    best_result = max(
        results,
        key=lambda item: item["best_cv_roc_auc"],
    )

    selected_model_name = str(best_result["model"])
    final_model = fitted_searches[
        selected_model_name
    ].best_estimator_

    model_path = model_dir / "heart_disease_pipeline.joblib"
    joblib.dump(final_model, model_path)

    comparison_rows = [
        {
            key: value
            for key, value in result.items()
            if key != "best_params"
        }
        for result in results
    ]

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "best_cv_roc_auc",
        ascending=False,
    )

    comparison_path = artifact_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    metadata = {
        "selected_model": selected_model_name,
        "selection_metric": "cross_validation_roc_auc",
        "model_path": str(model_path),
        "mlflow_database": str(mlflow_database_path),
        "features": FEATURES,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "results": results,
    }

    metadata_path = artifact_dir / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    (model_dir / "model_name.txt").write_text(
        selected_model_name,
        encoding="utf-8",
    )

    print("\nModel comparison:")
    print(comparison.to_string(index=False))

    print(f"\nSelected model: {selected_model_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved metadata: {metadata_path}")

    if use_mlflow:
        print(f"MLflow database: {mlflow_database_path}")

    return metadata


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--project-root",
        default=None,
        help=(
            "Optional project root. If omitted, the script uses "
            r"C:\Users\Navneet Kang\Desktop\heart-disease-mlops"
            r"\heart-disease-mlops when it exists."
        ),
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Train without recording MLflow runs.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train(
        project_root=args.project_root,
        use_mlflow=not args.no_mlflow,
    )
