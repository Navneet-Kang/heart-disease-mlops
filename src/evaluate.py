"""Evaluate the saved heart-disease model pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data import FEATURES, TARGET, load_processed_data

RANDOM_STATE = 42


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    """Resolve the project root."""
    if project_root:
        return Path(project_root).expanduser().resolve()
    return Path.cwd().resolve()


def evaluate(project_root: str | Path | None = None) -> dict[str, float]:
    """Load the saved model and evaluate it on the hold-out split."""
    root = resolve_project_root(project_root)

    model_path = root / "models" / "heart_disease_pipeline.joblib"
    artifact_dir = root / "artifacts"
    plot_dir = artifact_dir / "plots"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run `python -m src.train` before evaluation."
        )

    data = load_processed_data(root)
    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = joblib.load(model_path)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    print("\nClassification report:\n")
    print(classification_report(y_test, predictions, digits=4))

    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    metrics_path = artifact_dir / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Final Model")
    fig.tight_layout()
    confusion_path = plot_dir / "final_model_confusion_matrix.png"
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        ax=ax,
    )
    ax.set_title("ROC Curve — Final Model")
    fig.tight_layout()
    roc_path = plot_dir / "final_model_roc_curve.png"
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved confusion matrix: {confusion_path}")
    print(f"Saved ROC curve: {roc_path}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional project root. Defaults to the current directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.project_root)
