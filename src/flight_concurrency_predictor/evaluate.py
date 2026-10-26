"""
Train and evaluate models on the Avinor flight data.

Run from the repo root, for example:

    # Original baseline (3-fold CV), XGBoost only
    poetry run python -m flight_concurrency_predictor.evaluate --cv kfold

    # Time-aware cross-validation
    poetry run python -m flight_concurrency_predictor.evaluate --cv time

    # Several models
    poetry run python -m flight_concurrency_predictor.evaluate --models xg rf logreg

Every run appends one row per model to evaluation/experiment_log.csv.
"""
import argparse
import csv
import datetime as dt
import os
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

from .train import TrainConfig, train_with_cv
from .data_preprocessing import split_data, load_preprocessed
from .model import create_XGBoost, create_LogisticRegression, create_RandomForest, create_SVC

DATA_PATH = "data/historical_flights.csv"
EVAL_DIR = "evaluation"
LOG_PATH = os.path.join(EVAL_DIR, "experiment_log.csv")

MODELS = {
    "xg": create_XGBoost,
    "logreg": create_LogisticRegression,
    "rf": create_RandomForest,
    "svc": create_SVC,
}


def get_scores(model: BaseEstimator, X: pd.DataFrame):
    """Return continuous scores for ROC-AUC: probabilities if available, else decision function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series,
                   model_name: str, tag: str = "") -> dict:
    """
    Evaluate a trained model, save plots and print metrics.

    Two AUC values are reported:
      - auc_proba:  ROC-AUC computed from predicted probabilities (correct way)
      - auc_labels: ROC-AUC computed from hard 0/1 predictions (what the original
                    evaluate.py reported)

    Args:
        model: A fitted scikit-learn compatible model.
        X_test, y_test: Test features and labels.
        model_name: Short name used in file names.
        tag: Optional suffix for file names (e.g. "_time" or "_kfold").

    Returns:
        dict with 'auc_proba' and 'auc_labels'.
    """
    os.makedirs(EVAL_DIR, exist_ok=True)

    y_label = model.predict(X_test)
    y_score = get_scores(model, X_test)

    auc_labels = roc_auc_score(y_test, y_label)
    auc_proba = roc_auc_score(y_test, y_score)
    print(f"{model_name} - AUC from hard labels (original method): {auc_labels:.4f}")
    print(f"{model_name} - AUC from probabilities:                 {auc_proba:.4f}")

    # ROC curve (from probabilities)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_proba:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.savefig(f"{EVAL_DIR}/roc_curve_{model_name}{tag}.png")
    plt.close()

    # Confusion matrix (hard labels)
    cm = confusion_matrix(y_test, y_label)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"{EVAL_DIR}/confusion_matrix_{model_name}{tag}.png")
    plt.close()

    # Classification report
    report = classification_report(y_test, y_label, digits=4)
    print(f"Classification Report ({model_name}):\n{report}")

    return {"auc_proba": auc_proba, "auc_labels": auc_labels}


def log_result(row: dict) -> None:
    """Append one result row to evaluation/experiment_log.csv (creates it with a header)."""
    os.makedirs(EVAL_DIR, exist_ok=True)
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["xg"], choices=list(MODELS),
                        help="Models to train (default: xg)")
    parser.add_argument("--cv", choices=["time", "kfold"], default="time",
                        help="time = TimeSeriesSplit, kfold = original 3-fold CV")
    parser.add_argument("--splits", type=int, default=5, help="TimeSeriesSplit splits (default 5)")
    parser.add_argument("--window", type=float, default=5.0,
                        help="Concurrency window in minutes (default 5.0 = original behaviour)")
    parser.add_argument("--data", default=DATA_PATH, help="Path to historical flights CSV")
    parser.add_argument("--no-cache", action="store_true", help="Recompute preprocessing")
    parser.add_argument("--note", default="", help="Free-text note saved in the experiment log")
    args = parser.parse_args()

    df = load_preprocessed(args.data, window_minutes=args.window, use_cache=not args.no_cache)
    _x, _y, X_test, y_test = split_data(df)

    for name in args.models:
        print(f"\n=== training {name} (cv={args.cv}, window={args.window} min) ===")
        config = TrainConfig(
            flight_csv=args.data,
            model_function=MODELS[name],
            cv_strategy=args.cv,
            cv_splits=args.splits,
            window_minutes=args.window,
            use_cache=not args.no_cache,
        )
        model, info = train_with_cv(config)

        print(f"=== evaluating {name} ===")
        metrics = evaluate_model(model, X_test, y_test, name, tag=f"_{args.cv}")

        log_result({
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "model": name,
            "cv": info["cv_description"],
            "window_minutes": args.window,
            "cv_auc": round(info["cv_auc"], 4),
            "test_auc_proba": round(metrics["auc_proba"], 4),
            "test_auc_labels": round(metrics["auc_labels"], 4),
            "best_params": str(info["best_params"]),
            "note": args.note,
        })
        print(f"Logged to {LOG_PATH}")


if __name__ == "__main__":
    main()