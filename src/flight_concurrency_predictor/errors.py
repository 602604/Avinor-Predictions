"""
Error analysis for the XGBoost concurrency model.

Answers: where does the model fail, and are its probabilities trustworthy?

Run from the repo root:

    poetry run python -m flight_concurrency_predictor.errors

Outputs (in evaluation/):
    errors_overall.csv, errors_by_airport_group.csv, errors_by_hour.csv,
    errors_by_month.csv, calibration_bins.csv, reliability_diagram.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # save figures to file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .baselines import XGB_PARAMS
from .data_preprocessing import load_preprocessed, split_data, pipeline_preprocessor

DATA_PATH = "data/historical_flights.csv"
EVAL_DIR = "evaluation"


def metrics(y, p) -> dict:
    """Summary metrics for one slice of the test set."""
    y = np.asarray(y)
    p = np.asarray(p)
    both_classes = len(np.unique(y)) == 2
    return {
        "n": len(y),
        "actual_rate": round(y.mean(), 3),
        "mean_predicted": round(p.mean(), 3),
        "bias": round(p.mean() - y.mean(), 3),          # negative = model underpredicts
        "auc": round(roc_auc_score(y, p), 4) if both_classes else np.nan,
        "brier": round(brier_score_loss(y, p), 4),
    }


def by_slice(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(col):
        row = {col: key}
        row.update(metrics(g["y"], g["p"]))
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_plot(y, p, path: str, bins: int = 10) -> pd.DataFrame:
    """Reliability diagram: predicted probability vs observed frequency."""
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    table = pd.DataFrame({"mean_predicted": mean_pred.round(3), "observed_rate": frac_pos.round(3)})

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", label="XGBoost (test set)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency of concurrency")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--window", type=float, default=5.0)
    args = parser.parse_args()

    df = load_preprocessed(args.data, window_minutes=args.window)
    X_train, y_train, X_test, y_test = split_data(df)

    model = Pipeline([("pre", pipeline_preprocessor()), ("clf", XGBClassifier(**XGB_PARAMS))])
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]

    res = X_test[["airport_group", "hour"]].copy()
    res["month"] = pd.to_datetime(X_test["date"]).dt.to_period("M").astype(str)
    res["y"] = y_test.to_numpy()
    res["p"] = p

    os.makedirs(EVAL_DIR, exist_ok=True)

    overall = pd.DataFrame([{**metrics(res["y"], res["p"]), "log_loss": round(log_loss(res["y"], res["p"]), 4)}])
    print("\n=== Overall (test set) ===")
    print(overall.to_string(index=False))
    overall.to_csv(f"{EVAL_DIR}/errors_overall.csv", index=False)

    for col, name in [("airport_group", "airport_group"), ("hour", "hour"), ("month", "month")]:
        table = by_slice(res, col)
        print(f"\n=== By {name} ===")
        print(table.to_string(index=False))
        table.to_csv(f"{EVAL_DIR}/errors_by_{name}.csv", index=False)

    bins = reliability_plot(res["y"], res["p"], f"{EVAL_DIR}/reliability_diagram.png")
    print("\n=== Calibration bins (quantile) ===")
    print(bins.to_string(index=False))
    bins.to_csv(f"{EVAL_DIR}/calibration_bins.csv", index=False)
    print(f"\nSaved tables and reliability_diagram.png to {EVAL_DIR}/")


if __name__ == "__main__":
    main()