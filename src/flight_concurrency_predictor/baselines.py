"""
Baseline models for the Avinor concurrency task.

Purpose: find out how much of the XGBoost AUC comes from simple signals, so that
improvements (weather, new features) can be judged against meaningful reference points.

Run from the repo root:

    poetry run python -m flight_concurrency_predictor.baselines
    poetry run python -m flight_concurrency_predictor.baselines --window 5

Results are printed and saved to evaluation/baselines.csv.
All models are trained on data before the cutoff (2024-07-01) and scored on the test
set with ROC-AUC computed from continuous scores.
"""
import argparse
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .data_preprocessing import load_preprocessed, split_data, pipeline_preprocessor

DATA_PATH = "data/historical_flights.csv"
OUT_PATH = "evaluation/baselines.csv"

# Best XGBoost settings found by the earlier grid search (time-aware CV run)
XGB_PARAMS = dict(learning_rate=0.1, max_depth=6, n_estimators=100)


def lookup_baseline(X_train, y_train, X_test, keys):
    """Score = mean positive rate of the training rows with the same `keys`."""
    tmp = X_train[keys].copy()
    tmp["y"] = y_train.values
    rates = tmp.groupby(keys)["y"].mean().rename("score").reset_index()
    merged = X_test[keys].merge(rates, on=keys, how="left")
    return merged["score"].fillna(y_train.mean()).to_numpy()


def make_preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


def run_baselines(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    cat = ["airport_group", "season", "hour"]
    results = []

    def add(name, description, score):
        auc = roc_auc_score(y_test, score)
        results.append({"baseline": name, "description": description, "test_auc": round(auc, 4)})
        print(f"{name:<32} AUC = {auc:.4f}   ({description})")

    # 0. Trivial reference
    add("random_constant", "constant score (should be 0.5)", [0.0] * len(y_test))

    # 1. Raw single features as scores (no model at all)
    add("sched_concurrence_only", "raw scheduled concurrence count", X_test["sched_concurrence"].to_numpy())
    add("sched_flights_only", "raw scheduled number of flights", X_test["sched_flights"].to_numpy())

    # 2. Lookup tables of historical positive rate
    add("rate_by_group_hour", "train positive rate per airport group and hour",
        lookup_baseline(X_train, y_train, X_test, ["airport_group", "hour"]))
    add("rate_by_group_hour_season", "train positive rate per group, hour and season",
        lookup_baseline(X_train, y_train, X_test, ["airport_group", "hour", "season"]))

    # 3. Simple models on the same features as XGBoost
    logreg = Pipeline([("pre", pipeline_preprocessor()), ("clf", LogisticRegression(max_iter=1000))])
    logreg.fit(X_train, y_train)
    add("logistic_regression", "same features as XGBoost", logreg.predict_proba(X_test)[:, 1])

    tree = Pipeline([("pre", pipeline_preprocessor()), ("clf", DecisionTreeClassifier(max_depth=5, random_state=0))])
    tree.fit(X_train, y_train)
    add("decision_tree_depth5", "same features as XGBoost", tree.predict_proba(X_test)[:, 1])

    # 4. XGBoost reference and ablation
    xgb = Pipeline([("pre", pipeline_preprocessor()), ("clf", XGBClassifier(**XGB_PARAMS))])
    xgb.fit(X_train, y_train)
    add("xgboost_all_features", "reference model, tuned params", xgb.predict_proba(X_test)[:, 1])

    xgb_nc = Pipeline([("pre", make_preprocessor(["sched_flights"], cat)),
                       ("clf", XGBClassifier(**XGB_PARAMS))])
    xgb_nc.fit(X_train, y_train)
    add("xgboost_no_sched_concurrence", "ablation: drops sched_concurrence", xgb_nc.predict_proba(X_test)[:, 1])

    xgb_nt = Pipeline([("pre", make_preprocessor(["sched_flights", "sched_concurrence"], ["airport_group", "season"])),
                       ("clf", XGBClassifier(**XGB_PARAMS))])
    xgb_nt.fit(X_train, y_train)
    add("xgboost_no_hour", "ablation: drops hour of day", xgb_nt.predict_proba(X_test)[:, 1])

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--window", type=float, default=5.0, help="Concurrency window in minutes")
    args = parser.parse_args()

    df = load_preprocessed(args.data, window_minutes=args.window)
    X_train, y_train, X_test, y_test = split_data(df)
    print(f"train rows: {len(X_train)}, test rows: {len(X_test)}, "
          f"positive rate train/test: {y_train.mean():.3f} / {y_test.mean():.3f}\n")

    results = run_baselines(X_train, y_train, X_test, y_test)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    results.insert(0, "window_minutes", args.window)
    results.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()