from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.base import BaseEstimator
from typing import Callable, Any, Tuple, Dict
import pandas as pd
import numpy as np

from .data_preprocessing import load_preprocessed, split_data


@dataclass
class TrainConfig:
    """
    Configuration object for the training pipeline.

    Attributes:
        flight_csv (str):
            Path to the CSV file containing flight data.

        model_function (Callable[[], Tuple[BaseEstimator, Dict[str, Any]]]):
            A callable that returns a tuple '(model, params)' where:
              - model (BaseEstimator): An untrained scikit-learn estimator.
              - params (dict): A dictionary of hyperparameter candidates
                for grid search.

        cv_strategy (str):
            "time"  -> TimeSeriesSplit (default, avoids training on the future).
            "kfold" -> the original behaviour, cv=3 (stratified k-fold, no shuffle).
            Use "kfold" to reproduce the original baseline numbers.

        cv_splits (int):
            Number of splits for TimeSeriesSplit. Ignored for "kfold" (always 3).

        window_minutes (float):
            Concurrency window in minutes used when computing the target and features.

        use_cache (bool):
            Cache the preprocessed data in data/cache/ to avoid recomputing it.
    """
    flight_csv: str
    model_function: Callable
    cv_strategy: str = "time"
    cv_splits: int = 5
    window_minutes: float = 5.0
    use_cache: bool = True


def train_with_cv(cfg: TrainConfig) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Train a model with grid search and cross-validation (scored with ROC-AUC).

    Args:
        cfg (TrainConfig): Training configuration.

    Returns:
        Tuple[BaseEstimator, dict]:
            The best estimator, and an info dict with 'cv_auc', 'best_params'
            and 'cv_description'.
    """
    if cfg.cv_strategy not in ("time", "kfold"):
        raise ValueError("cv_strategy must be 'time' or 'kfold'")

    df = load_preprocessed(cfg.flight_csv,
                           window_minutes=cfg.window_minutes,
                           use_cache=cfg.use_cache)
    model, params = cfg.model_function()

    X_train, y_train, _x, _y = split_data(df)

    # preprocessing() sorts rows by date, hour and airport group, and split_data keeps
    # that order, so TimeSeriesSplit folds always validate on later data than they train on.
    if cfg.cv_strategy == "time":
        cv = TimeSeriesSplit(n_splits=cfg.cv_splits)
        cv_description = f"TimeSeriesSplit(n_splits={cfg.cv_splits})"
    else:
        cv = 3
        cv_description = "3-fold (original)"

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    print("CV strategy:", cv_description)
    print("Best parameters:", grid_search.best_params_)
    print("Best CV AUC-ROC score:", grid_search.best_score_)

    info = {
        "cv_auc": float(grid_search.best_score_),
        "best_params": grid_search.best_params_,
        "cv_description": cv_description,
    }
    return grid_search.best_estimator_, info


def train(cfg: TrainConfig) -> BaseEstimator:
    """
    Train a model with grid search and return only the best estimator.
    (Kept for compatibility with main.py; see `train_with_cv` for CV details.)
    """
    model, _info = train_with_cv(cfg)
    return model


def model_predict(data_path: str,
                  model: Any,
                  output_path: str = "data/predictions.csv",
                  window_minutes: float = 5.0) -> pd.DataFrame:
    """
    Run predictions using a trained model and save results to a CSV file.

    Args:
        data_path (str):
            Path to the input CSV file containing raw schedule data.
        model (Any):
            A trained ML model object that implements '.predict_proba()'.
        output_path (str):
            Where to save the predictions CSV (default data/predictions.csv).
        window_minutes (float):
            Must match the window the model was trained with.

    Returns:
        pd.DataFrame:
            A DataFrame containing columns:
                - airport_group (str): One-letter airport group code
                - date (str): Prediction date in format 'YYYY-MM-DD'
                - hour (int): Hour of the day (0-23)
                - pred (float): Predicted probability of concurrency
    """
    df = load_preprocessed(data_path, predict=True, window_minutes=window_minutes)

    prediction = np.round(model.predict_proba(df)[:, 1], 3)

    answer = pd.DataFrame({
        "airport_group": df["airport_group"],
        "date": df["date"],
        "hour": df["hour"],
        "pred": prediction,
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(output_path, encoding="UTF-8", index=False)

    # sort
    answer = answer.sort_values(by=["date", "hour", "airport_group"]).reset_index(drop=True)

    return answer