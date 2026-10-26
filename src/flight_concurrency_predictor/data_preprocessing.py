import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def compute_concurrency(times: pd.Series, window_minutes: float = 5.0) -> int:
    """
    Count pairs of events (i < j in time order) that occur within `window_minutes`
    of each other.

    Behaviour is identical to the original implementation, which looped over all
    pairs and tested -16 <= delta <= 5 minutes. Because the times are sorted and
    only later events are compared, delta is never negative, so the -16 bound
    never mattered: the effective window was 0..5 minutes. That window is now an
    explicit parameter (`window_minutes`) so the concurrency definition can be
    varied in experiments. This version is vectorised (O(n log n) instead of O(n^2)).

    Args:
        times (pd.Series): Series of datetime objects.
        window_minutes (float): Maximum gap in minutes between two events for
            them to count as concurrent. Default 5.0 (the original behaviour).

    Returns:
        int: Number of concurrent event pairs.
    """
    times = times.dropna().sort_values()
    n = len(times)
    if n < 2:
        return 0

    # Seconds since first event: independent of datetime resolution and timezone
    secs = (times - times.iloc[0]).dt.total_seconds().to_numpy()
    # For each i: index of the first event later than secs[i] + window
    upper = np.searchsorted(secs, secs + window_minutes * 60.0, side="right")
    # Events j > i inside the window: indices i+1 .. upper-1
    return int((upper - (np.arange(n) + 1)).sum())


def get_season(date: date) -> str:
    """
    Return the meteorological season for a given date.
    
    Args:
        date (date): A datetime.date object.
    
    Returns:
        str: Season name ('Winter', 'Spring', 'Summer', 'Fall').
    """

    month = date.month
    day = date.day
    if (month == 12 and day >= 21) or (month <= 3 and (month != 3 or day <= 19)):
        return 'Winter'
    elif (month == 3 and day >= 20) or (month in [4, 5]) or (month == 6 and day <= 20):
        return 'Spring'
    elif (month == 6 and day >= 21) or (month in [7, 8]) or (month == 9 and day <= 21):
        return 'Summer'
    else:
        return 'Fall'


def preprocessing(df: pd.DataFrame, predict: bool = False, window_minutes: float = 5.0) -> pd.DataFrame:
    """
    Preprocess flight data to compute scheduled and actual concurrency metrics,
    and annotate with season and other derived columns.

    Args:
        df (DataFrame): Raw flight data containing columns like
                        'dep_airport_group', 'arr_airport_group', 'std', 'atd', 'sta', 'ata', 'cancelled'.
        predict (bool): If True, ignores actual datetime/hour and concurrency calculations,
                        should be True if the dataset is going to be prepared for testing / predicting
        window_minutes (float): Window (minutes) used to define concurrency. Default 5.0.

    Returns:
        DataFrame: Aggregated flight data grouped by scheduled date, hour, and airport group,
                   including concurrency and season.
    """

    # Drop unnecessary columns
    df_reduced = df.drop(columns = ["service_type", "dep_airport", "arr_airport"], errors = 'ignore')

    # Remove cancelled flights and the whole coloumn 
    # because this will be another prediction we do not want to predict
    if not predict:
        df_cleaned = df_reduced[df_reduced["cancelled"] != 1].copy()
        df_cleaned = df_cleaned.drop(columns = "cancelled", errors = 'ignore')

        # Departures dataframe
        df_departures = df_cleaned[["dep_airport_group", "std", "atd"]].copy()
        df_departures = df_departures.rename(columns = {
            "dep_airport_group": "airport_group",
            "std": "datetime",
            "atd": "actual_datetime"
        })

        # Arrivals dataframe
        df_arrivals = df_cleaned[["arr_airport_group", "sta", "ata"]].copy()
        df_arrivals = df_arrivals.rename(columns={
            "arr_airport_group": "airport_group",
            "sta": "datetime",
            "ata": "actual_datetime"
        })

    else:
        df_cleaned = df_reduced.copy()

        # Departures dataframe
        df_departures = df_cleaned[["dep_airport_group", "std"]].copy()
        df_departures = df_departures.rename(columns = {
            "dep_airport_group": "airport_group",
            "std": "datetime"
        })

        # Arrivals dataframe
        df_arrivals = df_cleaned[["arr_airport_group", "sta"]].copy()
        df_arrivals = df_arrivals.rename(columns={
            "arr_airport_group": "airport_group",
            "sta": "datetime"
        })

    # Combine departures and arrivals
    df_events = pd.concat([df_departures, df_arrivals], ignore_index = True)

    if predict:
        df_events["datetime"] = pd.to_datetime(df_events["datetime"], errors = "coerce")

        df_events["sched_date"] = df_events["datetime"].dt.date
        df_events["sched_hour"] = df_events["datetime"].dt.hour
    else:
        df_events["datetime"] = pd.to_datetime(df_events["datetime"], errors = "coerce")
        df_events["actual_datetime"] = pd.to_datetime(df_events["actual_datetime"], errors="coerce")

        df_events["sched_date"] = df_events["datetime"].dt.date
        df_events["sched_hour"] = df_events["datetime"].dt.hour
        df_events["actual_date"] = df_events["actual_datetime"].dt.date
        df_events["actual_hour"] = df_events["actual_datetime"].dt.hour

    # Group by scheduled date, hour, and airport group
    def group_func(g):
        if predict:
            return pd.Series({
                "sched_flights": len(g),
                "sched_concurrence": compute_concurrency(g["datetime"], window_minutes),
            })
        else:
            return pd.Series({
                "sched_flights": len(g),
                "sched_concurrence": compute_concurrency(g["datetime"], window_minutes),
                "actual_concurrence": compute_concurrency(g["actual_datetime"], window_minutes)
            })

    df_grouped = (
        df_events.groupby(["sched_date", "sched_hour", "airport_group"])
        .apply(group_func, include_groups = False)
        .reset_index()
        .rename(columns={"sched_date": "date", "sched_hour": "hour"})
        .sort_values(by=["date", "hour", "airport_group"])
    )

    # Flag concurrency presence
    # removing number of concurrence since this is not important for the prediction
    if not predict:
        df_grouped["concurrency"] = (df_grouped["actual_concurrence"] > 0).astype(int)
        df_grouped = df_grouped.drop(columns = "actual_concurrence")

    # Add season
    df_grouped["season"] = df_grouped["date"].apply(get_season)

    # Reorder columns to place season after hour
    cols = df_grouped.columns.tolist()
    cols.insert(cols.index("hour") + 1, cols.pop(cols.index("season")))
    df_grouped = df_grouped[cols]

    return df_grouped


def load_preprocessed(csv_path: str,
                      predict: bool = False,
                      window_minutes: float = 5.0,
                      cache_dir: str = "data/cache",
                      use_cache: bool = True) -> pd.DataFrame:
    """
    Read a raw flight CSV and return the preprocessed DataFrame, caching the result.

    Preprocessing is slow, so the result is stored as a pickle in `cache_dir`.
    The cache is rebuilt automatically if the CSV is newer than the cache file,
    and the file name includes the concurrency window so different definitions
    never share a cache. Add `data/cache/` to .gitignore.

    Args:
        csv_path (str): Path to the raw CSV file.
        predict (bool): True for schedule-only data without actual times / labels.
        window_minutes (float): Concurrency window, see `compute_concurrency`.
        cache_dir (str): Folder for cached files.
        use_cache (bool): Set False to force recomputation and skip caching.

    Returns:
        pd.DataFrame: Preprocessed data (same output as `preprocessing`).
    """
    csv_path = Path(csv_path)
    kind = "pred" if predict else "train"
    cache_file = Path(cache_dir) / f"{csv_path.stem}_{kind}_w{window_minutes:g}.pkl"

    if use_cache and cache_file.exists() and cache_file.stat().st_mtime >= csv_path.stat().st_mtime:
        return pd.read_pickle(cache_file)

    df = preprocessing(pd.read_csv(csv_path), predict=predict, window_minutes=window_minutes)

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_file)
    return df


def split_data(df: pd.DataFrame, cutoff: str = "2024-07-01") -> tuple:
    """
    Split the dataset into training and testing sets using time series split.
    
    Args:
        df (DataFrame): The preprocessed flight data.
        cutoff (str): default = "2024-07-01" Date cutoff for splitting into train/test sets.

    Returns:
        tuple: A tuple containing the training and testing DataFrames.
    """
    target_col = "concurrency"

    df["date"] = pd.to_datetime(df["date"])

    # Split into train and test
    train = df[df["date"] < pd.Timestamp(cutoff)].copy()
    test = df[df["date"] >= pd.Timestamp(cutoff)].copy()

    # Separate features and target
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    return X_train, y_train, X_test, y_test

def pipeline_preprocessor() -> ColumnTransformer:
    """
    Create a preprocessing pipeline for the flight data.
    
    Returns:
        ColumnTransformer: A column transformer that applies scaling to numerical features
                          and one-hot encoding to categorical features.
    """
    
    numeric_features = ["sched_flights", "sched_concurrence"]
    categorical_features = ["airport_group", "season", "hour"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown = "ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor