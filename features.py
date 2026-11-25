"""Feature engineering shared by training and prediction."""
import numpy as np
import pandas as pd
from generate_data import ZONE_LIST

LAGS = (1, 2, 3, 24, 48, 168)

FEATURE_COLS = (
    ["hour", "dow", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
     "temp_c", "rain", "is_holiday", "event"]
    + [f"lag_{l}" for l in LAGS]
    + ["roll_mean_3", "roll_mean_24", "roll_std_24"]
    + [f"zone_{z}" for z in ZONE_LIST]
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, lag, rolling and zone features.

    Every lag/rolling feature uses only demand from *earlier* hours, so a row's
    features are all known one hour before the prediction time (no leakage).
    """
    df = df.sort_values(["zone", "timestamp"]).reset_index(drop=True).copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    g = df.groupby("zone")["demand"]
    for l in LAGS:
        df[f"lag_{l}"] = g.shift(l)
    df["roll_mean_3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    df["roll_mean_24"] = g.transform(lambda s: s.shift(1).rolling(24).mean())
    df["roll_std_24"] = g.transform(lambda s: s.shift(1).rolling(24).std())

    for z in ZONE_LIST:
        df[f"zone_{z}"] = (df["zone"] == z).astype(int)
    return df
