"""Forecast the next N hours of ride demand for a zone.

Usage:  python predict.py --zone Airport --hours 6
Uses the model saved by train.py and forecasts recursively
(each predicted hour becomes a lag feature for the next one).
Future weather is assumed dry and no events are assumed.
"""
import argparse
import joblib
import numpy as np
import pandas as pd

from features import add_features
from generate_data import ZONE_LIST


def forecast(df, bundle, zone, hours):
    model, cols = bundle["model"], bundle["features"]
    hist = df[df.zone == zone].sort_values("timestamp").tail(400).reset_index(drop=True)
    hist = hist[["timestamp", "zone", "demand", "temp_c", "rain", "is_holiday", "event"]].copy()
    out = []
    for _ in range(hours):
        t = hist.timestamp.iloc[-1] + pd.Timedelta(hours=1)
        new = pd.DataFrame({"timestamp": [t], "zone": [zone], "demand": [np.nan],
                            "temp_c": [hist.temp_c.iloc[-1]], "rain": [0.0],
                            "is_holiday": [0], "event": [0]})
        tmp = pd.concat([hist, new], ignore_index=True)
        X = add_features(tmp).tail(1)[cols]
        pred = float(max(0.0, model.predict(X)[0]))
        tmp.loc[tmp.index[-1], "demand"] = pred
        hist = tmp
        out.append((t, pred))
    return pd.DataFrame(out, columns=["timestamp", "predicted_requests"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="Business_District", choices=ZONE_LIST)
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--data", default="data/ride_demand.csv")
    a = ap.parse_args()
    data = pd.read_csv(a.data, parse_dates=["timestamp"])
    bundle = joblib.load("models/best_model.joblib")
    res = forecast(data, bundle, a.zone, a.hours)
    res["predicted_requests"] = res["predicted_requests"].round(1)
    print(f"Model: {bundle['name']}\nZone: {a.zone}\n")
    print(res.to_string(index=False))
