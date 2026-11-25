"""Generate a synthetic Ola/Uber-style ride request dataset.

Each row = ride requests in one city zone during one hour.
Demand depends on zone type, hour of day, weekday/weekend, weather,
holidays and local events, plus random (Poisson) noise.
Replace data/ride_demand.csv with real trip data aggregated the same way
(timestamp, zone, demand, temp_c, rain, is_holiday, event) to reuse the project.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ZONES = {
    "Airport": ("airport", 14),
    "Business_District": ("business", 22),
    "Railway_Station": ("transit", 18),
    "University": ("campus", 10),
    "Residential_North": ("residential", 12),
    "Residential_South": ("residential", 11),
    "Mall_Area": ("leisure", 9),
    "Nightlife_Hub": ("nightlife", 8),
}
ZONE_LIST = list(ZONES)


def _g(h, centre, width):
    d = np.abs(h - centre)
    d = np.minimum(d, 24 - d)  # circular distance over 24 hours
    return np.exp(-0.5 * (d / width) ** 2)


def profile(kind, h, weekend):
    wk, we = (~weekend).astype(float), weekend.astype(float)
    if kind == "business":
        return 0.15 + wk * (1.2 * _g(h, 9, 1.5) + 1.3 * _g(h, 18.5, 1.7)) + we * 0.35 * _g(h, 13, 3)
    if kind == "residential":
        return 0.2 + wk * (1.1 * _g(h, 8, 1.6) + 0.7 * _g(h, 19.5, 2.2)) + we * (0.5 * _g(h, 11, 2.5) + 0.5 * _g(h, 20, 2.5))
    if kind == "airport":
        return 0.5 + 0.5 * _g(h, 5, 2) + 0.6 * _g(h, 14, 3) + 0.7 * _g(h, 22, 2)
    if kind == "transit":
        return 0.3 + wk * (0.9 * _g(h, 8.5, 1.6) + 0.9 * _g(h, 18, 1.8)) + we * 0.6 * _g(h, 12, 3)
    if kind == "campus":
        return 0.1 + wk * (0.9 * _g(h, 9, 1.3) + 0.7 * _g(h, 13, 1.5) + 0.8 * _g(h, 17, 1.5)) + we * 0.1
    if kind == "leisure":
        return 0.1 + we * (0.9 * _g(h, 15, 2.5) + 0.9 * _g(h, 19, 2)) + wk * 0.4 * _g(h, 19, 2)
    if kind == "nightlife":
        return 0.05 + 1.2 * _g(h, 23, 1.8) + (1.0 * we + 0.5 * (1 - we)) * _g(h, 1, 1.8)
    raise ValueError(kind)


def generate(days=120, start="2026-06-01", seed=42):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=days * 24, freq="h")
    n = len(ts)
    hour = np.asarray(ts.hour, dtype=float)
    weekend = np.asarray(ts.dayofweek >= 5)

    # City-wide weather: rain follows a persistent on/off process.
    rain = np.zeros(n)
    raining = False
    for i in range(n):
        raining = rng.random() < (0.85 if raining else 0.012)
        rain[i] = rng.gamma(2.0, 1.5) if raining else 0.0
    day_idx = np.arange(n) // 24
    temp = 30 + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + np.repeat(rng.normal(0, 1.5, days), 24)
    temp = temp - 2.5 * (rain > 0)

    holiday_days = set(rng.choice(days, size=4, replace=False))
    holiday = np.isin(day_idx, list(holiday_days)).astype(int)

    frames = []
    for zone, (kind, base) in ZONES.items():
        mult = 1 + 0.003 * (np.arange(n) / 24)  # slow growth over time
        lam = base * profile(kind, hour, weekend) * mult
        lam = lam * (1 + 0.30 * (rain > 0.5) * (kind in ("business", "transit", "residential")) + 0.10 * (rain > 0.5))
        lam = lam * (1 + 0.012 * np.clip(temp - 32, 0, None))
        if kind == "business":
            lam = lam * np.where(holiday == 1, 0.5, 1.0)
        if kind in ("leisure", "nightlife"):
            lam = lam * np.where(holiday == 1, 1.5, 1.0)

        event = np.zeros(n, dtype=int)
        if kind in ("leisure", "nightlife", "campus", "airport"):
            for d in rng.choice(days, size=8, replace=False):
                s = d * 24 + int(rng.integers(17, 21))
                event[s:s + 5] = 1
            lam = lam * np.where(event == 1, 1.8, 1.0)

        demand = rng.poisson(lam)
        frames.append(pd.DataFrame({
            "timestamp": ts, "zone": zone, "demand": demand,
            "temp_c": temp.round(1), "rain": rain.round(2),
            "is_holiday": holiday, "event": event,
        }))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = generate()
    df.to_csv("data/ride_demand.csv", index=False)
    print(f"Saved data/ride_demand.csv  rows={len(df):,}  zones={df.zone.nunique()}")
