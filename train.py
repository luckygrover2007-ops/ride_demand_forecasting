"""Train and compare ride-demand forecasting models.

Run:  python train.py
Outputs: results/metrics.csv, results/*.png, models/best_model.joblib
"""
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS, add_features
from generate_data import generate

TEST_DAYS = 21
Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = (np.abs(y) + np.abs(p)) / 2
    return float(np.mean(np.where(denom == 0, 0, np.abs(y - p) / np.where(denom == 0, 1, denom))) * 100)


def score(name, y, p):
    return {"model": name,
            "MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "sMAPE_%": smape(y, p),
            "R2": r2_score(y, p)}


def main():
    csv = Path("data/ride_demand.csv")
    if csv.exists():
        raw = pd.read_csv(csv, parse_dates=["timestamp"])
    else:
        raw = generate()
    df = add_features(raw).dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Chronological split: the last TEST_DAYS days are held out (never shuffle time series).
    cutoff = df["timestamp"].max() - pd.Timedelta(days=TEST_DAYS)
    train, test = df[df.timestamp <= cutoff], df[df.timestamp > cutoff]
    Xtr, ytr, Xte, yte = train[FEATURE_COLS], train["demand"], test[FEATURE_COLS], test["demand"]
    print(f"train rows={len(train):,}  test rows={len(test):,}  test starts {test.timestamp.min()}")

    preds = {}
    # Baseline 1: demand one week earlier in the same zone.
    preds["Seasonal naive (last week)"] = test["lag_168"].values
    # Baseline 2: average of the same zone / hour / weekday in the training period.
    avg = train.groupby(["zone", "hour", "dow"])["demand"].mean().rename("avg")
    preds["Historical average"] = test.join(avg, on=["zone", "hour", "dow"])["avg"].fillna(train.demand.mean()).values

    models = {
        "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=42),
        "Gradient Boosting (Poisson)": HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.06, max_iter=400, max_leaf_nodes=31, random_state=42),
    }
    fitted = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        preds[name] = np.clip(m.predict(Xte), 0, None)
        fitted[name] = m

    metrics = pd.DataFrame([score(n, yte, p) for n, p in preds.items()]).sort_values("RMSE").round(3)
    metrics.to_csv("results/metrics.csv", index=False)
    print("\n" + metrics.to_string(index=False))

    best = metrics[metrics.model.isin(fitted)].iloc[0]["model"]
    joblib.dump({"model": fitted[best], "features": FEATURE_COLS, "name": best}, "models/best_model.joblib")
    print(f"\nBest ML model: {best} (saved to models/best_model.joblib)")

    # ---------- plots ----------
    pivot = df.pivot_table(index="zone", columns="hour", values="demand", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("Hour of day"); ax.set_title("Average ride requests per hour, by zone")
    fig.colorbar(im, label="requests / hour"); fig.tight_layout(); fig.savefig("results/demand_heatmap.png", dpi=150); plt.close(fig)

    order = metrics.sort_values("RMSE", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(order.model, order.RMSE, color="#5A3FD6"); ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Model comparison on held-out test period"); fig.tight_layout()
    fig.savefig("results/model_comparison.png", dpi=150); plt.close(fig)

    zone = "Business_District"
    sel = (test.zone == zone).values
    t7 = test[sel].timestamp.values
    last_week = t7 >= (test.timestamp.max() - pd.Timedelta(days=7)).to_datetime64()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t7[last_week], yte.values[sel][last_week], label="Actual", color="#222", lw=1.6)
    ax.plot(t7[last_week], preds[best][sel][last_week], label=f"Predicted ({best})", color="#FF7A59", lw=1.6)
    ax.set_title(f"{zone}: actual vs predicted demand, last 7 days"); ax.legend(); ax.set_ylabel("requests / hour")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig("results/actual_vs_predicted.png", dpi=150); plt.close(fig)

    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(imp.index, imp.values, color="#2EC4B6"); ax.set_title("Random Forest: top feature importances")
    fig.tight_layout(); fig.savefig("results/feature_importance.png", dpi=150); plt.close(fig)
    print("Plots saved in results/")


if __name__ == "__main__":
    main()
