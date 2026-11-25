# Ride Demand Forecasting (Ola / Uber style)

Predict how many ride requests each city zone will receive in the next hour, so
drivers can be positioned where riders will be and pricing can react early.

The project covers the full machine learning workflow: data, feature engineering,
baselines, model comparison, time-aware evaluation, plots and a forecasting script.

## Project structure

| File | Purpose |
|------|---------|
| `generate_data.py` | Creates a synthetic hourly ride-request dataset for 8 city zones (120 days) |
| `features.py` | Calendar, lag, rolling-window and zone features (no data leakage) |
| `train.py` | Trains and compares 5 models, writes metrics, plots and the best model |
| `predict.py` | Recursive multi-hour forecast for a chosen zone |
| `data/ride_demand.csv` | Generated dataset |
| `results/` | `metrics.csv` and four plots |
| `models/best_model.joblib` | Saved best model |

## How to run

```bash
pip install -r requirements.txt
python generate_data.py          # creates data/ride_demand.csv
python train.py                  # trains models, saves results/ and models/
python predict.py --zone Airport --hours 6
```

## Problem formulation

- **Target:** number of ride requests in a zone during one hour
- **Type:** time-series regression (count prediction)
- **Horizon:** next hour (`predict.py` extends this to several hours recursively)
- **Data schema:** `timestamp, zone, demand, temp_c, rain, is_holiday, event`

## Features

- **Calendar:** hour, day of week, weekend flag, sine/cosine encodings of hour and weekday
- **Lag demand:** 1, 2, 3, 24, 48 and 168 hours earlier (same zone)
- **Rolling statistics:** 3-hour mean, 24-hour mean and 24-hour standard deviation
- **External:** temperature, rain, public holiday, local event
- **Zone:** one-hot encoding of the zone

All lag and rolling features use only past hours, so nothing from the future leaks into training.

## Models

1. Seasonal naive (demand at the same hour last week)
2. Historical average (zone x hour x weekday mean)
3. Ridge regression
4. Random Forest
5. Gradient Boosting with a Poisson loss (suited to count data)

## Evaluation

The last 21 days are held out as the test set (chronological split, no shuffling).
Metrics: MAE, RMSE, sMAPE and R2. sMAPE is used instead of MAPE because many
hours have very low or zero demand.

Results from one run on the generated data:

| Model | MAE | RMSE | sMAPE % | R2 |
|-------|-----|------|---------|----|
| Gradient Boosting (Poisson) | 2.607 | 3.704 | 46.9 | 0.826 |
| Random Forest | 2.674 | 3.826 | 47.9 | 0.814 |
| Historical average | 2.663 | 3.990 | 48.2 | 0.798 |
| Ridge regression | 2.883 | 4.037 | 50.2 | 0.793 |
| Seasonal naive (last week) | 3.371 | 4.904 | 55.3 | 0.695 |

Gradient boosting performs best because it captures non-linear effects such as
rush-hour peaks, rain and events. Ridge regression cannot represent those
interactions as well. The seasonal naive baseline is weakest because it copies
last week's random noise.

## Important note on the data

The dataset is **synthetic**. It follows realistic patterns (office rush hours,
airport and nightlife peaks, rain and holiday effects), but it is not real Ola or
Uber data, so the scores above describe this simulated city only. To use real data,
aggregate trip records into hourly counts per zone (for example with H3 hexagons or
a geohash grid), save them in the same schema as `data/ride_demand.csv`, and run
`python train.py`.

## Ideas for extension

- Add zone-neighbour demand as a feature (spatial effect)
- Try LSTM/GRU or graph neural network models
- Predict 15-minute intervals and several horizons at once
- Add prediction intervals to express uncertainty
- Tune hyperparameters with time-series cross-validation
