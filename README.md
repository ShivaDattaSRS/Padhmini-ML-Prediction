# Supplier Delay and Production Loss Predictor

A Streamlit dashboard and Python pipeline for estimating shipment delay risk,
potential production loss, and alternative suppliers for automotive orders.

> The included CSVs contain simulated/demo data. Replace them with approved,
> representative shipment and supplier data before using results for business
> decisions.

## Project files

- `app1.py` — Streamlit dashboard with a risk watchlist and an order assessment
  form.
- `delay_pipeline.py` — data preparation, model training and prediction,
  alternative supplier ranking, and mitigation calculations.
- `automotive_supplier_delivery_data_realistic_100k.csv` — delivery history
  used by the dashboard.
- `automotive_alternative_supplier_data_realistic_100k.csv` — supplier offers,
  availability, price, lead time, and quality data.
- `models/delay_models.joblib` — trained model bundle used by the dashboard.

## Requirements

Use Python 3.10 or later. Install the packages used by the app and pipeline:

```powershell
python -m pip install streamlit pandas numpy joblib scikit-learn
```

LightGBM is optional. The training pipeline falls back to scikit-learn if
LightGBM is not installed. To use LightGBM when training:

```powershell
python -m pip install lightgbm
```

## Run the dashboard

Open PowerShell in this project folder (the folder containing `app1.py` and
`delay_pipeline.py`) and run:

```powershell
python -m streamlit run app1.py
```

Open the **Local URL** printed in the terminal, usually
`http://localhost:8501`. Use `Ctrl+C` in the terminal to stop the server.

The sidebar lets you set the delivery and supplier CSV paths, number of recent
orders to score, minimum supplier quality, and minimum savings required before
switching suppliers.

## Retrain the model

The included model bundle is ready to use with the included data. To train a
new bundle from CSV files in this folder:

```powershell
python delay_pipeline.py train --delivery automotive_supplier_delivery_data_realistic_100k.csv --alt automotive_alternative_supplier_data_realistic_100k.csv
```

Training writes the model bundle to `models/delay_models.joblib`. Retrain after
changing the pipeline or replacing the input data; model bundles saved by older
pipeline versions may not be compatible.

## Run the command-line demo

```powershell
python delay_pipeline.py demo --delivery automotive_supplier_delivery_data_realistic_100k.csv --alt automotive_alternative_supplier_data_realistic_100k.csv
```

The demo scores the most recent orders and prints the highest expected-loss
examples.

## How to interpret the dashboard

- **Risk watchlist** scores recent orders and sorts them by expected production
  loss. Select a shipment to see its prediction, possible alternatives, and
  mitigation options.
- **Check an order** lets you enter a proposed order and assess its delay risk
  and expected loss.
- Alternative suppliers are filtered by item, active status, availability,
  order quantity, lead time, and the selected quality threshold. Remaining
  candidates are compared by purchase cost, shipping, and predicted loss.
- Risk bands are High at 40% or greater, Medium from 20% to below 40%, and Low
  below 20%.

Predictions are estimates, not delivery guarantees. Confirm supplier capacity,
pricing, and delivery commitments before making a purchasing decision.
