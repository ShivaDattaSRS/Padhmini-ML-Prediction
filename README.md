# Supplier Delay and Production Loss Predictor

A Streamlit dashboard and Python pipeline for estimating shipment delay risk,
potential production loss, and alternative suppliers for automotive orders.

> The included CSVs contain simulated/demo data. Replace them with approved,
> representative shipment and supplier data before using results for business
> decisions.

## Project files

- `app1.py` — Streamlit dashboard with a risk watchlist and an order assessment
  form.
- `delay_pipeline_quantile_calibrated.py` — data preparation, model training and prediction,
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
`delay_pipeline_quantile_calibrated.py`) and run:

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
python delay_pipeline_quantile_calibrated.py train --delivery automotive_supplier_delivery_data_realistic_100k.csv --alt automotive_alternative_supplier_data_realistic_100k.csv
```

This writes `models/delay_models.joblib` and chronological holdout metrics under
`evaluation/`. The LightGBM classifier uses early stopping on the validation
period; the threshold is selected on that period, delay quantiles are calibrated
on the following period, and the newest 20% is reserved for final evaluation.
When LightGBM is unavailable, the scikit-learn histogram-gradient-boosting
classifier is grown in warm-started batches and the best iteration is selected
by log loss on the chronological validation partition. Its internal random
early-stopping split is disabled.
Retrain after changing the pipeline or replacing the input data; model bundles
saved by older pipeline versions may not be compatible.

Training also fits a separate stock-out classifier (`Delay_Days >
Stock_On_Hand_Days`) and an excess-delay model for stock-out events. Its
probability, expected excess delay, and calibration-adjusted loss estimate are
shown separately in the dashboard and evaluation exports. This estimate is
experimental: the quantile-based expected loss remains the value used for
supplier rankings and mitigation until rolling chronological evaluation shows
a reliable improvement. `evaluation/metrics_summary.csv` reports stock-out
ranking scores, excess-delay error, and both stock-out loss errors and
calibration bias alongside the existing model results.
`evaluation/stockout_reliability.csv` bins final-test stock-out probabilities
and compares each bin's mean probability with its observed stock-out frequency;
the Brier score, expected calibration error, and largest bin gap are also in
`evaluation/metrics_summary.csv`. ROC-AUC is reported as a ranking metric, not
as evidence that probabilities are calibrated. Stock-out probabilities are
constrained to be no greater than late probabilities; reliability tables
evaluate their calibration rather than applying a separate probability
recalibration transformation.

The watchlist and recommendation replay process orders chronologically. At
each order date, only outcomes from shipments ordered earlier and delivered
before that date are available to the model. The replay's actual delivery
dates, delays, and losses are retained only for retrospective reporting and
are excluded from prediction features. Rows whose recorded actual delivery
precedes their order date are excluded from supplier-history features.
`evaluation/sequential_replay_metrics.csv` and
`evaluation/sequential_replay_stockout_reliability.csv` report the most recent
replay's probability and loss evaluation.

To regenerate the recent-order supplier recommendations using the current model:

```powershell
python run_supplier_recommendations.py --delivery automotive_supplier_delivery_data_realistic_100k.csv --alt automotive_alternative_supplier_data_realistic_100k.csv --model models/delay_models.joblib --pipeline delay_pipeline_quantile_calibrated.py --recent-n 300 --top-n 3 --out recommendation_output
```

The script refreshes the CSVs in `recommendation_output/`. Its savings are
retrospective model estimates, not confirmed savings; verify current supplier
quotes, stock, capacity, and delivery commitments before acting.
Savings are not calculated when the current supplier-item shipping quote is
missing, invalid, or ambiguous. Alternative offers with missing shipping cost
are excluded from cost ranking, and the recommendation status reports that
limitation instead of treating an unknown cost as zero.

## Generate error analysis and business-impact reports

After training the model and generating the sequential replay, run:

```powershell
python analyze_business_impact.py
```

This writes supplier/item error summaries, recent-period distribution and
performance comparisons, financial error contributions, business-impact
counts, and `evaluation/final_evaluation_report.md`. Delay-regression error is
measured only on actually late shipments. Group error screens include event
counts and minimum-sample rules. The latest replay is nested within the
chronological test period, so it is compared with the non-overlapping earlier
part of that test set for drift analysis; replay/test metrics are not
independent. Alternative savings are explicitly estimates, not realized
savings. The report does not change model training or supplier-ranking costs.

## Run the command-line demo

```powershell
python delay_pipeline_quantile_calibrated.py demo --delivery automotive_supplier_delivery_data_realistic_100k.csv --alt automotive_alternative_supplier_data_realistic_100k.csv
```

The demo scores the most recent orders and prints the highest expected-loss
examples.

## How to interpret the dashboard

- **Risk watchlist** scores recent orders and sorts them by expected production
  loss. Select a shipment to see its prediction, possible alternatives, and
  mitigation options.
- **Check an order** lets you enter a proposed order, the supplier-promised
  expected delivery date, and stock on hand, then assess its delay risk,
  expected loss, and alternative suppliers. The dashboard distinguishes
  expected delay across all orders from estimated delay when a shipment is late.
- Stock-out probability is distinct from late probability: it estimates the
  chance that the delivery delay exceeds available stock. The experimental
  stock-out loss is shown separately and does not drive supplier cost rankings.
- Alternative suppliers are filtered by item, active status, availability,
  order quantity, lead time, and the selected quality threshold. Remaining
  candidates are compared by purchase cost, shipping, and predicted loss.
- High-risk threshold is selected from the chronological validation data;
  medium risk is defined relative to that threshold and shown in the dashboard.

Predictions are estimates, not delivery guarantees. Confirm supplier capacity,
pricing, and delivery commitments before making a purchasing decision.
