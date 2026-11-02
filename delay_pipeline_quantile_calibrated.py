"""
Automobile supplier delay pipeline
  1. load + join       2. leak-free history features
  3. train (LightGBM; falls back to scikit-learn if LightGBM is not installed)
  4. predict delay + loss     5. rank alternative suppliers / mitigation options

Usage
  python delay_pipeline_quantile_calibrated.py train --delivery DELIVERY.csv --alt ALT.csv
  python delay_pipeline_quantile_calibrated.py demo  --delivery DELIVERY.csv --alt ALT.csv

CHANGE vs previous version
  Expected loss used ONE typical delay (L1 regressor) inside max(0, delay - stock) * loss/day.
  Loss is convex (zero until stock runs out), so that systematically under-states loss (Jensen's inequality).
  Now expected loss = P(late) * average of the loss over several quantiles of the delay distribution.
  Models saved with the old version are incompatible -> retrain with `train`.
"""
import argparse, copy, inspect, json, os
import numpy as np, pandas as pd, joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, mean_absolute_error,
    mean_squared_error, r2_score, confusion_matrix, precision_score,
    recall_score, f1_score, accuracy_score, brier_score_loss, log_loss,
    precision_recall_curve
)

MODEL_DIR = "models"
CAT_COLS = ["Supplier_Location", "Part_Type"]
FEATURES = ["promised_lt", "buffer", "Quantity_Ordered", "Unit_Cost_INR", "order_month",
            "Lead_Time_Days", "Quality_Rating", "Supplier_Location", "Part_Type",
            "h_n", "h_late", "h_delay", "h_recent"]
STOCKOUT_FEATURES = [*FEATURES, "Stock_On_Hand_Days"]

# Quantiles of "days late, given late". Mid-points of 5 equal-probability bins (each bin = 20% of outcomes),
# so a plain average over them approximates the average over the whole delay distribution.
QUANTS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Non-supplier fixes: cost as a share of order value, share of the loss they remove.
# Replace with the client's real numbers.
MITIGATION = {
    "Expedite by air freight": {"cost_pct": 0.18, "recovery": 0.70},
    "Switch transport mode":   {"cost_pct": 0.10, "recovery": 0.40},
    "Use buffer / safety stock": {"cost_pct": 0.02, "recovery": 0.50},
    "Reschedule production line": {"cost_pct": 0.04, "recovery": 0.50},
}


# ------------------------------------------------------------------ 1. load + join
def load_data(delivery_path, alt_path):
    d = pd.read_csv(delivery_path, parse_dates=["Order_Date", "Promised_Date", "Actual_Delivery_Date"])
    a = pd.read_csv(alt_path)
    return d, a


def attach_static(d, a):
    """Add supplier/part attributes and the supplier's quoted lead time for this item."""
    sup = a.groupby("Supplier_ID").agg(Supplier_Location=("Supplier_Location", lambda s: s.mode().iat[0]),
                                       Quality_Rating=("Quality_Rating", "mean"),
                                       sup_lt=("Lead_Time_Days", "median")).reset_index()
    part = a.groupby("Item_ID").Part_Type.first().reset_index()
    pair = a[["Supplier_ID", "Item_ID", "Lead_Time_Days"]].drop_duplicates(["Supplier_ID", "Item_ID"])
    d = d.merge(sup, on="Supplier_ID", how="left").merge(part, on="Item_ID", how="left") \
         .merge(pair, on=["Supplier_ID", "Item_ID"], how="left")
    d["Lead_Time_Days"] = d.Lead_Time_Days.fillna(d.sup_lt).fillna(a.Lead_Time_Days.median())
    d["promised_lt"] = (d.Promised_Date - d.Order_Date).dt.days
    d["buffer"] = d.promised_lt - d.Lead_Time_Days
    d["order_month"] = d.Order_Date.dt.month
    return d


# ------------------------------------------------------------------ 2. history features
def history_features(target, hist):
    """For every row in `target`, use only shipments in `hist` DELIVERED BEFORE that order date.
    This removes look-ahead leakage (the supplier's future is never visible)."""
    out = pd.DataFrame(index=target.index, columns=["h_n", "h_late", "h_delay", "h_recent"], dtype=float)
    valid_history = hist[
        hist.Actual_Delivery_Date.notna()
        & (hist.Actual_Delivery_Date >= hist.Order_Date)
    ]
    prior_late = (valid_history.Delay_Days > 0).astype(float)
    h = valid_history.assign(_late=prior_late).sort_values("Actual_Delivery_Date")
    groups = {k: g for k, g in h.groupby("Supplier_ID")}
    for sid, t in target.groupby("Supplier_ID"):
        g = groups.get(sid)
        if g is None:
            out.loc[t.index] = [0, np.nan, np.nan, np.nan]
            continue
        dates = g.Actual_Delivery_Date.values
        cl = np.concatenate([[0], g._late.cumsum().values])
        cd = np.concatenate([[0], g.Delay_Days.cumsum().values])
        idx = np.searchsorted(dates, t.Order_Date.values, side="left")   # shipments delivered before order
        n = idx.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.loc[t.index, "h_n"] = n
            out.loc[t.index, "h_late"] = np.where(n > 0, cl[idx] / n, np.nan)
            out.loc[t.index, "h_delay"] = np.where(n > 0, cd[idx] / n, np.nan)
            k = np.minimum(idx, 5).astype(float)
            out.loc[t.index, "h_recent"] = np.where(k > 0, (cl[idx] - cl[np.maximum(idx - 5, 0)]) / k, np.nan)
    return pd.concat([target, out], axis=1)


def encode(df, cat_maps):
    X = df[FEATURES].copy()
    for c in CAT_COLS:
        X[c] = X[c].map({v: i for i, v in enumerate(cat_maps[c])}).fillna(-1).astype(int)
    return X.astype(float)


def encode_stockout(df, cat_maps):
    X = encode(df, cat_maps)
    X["Stock_On_Hand_Days"] = pd.to_numeric(
        df["Stock_On_Hand_Days"], errors="raise"
    ).to_numpy()
    return X[STOCKOUT_FEATURES]


# ------------------------------------------------------------------ 3. models
def make_models():
    """Returns: classifier (late or not), regressor (typical days late), list of quantile regressors (one per QUANTS)."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        shared_kw = dict(
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        clf = LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=50,
            **shared_kw,
        )
        reg_kw = dict(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            **shared_kw,
        )
        reg = LGBMRegressor(objective="regression_l1", **reg_kw)  # typical days late (median-like)
        qmodels = [
            LGBMRegressor(objective="quantile", alpha=q, **reg_kw)
            for q in QUANTS
        ]
        print("using LightGBM")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier as C, HistGradientBoostingRegressor as R
        kw = dict(max_iter=300, learning_rate=0.05, random_state=42)
        clf = C(**kw); reg = R(loss="absolute_error", **kw)
        qmodels = [R(loss="quantile", quantile=q, **kw) for q in QUANTS]
        print("LightGBM not installed -> using scikit-learn HistGradientBoosting (same idea)")
    return clf, reg, qmodels


def make_stockout_models():
    """Build a stock-out classifier and a positive excess-delay regressor."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        shared_kw = dict(
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        stockout_clf = LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=50,
            **shared_kw,
        )
        excess_model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            **shared_kw,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        shared_kw = dict(max_iter=300, learning_rate=0.05, random_state=42)
        stockout_clf = HistGradientBoostingClassifier(**shared_kw)
        excess_model = HistGradientBoostingRegressor(
            loss="absolute_error", **shared_kw
        )
    return stockout_clf, excess_model


def fit_with_early_stopping(model, X_train, y_train, X_valid, y_valid):
    if len(X_valid) == 0:
        raise ValueError("Chronological validation data is required for early stopping.")
    if model.__class__.__module__.startswith("lightgbm"):
        from lightgbm import early_stopping

        validation_args = (
            {"eval_X": X_valid, "eval_y": y_valid}
            if "eval_X" in inspect.signature(model.fit).parameters
            else {"eval_set": [(X_valid, y_valid)]}
        )
        model.fit(
            X_train,
            y_train,
            callbacks=[early_stopping(60, verbose=False)],
            **validation_args,
        )
        return int(model.best_iteration_)

    if model.__class__.__name__ == "HistGradientBoostingClassifier":
        max_iterations = int(model.max_iter)
        step = min(10, max_iterations)
        model.set_params(early_stopping=False, warm_start=True, max_iter=step)
        best_model = None
        best_loss = np.inf
        best_iteration = 0
        stale_iterations = 0

        for iteration in range(step, max_iterations + step, step):
            current_iteration = min(iteration, max_iterations)
            model.set_params(max_iter=current_iteration)
            model.fit(X_train, y_train)
            validation_probability = model.predict_proba(X_valid)
            validation_loss = log_loss(
                y_valid,
                validation_probability,
                labels=model.classes_,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_iteration = current_iteration
                best_model = copy.deepcopy(model)
                stale_iterations = 0
            else:
                stale_iterations += step
            if stale_iterations >= 60 or current_iteration == max_iterations:
                break

        if best_model is None:
            raise RuntimeError(
                "Validation-based early stopping did not produce a fitted classifier."
            )
        model.__dict__.update(best_model.__dict__)
        return best_iteration

    raise TypeError(
        "Early stopping is only implemented for LightGBM classifiers and "
        "scikit-learn HistGradientBoostingClassifier."
    )


def loss_formula(delay_days, stock_days, loss_per_day):
    return np.maximum(0, np.asarray(delay_days) - np.asarray(stock_days)) * np.asarray(loss_per_day)


def predict_quantiles(X, qmodels, offsets=None):
    """Predict calibrated days-late quantiles. Offsets are learned on a separate calibration period."""
    d = np.column_stack([m.predict(X) for m in qmodels])
    if offsets is not None:
        d = d + np.asarray(offsets, dtype=float)[None, :]
    return np.maximum(1, d)


def delay_draws(X, qmodels, offsets=None):
    """Days-late predictions at each quantile, sorted per row for loss calculations."""
    return np.sort(predict_quantiles(X, qmodels, offsets), axis=1)


def expected_loss(p, draws, stock, loss_day):
    """E[loss] = P(late) * average over the delay distribution of max(0, delay - stock) * loss_per_day.
    Averaging the loss (not the delay) keeps the 'stock runs out' tail that a single typical delay would hide."""
    stock = np.asarray(stock, dtype=float)[:, None]
    loss_day = np.asarray(loss_day, dtype=float)[:, None]
    per_draw = np.maximum(0, draws - stock) * loss_day
    return np.asarray(p) * per_draw.mean(axis=1)


def stockout_probabilities(stockout_model, X, late_probability):
    """Keep the nested-event constraint P(stock-out) <= P(late) explicit."""
    raw_probability = stockout_model.predict_proba(X)[:, 1]
    return np.minimum(raw_probability, np.asarray(late_probability, dtype=float))


def stockout_reliability_table(y_true, probability, bins=10):
    """Summarize observed stock-out frequency against probability bins."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y_true) != len(probability):
        raise ValueError("Labels and stock-out probabilities must have equal lengths.")
    if bins < 1:
        raise ValueError("bins must be at least one.")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Stock-out probabilities must be finite values between 0 and 1.")

    bin_indices = np.minimum((probability * bins).astype(int), bins - 1)
    rows = []
    for bin_index in range(bins):
        selected = bin_indices == bin_index
        count = int(selected.sum())
        rows.append({
            "bin": bin_index + 1,
            "probability_lower": bin_index / bins,
            "probability_upper": (bin_index + 1) / bins,
            "shipment_count": count,
            "mean_predicted_probability": (
                float(probability[selected].mean()) if count else np.nan
            ),
            "observed_stockout_frequency": (
                float(y_true[selected].mean()) if count else np.nan
            ),
            "absolute_calibration_gap": (
                float(abs(probability[selected].mean() - y_true[selected].mean()))
                if count else np.nan
            ),
        })
    return pd.DataFrame(rows)


def stockout_calibration_metrics(reliability):
    populated = reliability[reliability.shipment_count > 0]
    if populated.empty:
        return {"expected_calibration_error": np.nan, "max_calibration_error": np.nan}
    weights = populated.shipment_count / populated.shipment_count.sum()
    gaps = populated.absolute_calibration_gap
    return {
        "expected_calibration_error": float((weights * gaps).sum()),
        "max_calibration_error": float(gaps.max()),
    }


def sequential_historical_replay(replay, initial_history, alternatives, bundle):
    """Score orders chronologically, admitting replay outcomes only after delivery."""
    if replay.empty:
        return pd.DataFrame()
    required = {
        "Shipment_ID", "Order_Date", "Actual_Delivery_Date",
        "Delay_Days", "Production_Loss_Cost_INR",
    }
    missing = required.difference(replay.columns)
    if missing:
        raise ValueError(
            "Sequential replay requires outcome columns: "
            + ", ".join(sorted(missing))
        )

    ordered = replay.sort_values(
        ["Order_Date", "Shipment_ID"], kind="stable"
    ).reset_index(drop=True)
    # Build order-time history snapshots once: only deliveries strictly before
    # each order date enter the prefix, equivalent to admitting outcomes on arrival.
    history_pool = pd.concat(
        [initial_history, ordered], ignore_index=True
    )
    outcome_columns = [
        "Actual_Delivery_Date", "Delay_Days", "Production_Loss_Cost_INR",
        "late", "stockout", "excess_delay_days",
    ]
    orders = ordered.drop(columns=outcome_columns, errors="ignore").copy()
    orders["Order_Date"] = pd.to_datetime(orders["Order_Date"])
    orders["Promised_Date"] = pd.to_datetime(orders["Promised_Date"])
    prepared = attach_static(orders, alternatives)
    prepared = history_features(prepared, history_pool)
    predictions = _predict_prepared_orders(prepared, bundle)
    predictions["replay_sequence"] = np.arange(len(predictions))
    return predictions



def _classification_metrics(y_true, probability, threshold):
    """Calculate threshold and probability metrics without changing the threshold."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    recall = recall_score(y_true, prediction, zero_division=0)
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probability))
        result["average_precision"] = float(average_precision_score(y_true, probability))
        result["brier_score"] = float(brier_score_loss(y_true, probability))
        result["log_loss"] = float(log_loss(y_true, probability, labels=[0, 1]))
    else:
        result.update({"roc_auc": np.nan, "average_precision": np.nan,
                       "brier_score": np.nan, "log_loss": np.nan})
    return result, prediction


def _choose_threshold(y_true, probability):
    """Choose the F1-maximising threshold on validation data only."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (2 * precision[:-1] * recall[:-1] /
                 np.maximum(precision[:-1] + recall[:-1], 1e-12))
    return float(thresholds[int(np.nanargmax(f1_values))])


def train(delivery_path, alt_path):
    d, a = load_data(delivery_path, alt_path)
    invalid_history_rows = int(
        (
            d.Actual_Delivery_Date.isna()
            | (d.Actual_Delivery_Date < d.Order_Date)
        ).sum()
    )
    d = attach_static(d, a).sort_values("Order_Date").reset_index(drop=True)
    d = history_features(d, d)
    d["late"] = (d.Delay_Days > 0).astype(int)
    d["stockout"] = (d.Delay_Days > d.Stock_On_Hand_Days).astype(int)
    d["excess_delay_days"] = (
        d.Delay_Days - d.Stock_On_Hand_Days
    ).clip(lower=0)

    # Four chronological partitions: 70% train, 5% threshold validation,
    # 5% quantile calibration, and newest 20% untouched final test.
    dates = d.Order_Date.sort_values()
    cut_train = dates.quantile(0.70)
    cut_threshold = dates.quantile(0.75)
    cut_cal = dates.quantile(0.80)
    tr = d[d.Order_Date <= cut_train].copy()
    va = d[(d.Order_Date > cut_train) & (d.Order_Date <= cut_threshold)].copy()
    cal = d[(d.Order_Date > cut_threshold) & (d.Order_Date <= cut_cal)].copy()
    te = d[d.Order_Date > cut_cal].copy()

    if tr.empty or va.empty or cal.empty or te.empty:
        raise ValueError("Chronological split produced an empty partition. Check Order_Date values.")
    if tr.late.nunique() < 2:
        raise ValueError("Training partition must contain both late and on-time shipments.")
    if tr.stockout.nunique() < 2:
        raise ValueError("Training partition must contain both stock-out and no-stock-out shipments.")
    if va.late.nunique() < 2:
        print("WARNING: threshold-validation partition contains only one class; using threshold 0.5.")
    if (cal.late == 1).sum() < 20:
        raise ValueError("Too few late shipments in quantile-calibration partition; need at least 20.")
    if te.late.nunique() < 2:
        print("WARNING: test partition contains only one class; ROC-AUC/AP will be unavailable.")

    # Fit category maps on training data only; unseen categories map to -1.
    cat_maps = {c: sorted(tr[c].dropna().unique().tolist()) for c in CAT_COLS}
    Xtr, Xva, Xcal, Xte = (
        encode(tr, cat_maps),
        encode(va, cat_maps),
        encode(cal, cat_maps),
        encode(te, cat_maps),
    )
    Xtr_stockout, Xva_stockout, Xcal_stockout, Xte_stockout = (
        encode_stockout(tr, cat_maps),
        encode_stockout(va, cat_maps),
        encode_stockout(cal, cat_maps),
        encode_stockout(te, cat_maps),
    )

    clf, reg, qmodels = make_models()
    classifier_best_iteration = fit_with_early_stopping(
        clf, Xtr, tr.late, Xva, va.late
    )
    classifier_early_stopping_method = (
        "lightgbm_validation_callback"
        if clf.__class__.__module__.startswith("lightgbm")
        else "sklearn_chronological_validation_log_loss"
    )
    print(
        "Late classifier selected "
        f"{classifier_best_iteration} iterations using "
        f"{classifier_early_stopping_method}."
    )
    late_tr = tr.late == 1
    if late_tr.sum() < 10:
        raise ValueError("Too few late training rows to train delay regressors reliably.")
    reg.fit(Xtr.loc[late_tr], tr.loc[late_tr, "Delay_Days"])
    for model in qmodels:
        model.fit(Xtr.loc[late_tr], tr.loc[late_tr, "Delay_Days"])

    stockout_clf, excess_delay_model = make_stockout_models()
    stockout_best_iteration = fit_with_early_stopping(
        stockout_clf,
        Xtr_stockout,
        tr.stockout,
        Xva_stockout,
        va.stockout,
    )
    stockout_early_stopping_method = (
        "lightgbm_validation_callback"
        if stockout_clf.__class__.__module__.startswith("lightgbm")
        else "sklearn_chronological_validation_log_loss"
    )
    print(
        "Stock-out classifier selected "
        f"{stockout_best_iteration} iterations using "
        f"{stockout_early_stopping_method}."
    )
    stockout_tr = tr.stockout == 1
    if stockout_tr.sum() < 10:
        raise ValueError(
            "Too few training stock-outs to train excess-delay severity model."
        )
    excess_delay_model.fit(
        Xtr_stockout.loc[stockout_tr],
        tr.loc[stockout_tr, "excess_delay_days"],
    )

    # Select classifier threshold on the first validation slice only.
    p_valid = clf.predict_proba(Xva)[:, 1]
    threshold = _choose_threshold(va.late, p_valid)
    valid_metrics, _ = _classification_metrics(va.late, p_valid, threshold)

    # Calibrate each conditional-delay quantile on a later, separate time slice.
    # Add the q-th quantile of (actual - predicted) residuals to that quantile model.
    cal_late = cal.late.to_numpy() == 1
    cal_actual = cal.Delay_Days.to_numpy(dtype=float)[cal_late]
    cal_raw = np.column_stack([m.predict(Xcal.loc[cal_late]) for m in qmodels])
    quantile_offsets = np.array([
        np.quantile(cal_actual - cal_raw[:, j], q)
        for j, q in enumerate(QUANTS)
    ], dtype=float)
    print("\n[Quantile calibration offsets learned on separate chronological period]")
    for q, offset in zip(QUANTS, quantile_offsets):
        print(f"Q{int(q*100):02d}: additive adjustment {offset:+.3f} days")

    # Calibrate the stock-out expected-loss total on a separate chronological slice.
    # The final test slice remains excluded from model fitting and calibration.
    cal_late_probability = clf.predict_proba(Xcal)[:, 1]
    cal_stockout_probability = stockout_probabilities(
        stockout_clf, Xcal_stockout, cal_late_probability
    )
    cal_excess_days = np.maximum(
        0, excess_delay_model.predict(Xcal_stockout)
    )
    cal_stockout_loss = (
        cal_stockout_probability
        * cal_excess_days
        * cal.Line_Loss_Per_Day_INR.to_numpy(dtype=float)
    )
    cal_actual_loss = cal.Production_Loss_Cost_INR.to_numpy(dtype=float)
    cal_predicted_loss_total = float(cal_stockout_loss.sum())
    if cal_predicted_loss_total <= 0:
        raise ValueError(
            "Stock-out loss calibration predictions sum to zero; cannot calibrate."
        )
    stockout_loss_calibration_factor = float(
        cal_actual_loss.sum() / cal_predicted_loss_total
    )
    print(
        "Stock-out expected-loss calibration factor from calibration period: "
        f"{stockout_loss_calibration_factor:.4f}"
    )

    # Final untouched test evaluation.
    p = clf.predict_proba(Xte)[:, 1]
    class_metrics, pred_late = _classification_metrics(te.late, p, threshold)
    days_if_late = np.maximum(1, reg.predict(Xte))
    raw_quantiles = predict_quantiles(Xte, qmodels, quantile_offsets)
    # Sorting is used for the expected-loss distribution to avoid crossing quantiles.
    draws = np.sort(raw_quantiles, axis=1)
    exp_loss_new = expected_loss(p, draws, te.Stock_On_Hand_Days, te.Line_Loss_Per_Day_INR)
    exp_loss_old = p * loss_formula(days_if_late, te.Stock_On_Hand_Days, te.Line_Loss_Per_Day_INR)
    true_loss = te.Production_Loss_Cost_INR.to_numpy(dtype=float)

    validation_stockout_probability = stockout_probabilities(
        stockout_clf,
        Xva_stockout,
        clf.predict_proba(Xva)[:, 1],
    )
    stockout_threshold = _choose_threshold(
        va.stockout, validation_stockout_probability
    )
    stockout_probability = stockout_probabilities(
        stockout_clf, Xte_stockout, p
    )
    stockout_prediction = (stockout_probability >= stockout_threshold).astype(int)
    excess_days_if_stockout = np.maximum(
        0, excess_delay_model.predict(Xte_stockout)
    )
    expected_excess_delay_days = stockout_probability * excess_days_if_stockout
    stockout_loss_uncalibrated = (
        expected_excess_delay_days * te.Line_Loss_Per_Day_INR.to_numpy(dtype=float)
    )
    stockout_loss_calibrated = (
        stockout_loss_uncalibrated * stockout_loss_calibration_factor
    )
    stockout_metrics, _ = _classification_metrics(
        te.stockout, stockout_probability, stockout_threshold
    )
    stockout_reliability = stockout_reliability_table(
        te.stockout, stockout_probability
    )
    stockout_calibration = stockout_calibration_metrics(stockout_reliability)
    actual_stockout = te.stockout.to_numpy(dtype=bool)
    actual_excess_days = te.excess_delay_days.to_numpy(dtype=float)
    if actual_stockout.any():
        excess_delay_mae = float(
            mean_absolute_error(
                actual_excess_days[actual_stockout],
                excess_days_if_stockout[actual_stockout],
            )
        )
        excess_delay_baseline = float(
            mean_absolute_error(
                actual_excess_days[actual_stockout],
                np.full(
                    actual_stockout.sum(),
                    tr.loc[stockout_tr, "excess_delay_days"].median(),
                ),
            )
        )
    else:
        excess_delay_mae = excess_delay_baseline = np.nan
    stockout_loss_mae_uncalibrated = float(
        mean_absolute_error(true_loss, stockout_loss_uncalibrated)
    )
    stockout_loss_rmse_uncalibrated = float(
        np.sqrt(mean_squared_error(true_loss, stockout_loss_uncalibrated))
    )
    stockout_loss_mae = float(mean_absolute_error(true_loss, stockout_loss_calibrated))
    stockout_loss_rmse = float(
        np.sqrt(mean_squared_error(true_loss, stockout_loss_calibrated))
    )
    stockout_loss_bias_pct = (
        (stockout_loss_calibrated.sum() / true_loss.sum() - 1) * 100
        if true_loss.sum()
        else np.nan
    )

    print("\n=== Chronological evaluation: 70% train / 5% threshold validation / 5% calibration / 20% test ===")
    print(f"Rows: train {len(tr):,} | threshold validation {len(va):,} | calibration {len(cal):,} | final test {len(te):,}")
    print(
        f"History rows excluded for missing/impossible delivery chronology: "
        f"{invalid_history_rows:,}"
    )
    print(f"Late rates: train {tr.late.mean():.2%} | threshold validation {va.late.mean():.2%} | calibration {cal.late.mean():.2%} | test {te.late.mean():.2%}")
    print("\n[Classification: final test; threshold selected on validation]")
    print(f"Validation-selected threshold: {threshold:.4f}")
    print(f"ROC-AUC: {class_metrics['roc_auc']:.3f} | Average precision: {class_metrics['average_precision']:.3f}")
    print(f"Accuracy: {class_metrics['accuracy']:.3f} | Precision: {class_metrics['precision']:.3f}")
    print(f"Recall: {class_metrics['recall']:.3f} | F1: {class_metrics['f1']:.3f}")
    print(f"Specificity: {class_metrics['specificity']:.3f} | Balanced accuracy: {class_metrics['balanced_accuracy']:.3f}")
    print(f"Brier: {class_metrics['brier_score']:.4f} | Log loss: {class_metrics['log_loss']:.4f}")
    print("Confusion matrix:")
    print("                 Predicted on-time  Predicted late")
    print(f"Actual on-time       {class_metrics['true_negative']:10,d}       {class_metrics['false_positive']:10,d}")
    print(f"Actual late          {class_metrics['false_negative']:10,d}       {class_metrics['true_positive']:10,d}")

    # Also report threshold-independent risk ranking at a fixed top-k capacity.
    if te.late.sum() > 0:
        top_k_n = max(1, int(np.ceil(len(te) * te.late.mean())))
        top_idx = np.argsort(p)[-top_k_n:]
        top_k_recall = float(te.late.to_numpy()[top_idx].sum() / te.late.sum())
        top_k_precision = float(te.late.to_numpy()[top_idx].mean())
        print(f"Top {te.late.mean():.0%} risk-list recall: {top_k_recall:.2%} | precision: {top_k_precision:.2%}")
    else:
        top_k_recall = top_k_precision = np.nan

    # Delay regression: evaluate only on shipments that actually arrived late.
    late_test = te.late.to_numpy() == 1
    actual_late_days = te.Delay_Days.to_numpy(dtype=float)[late_test]
    pred_late_days = days_if_late[late_test]
    median_baseline = float(tr.loc[tr.late == 1, "Delay_Days"].median())
    if late_test.any():
        delay_mae = float(mean_absolute_error(actual_late_days, pred_late_days))
        delay_rmse = float(np.sqrt(mean_squared_error(actual_late_days, pred_late_days)))
        delay_r2 = float(r2_score(actual_late_days, pred_late_days)) if len(actual_late_days) > 1 else np.nan
        median_mae = float(mean_absolute_error(actual_late_days, np.full(len(actual_late_days), median_baseline)))
    else:
        delay_mae = delay_rmse = delay_r2 = median_mae = np.nan
    print("\n[Delay regression: actual late shipments only]")
    print(f"MAE: {delay_mae:.2f} days | RMSE: {delay_rmse:.2f} days | R2: {delay_r2:.3f}")
    print(f"Training-median baseline MAE: {median_mae:.2f} days")

    # Quantile coverage and pinball loss, calculated on actual late test shipments.
    quantile_rows = []
    print("\n[Quantile calibration: actual late shipments only]")
    for j, q in enumerate(QUANTS):
        if late_test.any():
            q_pred = raw_quantiles[late_test, j]
            coverage = float(np.mean(actual_late_days <= q_pred))
            errors = actual_late_days - q_pred
            pinball = float(np.mean(np.maximum(q * errors, (q - 1) * errors)))
        else:
            coverage = pinball = np.nan
        quantile_rows.append({"quantile": q, "target_coverage": q,
                              "actual_coverage": coverage, "pinball_loss_days": pinball})
        print(f"Q{int(q*100):02d}: target {q:.0%} | observed {coverage:.2%} | pinball {pinball:.3f}")

    # Financial loss: compare to recorded actual Production_Loss_Cost_INR.
    loss_mae_new = float(mean_absolute_error(true_loss, exp_loss_new))
    loss_rmse_new = float(np.sqrt(mean_squared_error(true_loss, exp_loss_new)))
    loss_mae_old = float(mean_absolute_error(true_loss, exp_loss_old))
    loss_rmse_old = float(np.sqrt(mean_squared_error(true_loss, exp_loss_old)))
    loss_baseline = float(mean_absolute_error(true_loss, np.full(len(te), true_loss.mean())))
    actual_total = float(true_loss.sum())
    predicted_total_new = float(exp_loss_new.sum())
    predicted_total_old = float(exp_loss_old.sum())
    total_bias_pct = ((predicted_total_new - actual_total) / actual_total * 100
                      if actual_total else np.nan)
    print("\n[Financial loss]")
    print(f"New model MAE: INR {loss_mae_new:,.0f} | RMSE: INR {loss_rmse_new:,.0f}")
    print(f"Old model MAE: INR {loss_mae_old:,.0f} | RMSE: INR {loss_rmse_old:,.0f}")
    print(f"Mean baseline MAE: INR {loss_baseline:,.0f}")
    print(f"Actual total: INR {actual_total:,.0f}")
    print(f"New predicted total: INR {predicted_total_new:,.0f} | total bias: {total_bias_pct:+.2f}%")
    print(f"Old predicted total: INR {predicted_total_old:,.0f}")
    print("\n[Experimental stock-out model]")
    print(
        f"Stock-out ROC-AUC: {stockout_metrics['roc_auc']:.3f} | "
        f"average precision: {stockout_metrics['average_precision']:.3f} | "
        f"Brier: {stockout_metrics['brier_score']:.4f}"
    )
    print(
        f"Stock-out reliability: ECE {stockout_calibration['expected_calibration_error']:.4f} | "
        f"maximum bin gap {stockout_calibration['max_calibration_error']:.4f}"
    )
    print(
        f"Excess-delay MAE on actual stock-outs: {excess_delay_mae:.2f} days | "
        f"training-median baseline: {excess_delay_baseline:.2f} days"
    )
    print(
        f"Calibrated stock-out loss MAE: INR {stockout_loss_mae:,.0f} | "
        f"RMSE: INR {stockout_loss_rmse:,.0f} | total bias: {stockout_loss_bias_pct:+.2f}%"
    )
    print(
        f"Uncalibrated stock-out loss MAE: INR {stockout_loss_mae_uncalibrated:,.0f} | "
        f"RMSE: INR {stockout_loss_rmse_uncalibrated:,.0f}"
    )

    # Supplier-level classification summary to expose uneven performance.
    supplier_rows = []
    test_supplier = te[["Supplier_ID", "late"]].copy()
    test_supplier["prob_late"] = p
    test_supplier["predicted_late"] = pred_late
    for supplier_id, group in test_supplier.groupby("Supplier_ID"):
        if group.late.nunique() < 2:
            supplier_auc = np.nan
        else:
            supplier_auc = float(roc_auc_score(group.late, group.prob_late))
        supplier_rows.append({
            "Supplier_ID": supplier_id, "shipments": int(len(group)),
            "actual_late_rate": float(group.late.mean()),
            "predicted_late_rate": float(group.predicted_late.mean()),
            "precision": float(precision_score(group.late, group.predicted_late, zero_division=0)),
            "recall": float(recall_score(group.late, group.predicted_late, zero_division=0)),
            "roc_auc": supplier_auc
        })

    # Save detailed outputs.
    os.makedirs("evaluation", exist_ok=True)
    prediction_report = te[["Shipment_ID", "Supplier_ID", "Item_ID", "Order_Date",
                            "late", "Delay_Days", "Production_Loss_Cost_INR"]].copy()
    prediction_report["prob_late"] = p
    prediction_report["predicted_late"] = pred_late
    prediction_report["predicted_days_if_late"] = days_if_late
    for j, q in enumerate(QUANTS):
        prediction_report[f"predicted_q{int(q*100)}_days"] = raw_quantiles[:, j]
    prediction_report["predicted_loss_new_inr"] = exp_loss_new
    prediction_report["predicted_loss_old_inr"] = exp_loss_old
    prediction_report["actual_stockout"] = te.stockout.to_numpy()
    prediction_report["stockout_probability"] = stockout_probability
    prediction_report["predicted_stockout"] = stockout_prediction
    prediction_report["predicted_excess_days_if_stockout"] = excess_days_if_stockout
    prediction_report["predicted_expected_excess_delay_days"] = expected_excess_delay_days
    prediction_report["predicted_stockout_loss_uncalibrated_inr"] = stockout_loss_uncalibrated
    prediction_report["predicted_stockout_loss_calibrated_inr"] = stockout_loss_calibrated
    prediction_report.to_csv("evaluation/test_predictions.csv", index=False)
    stockout_reliability.to_csv(
        "evaluation/stockout_reliability.csv", index=False
    )
    pd.DataFrame(quantile_rows).to_csv("evaluation/quantile_coverage.csv", index=False)
    pd.DataFrame(supplier_rows).to_csv("evaluation/supplier_metrics.csv", index=False)

    metrics = {
        "train_rows": len(tr), "validation_rows": len(va), "test_rows": len(te),
        "train_late_rate": float(tr.late.mean()), "validation_late_rate": float(va.late.mean()),
        "test_late_rate": float(te.late.mean()), "validation_selected_threshold": threshold,
        "classifier_best_iteration": classifier_best_iteration,
        "classifier_early_stopping_method": classifier_early_stopping_method,
        "stockout_best_iteration": stockout_best_iteration,
        "stockout_early_stopping_method": stockout_early_stopping_method,
        "invalid_history_rows_excluded": invalid_history_rows,
        "stockout_validation_selected_threshold": stockout_threshold,
        "stockout_loss_calibration_factor": stockout_loss_calibration_factor,
        "stockout_expected_calibration_error": stockout_calibration[
            "expected_calibration_error"
        ],
        "stockout_max_calibration_error": stockout_calibration[
            "max_calibration_error"
        ],
        **class_metrics,
        **{f"stockout_{key}": value for key, value in stockout_metrics.items()},
        "top_base_rate_fraction_recall": top_k_recall,
        "top_base_rate_fraction_precision": top_k_precision,
        "delay_mae_days_late_only": delay_mae, "delay_rmse_days_late_only": delay_rmse,
        "delay_r2_late_only": delay_r2, "training_median_baseline_mae_days": median_mae,
        "stockout_excess_delay_mae_days": excess_delay_mae,
        "stockout_excess_delay_training_median_baseline_mae_days": excess_delay_baseline,
        "loss_mae_new_inr": loss_mae_new, "loss_rmse_new_inr": loss_rmse_new,
        "loss_mae_old_inr": loss_mae_old, "loss_rmse_old_inr": loss_rmse_old,
        "stockout_loss_mae_uncalibrated_inr": stockout_loss_mae_uncalibrated,
        "stockout_loss_rmse_uncalibrated_inr": stockout_loss_rmse_uncalibrated,
        "stockout_loss_mae_inr": stockout_loss_mae,
        "stockout_loss_rmse_inr": stockout_loss_rmse,
        "stockout_total_loss_bias_pct": stockout_loss_bias_pct,
        "loss_mean_baseline_mae_inr": loss_baseline,
        "actual_total_loss_inr": actual_total,
        "predicted_total_loss_new_inr": predicted_total_new,
        "predicted_total_loss_old_inr": predicted_total_old,
        "new_total_loss_bias_pct": total_bias_pct
    }
    pd.DataFrame([metrics]).to_csv("evaluation/metrics_summary.csv", index=False)

    # Feature importance for LightGBM / sklearn tree estimators when exposed.
    try:
        importance = getattr(clf, "feature_importances_", None)
        if importance is not None:
            pd.DataFrame({"feature": FEATURES, "importance": importance}).sort_values(
                "importance", ascending=False
            ).to_csv("evaluation/feature_importance.csv", index=False)
    except Exception as exc:
        print(f"Could not save feature importance: {exc}")

    print("\nSaved evaluation files:")
    for name in ["metrics_summary.csv", "quantile_coverage.csv", "test_predictions.csv",
                 "stockout_reliability.csv", "supplier_metrics.csv", "feature_importance.csv"]:
        if os.path.exists(os.path.join("evaluation", name)):
            print(f"  evaluation/{name}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "clf": clf, "reg": reg, "qmodels": qmodels, "quants": QUANTS,
        "stockout_clf": stockout_clf,
        "excess_delay_model": excess_delay_model,
        "stockout_features": STOCKOUT_FEATURES,
        "stockout_threshold": stockout_threshold,
        "stockout_best_iteration": stockout_best_iteration,
        "stockout_early_stopping_method": stockout_early_stopping_method,
        "stockout_loss_calibration_factor": stockout_loss_calibration_factor,
        "quantile_offsets": quantile_offsets,
        "cat_maps": cat_maps, "classification_threshold": threshold,
        "classifier_best_iteration": classifier_best_iteration,
        "classifier_early_stopping_method": classifier_early_stopping_method,
        "mean_late_days": float(tr.loc[tr.late == 1, "Delay_Days"].mean()),
        "features": FEATURES, "cat_cols": CAT_COLS
    }, f"{MODEL_DIR}/delay_models.joblib")
    print(f"\nSaved trained models -> {MODEL_DIR}/delay_models.joblib")


# ------------------------------------------------------------------ 4. predict delay + loss
def predict_orders(orders, hist, a, bundle):
    """orders: new orders with Shipment_ID, Supplier_ID, Item_ID, Order_Date, Promised_Date, Quantity_Ordered,
    Unit_Cost_INR, Stock_On_Hand_Days, Line_Loss_Per_Day_INR.  hist: past delivered shipments."""
    o = attach_static(orders.copy(), a)
    o = history_features(o, hist)
    return _predict_prepared_orders(o, bundle)


def _predict_prepared_orders(o, bundle):
    if "qmodels" not in bundle:
        raise RuntimeError("Saved models are from the old version. Re-run `train` to create models with 'qmodels'.")
    required_stockout_keys = {
        "stockout_clf",
        "excess_delay_model",
        "stockout_loss_calibration_factor",
    }
    if not required_stockout_keys.issubset(bundle):
        raise RuntimeError(
            "Saved model bundle does not include stock-out models. "
            "Retrain with delay_pipeline_quantile_calibrated.py."
        )
    if list(bundle.get("stockout_features", [])) != STOCKOUT_FEATURES:
        raise RuntimeError(
            "Saved stock-out model features do not match this pipeline. "
            "Retrain with delay_pipeline_quantile_calibrated.py."
        )
    X = encode(o, bundle["cat_maps"])
    p = bundle["clf"].predict_proba(X)[:, 1]
    days = np.maximum(1, bundle["reg"].predict(X))
    draws = delay_draws(X, bundle["qmodels"], bundle.get("quantile_offsets"))
    lo, hi = draws[:, 0], draws[:, -1]                    # 10th and 90th percentile of days late
    o["prob_late"] = p
    o["exp_delay_days"] = p * days
    o["days_if_late"] = days
    o["days_if_late_range"] = [f"{l:.0f}-{h:.0f}" for l, h in zip(lo, hi)]
    o["exp_loss_inr"] = expected_loss(p, draws, o.Stock_On_Hand_Days, o.Line_Loss_Per_Day_INR)
    X_stockout = encode_stockout(o, bundle["cat_maps"])
    stockout_probability = stockout_probabilities(
        bundle["stockout_clf"], X_stockout, p
    )
    excess_days_if_stockout = np.maximum(
        0, bundle["excess_delay_model"].predict(X_stockout)
    )
    o["stockout_probability"] = stockout_probability
    o["excess_days_if_stockout"] = excess_days_if_stockout
    o["expected_excess_delay_days"] = (
        stockout_probability * excess_days_if_stockout
    )
    o["stockout_expected_loss_inr"] = (
        o["expected_excess_delay_days"]
        * o.Line_Loss_Per_Day_INR
        * bundle["stockout_loss_calibration_factor"]
    )
    # "if late" loss at the 90th-percentile delay (NOT multiplied by P(late))
    o["worst_case_loss_inr"] = loss_formula(hi, o.Stock_On_Hand_Days, o.Line_Loss_Per_Day_INR)
    return o


# ------------------------------------------------------------------ 5. alternatives
def recommend(order, a, hist, bundle, as_of, top_n=3, switch_threshold=0.02, min_quality=4.0):
    """Rank other suppliers for the same item. Every alternative is scored with the SAME trained delay model
    (its own history, location, quality, lead time) under this order's conditions, so the comparison is like for like.
    Total expected cost = purchase + shipping + expected loss."""
    qty, stock, loss_day = order.Quantity_Ordered, order.Stock_On_Hand_Days, order.Line_Loss_Per_Day_INR
    days_left = (order.Promised_Date - pd.Timestamp(as_of)).days

    result_columns = [
        "Supplier_ID", "Supplier_Location", "Unit_Cost_INR",
        "Shipping_Cost_INR", "Lead_Time_Days", "Quality_Rating",
        "prob_late", "stockout_probability", "exp_loss_inr",
        "stockout_expected_loss_inr", "total_cost_inr", "saving_inr",
    ]

    def unavailable(status):
        empty = pd.DataFrame(columns=result_columns)
        empty.attrs["status"] = status
        return np.nan, empty

    # A missing baseline shipping quote makes savings impossible to calculate.
    cur = a[(a.Supplier_ID == order.Supplier_ID) & (a.Item_ID == order.Item_ID)]
    if cur.empty:
        return unavailable("missing_current_supplier_quote")
    if "Shipping_Cost_INR" not in cur:
        return unavailable("missing_current_supplier_shipping_cost")
    current_shipping = pd.to_numeric(cur.Shipping_Cost_INR, errors="coerce")
    if (
        not np.isfinite(current_shipping.to_numpy(dtype=float)).all()
        or (current_shipping < 0).any()
    ):
        return unavailable("missing_current_supplier_shipping_cost")
    distinct_shipping = current_shipping.unique()
    if len(distinct_shipping) != 1:
        return unavailable("ambiguous_current_supplier_shipping_cost")
    cur_ship = float(distinct_shipping[0])
    current_cost = qty * order.Unit_Cost_INR + cur_ship + order.exp_loss_inr

    c = a[(a.Item_ID == order.Item_ID) & (a.Supplier_ID != order.Supplier_ID) & (a.Supplier_Status == "Active")
          & (a.Available_Quantity >= qty) & (a.Minimum_Order_Quantity <= qty) & (a.Maximum_Supply_Quantity >= qty)
          & (a.Quality_Rating >= min_quality)].copy()
    c = c[c.Lead_Time_Days <= days_left + stock]            # must still arrive in time (stock covers the gap)
    candidate_shipping = pd.to_numeric(c.Shipping_Cost_INR, errors="coerce")
    cost_available = (
        np.isfinite(candidate_shipping.to_numpy(dtype=float))
        & (candidate_shipping.to_numpy(dtype=float) >= 0)
    )
    excluded_missing_shipping = int((~cost_available).sum())
    c = c.loc[cost_available].copy()
    c["Shipping_Cost_INR"] = candidate_shipping.loc[c.index]
    if c.empty:
        empty = pd.DataFrame(columns=result_columns)
        empty.attrs["status"] = (
            "alternatives_missing_shipping_cost"
            if excluded_missing_shipping
            else "no_eligible_alternatives"
        )
        return current_cost, empty

    # build one hypothetical order per candidate: same dates, quantity, stock and line, different supplier + price
    cand = pd.DataFrame({"Shipment_ID": order.Shipment_ID, "Supplier_ID": c.Supplier_ID.values, "Item_ID": order.Item_ID,
                         "Order_Date": pd.Timestamp(as_of), "Promised_Date": order.Promised_Date, "Quantity_Ordered": qty,
                         "Unit_Cost_INR": c.Unit_Cost_INR.values, "Stock_On_Hand_Days": stock,
                         "Line_Loss_Per_Day_INR": loss_day})
    scored = predict_orders(cand, hist, a, bundle)
    c["prob_late"] = scored.prob_late.values
    c["stockout_probability"] = scored.stockout_probability.values
    c["exp_loss_inr"] = scored.exp_loss_inr.values
    c["stockout_expected_loss_inr"] = scored.stockout_expected_loss_inr.values
    c["total_cost_inr"] = qty * c.Unit_Cost_INR + c.Shipping_Cost_INR + c.exp_loss_inr
    c["saving_inr"] = current_cost - c.total_cost_inr
    c = c.sort_values("total_cost_inr")
    c = c[c.saving_inr > switch_threshold * qty * order.Unit_Cost_INR]   # switch only if clearly cheaper
    recommendations = c[result_columns].head(top_n)
    recommendations.attrs["status"] = (
        "alternatives_missing_shipping_cost"
        if excluded_missing_shipping
        else "available"
    )
    return current_cost, recommendations


def mitigation_options(order):
    value = order.Quantity_Ordered * order.Unit_Cost_INR
    rows = [{"option": k, "cost_inr": v["cost_pct"] * value, "loss_avoided_inr": v["recovery"] * order.exp_loss_inr,
             "net_benefit_inr": v["recovery"] * order.exp_loss_inr - v["cost_pct"] * value}
            for k, v in MITIGATION.items()]
    return pd.DataFrame(rows).sort_values("net_benefit_inr", ascending=False)


# ------------------------------------------------------------------ demo
def demo(delivery_path, alt_path, n=300, show=3):
    d, a = load_data(delivery_path, alt_path)
    bundle = joblib.load(f"{MODEL_DIR}/delay_models.joblib")
    d = d.sort_values("Order_Date").reset_index(drop=True)
    new, hist = d.tail(n).copy(), d.iloc[:-n]
    actual = new.set_index("Shipment_ID")[["Delay_Days", "Production_Loss_Cost_INR"]]
    scored = sequential_historical_replay(new, hist, a, bundle)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"Scored the last {n} orders as if they were new; showing the {show} with the highest expected loss.")
    for _, o in scored.sort_values("exp_loss_inr", ascending=False).head(show).iterrows():
        print("-" * 78)
        print(f"{o.Shipment_ID}  {o.Supplier_ID} / {o.Item_ID}  qty {o.Quantity_Ordered}")
        print(f"  P(late) {o.prob_late:.0%} | expected delay {o.exp_delay_days:.1f} d | if late {o.days_if_late_range} d "
              f"| expected loss INR {o.exp_loss_inr:,.0f} | if-late 1-in-10 worst case INR {o.worst_case_loss_inr:,.0f}")
        print(f"  (actual: delay {actual.loc[o.Shipment_ID, 'Delay_Days']} d, loss INR {actual.loc[o.Shipment_ID, 'Production_Loss_Cost_INR']:,})")
        if o.prob_late >= bundle.get("classification_threshold", 0.5):
            cur, alts = recommend(o, a, d, bundle, as_of=o.Order_Date)
            print(f"  High risk. Current supplier total expected cost INR {cur:,.0f}")
            print(alts.round(2).to_string(index=False) if len(alts) else "  No alternative is clearly better.")
            print(mitigation_options(o).round(0).head(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "demo"])
    ap.add_argument("--delivery", required=True)
    ap.add_argument("--alt", required=True)
    args = ap.parse_args()
    train(args.delivery, args.alt) if args.cmd == "train" else demo(args.delivery, args.alt)