"""
Automobile supplier delay pipeline
  1. load + join       2. leak-free history features
  3. train (LightGBM; falls back to scikit-learn if LightGBM is not installed)
  4. predict delay + loss     5. rank alternative suppliers / mitigation options

Usage
  python delay_pipeline.py train  --delivery DELIVERY.csv --alt ALT.csv
  python delay_pipeline.py demo   --delivery DELIVERY.csv --alt ALT.csv     # scores the last 300 orders, shows the 3 riskiest

CHANGE vs previous version
  Expected loss used ONE typical delay (L1 regressor) inside max(0, delay - stock) * loss/day.
  Loss is convex (zero until stock runs out), so that systematically under-states loss (Jensen's inequality).
  Now expected loss = P(late) * average of the loss over several quantiles of the delay distribution.
  Models saved with the old version are incompatible -> retrain with `train`.
"""
import argparse, json, os
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

MODEL_DIR = "models"
CAT_COLS = ["Supplier_Location", "Part_Type"]
FEATURES = ["promised_lt", "buffer", "Quantity_Ordered", "Unit_Cost_INR", "order_month",
            "Lead_Time_Days", "Quality_Rating", "Supplier_Location", "Part_Type",
            "h_n", "h_late", "h_delay", "h_recent"]

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
    prior_late = (hist.Delay_Days > 0).astype(float)
    h = hist.assign(_late=prior_late).sort_values("Actual_Delivery_Date")
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


# ------------------------------------------------------------------ 3. models
def make_models():
    """Returns: classifier (late or not), regressor (typical days late), list of quantile regressors (one per QUANTS)."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        kw = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
        clf = LGBMClassifier(**kw)
        reg = LGBMRegressor(objective="regression_l1", **kw)          # typical days late (median-like)
        qmodels = [LGBMRegressor(objective="quantile", alpha=q, **kw) for q in QUANTS]
        print("using LightGBM")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier as C, HistGradientBoostingRegressor as R
        kw = dict(max_iter=300, learning_rate=0.05, random_state=42)
        clf = C(**kw); reg = R(loss="absolute_error", **kw)
        qmodels = [R(loss="quantile", quantile=q, **kw) for q in QUANTS]
        print("LightGBM not installed -> using scikit-learn HistGradientBoosting (same idea)")
    return clf, reg, qmodels


def loss_formula(delay_days, stock_days, loss_per_day):
    return np.maximum(0, np.asarray(delay_days) - np.asarray(stock_days)) * np.asarray(loss_per_day)


def delay_draws(X, qmodels):
    """Days-late predictions at each quantile, shape (n_orders, len(QUANTS)). Sorted per row so quantiles never cross."""
    d = np.column_stack([np.maximum(1, m.predict(X)) for m in qmodels])
    return np.sort(d, axis=1)


def expected_loss(p, draws, stock, loss_day):
    """E[loss] = P(late) * average over the delay distribution of max(0, delay - stock) * loss_per_day.
    Averaging the loss (not the delay) keeps the 'stock runs out' tail that a single typical delay would hide."""
    stock = np.asarray(stock, dtype=float)[:, None]
    loss_day = np.asarray(loss_day, dtype=float)[:, None]
    per_draw = np.maximum(0, draws - stock) * loss_day
    return np.asarray(p) * per_draw.mean(axis=1)


def train(delivery_path, alt_path):
    d, a = load_data(delivery_path, alt_path)
    d = attach_static(d, a).sort_values("Order_Date").reset_index(drop=True)
    d = history_features(d, d)
    d["late"] = (d.Delay_Days > 0).astype(int)
    cat_maps = {c: sorted(d[c].dropna().unique().tolist()) for c in CAT_COLS}

    cut = d.Order_Date.quantile(0.8)                       # time split, never random
    tr, te = d[d.Order_Date <= cut], d[d.Order_Date > cut]
    Xtr, Xte = encode(tr, cat_maps), encode(te, cat_maps)
    clf, reg, qmodels = make_models()
    clf.fit(Xtr, tr.late)
    late_tr = tr.late == 1
    for m in [reg] + qmodels:
        m.fit(Xtr[late_tr], tr.Delay_Days[late_tr])

    p = clf.predict_proba(Xte)[:, 1]
    days_if_late = np.maximum(1, reg.predict(Xte))
    draws = delay_draws(Xte, qmodels)
    exp_delay = p * days_if_late
    exp_loss_new = expected_loss(p, draws, te.Stock_On_Hand_Days, te.Line_Loss_Per_Day_INR)
    exp_loss_old = p * loss_formula(days_if_late, te.Stock_On_Hand_Days, te.Line_Loss_Per_Day_INR)   # old method, for comparison
    true_loss = te.Production_Loss_Cost_INR.values
    thr = np.quantile(p, 1 - te.late.mean())               # flag the top-N% riskiest = base rate
    flagged = p >= thr
    print("\n=== Time-split test (newest 20% of orders) ===")
    print(f"Late vs on-time  AUC {roc_auc_score(te.late, p):.3f} | avg precision {average_precision_score(te.late, p):.3f} "
          f"(base rate {te.late.mean():.3f})")
    print(f"Recall when flagging top {te.late.mean():.0%} riskiest: {te.late[flagged].sum() / te.late.sum():.2%}")
    lt = te.late == 1
    print(f"Days late (late shipments only): MAE {mean_absolute_error(te.Delay_Days[lt], days_if_late[lt]):.2f} days "
          f"vs {mean_absolute_error(te.Delay_Days[lt], [tr.Delay_Days[late_tr].median()] * lt.sum()):.2f} for guessing the median")
    base = mean_absolute_error(true_loss, [true_loss.mean()] * len(te))
    print(f"Loss INR: MAE {mean_absolute_error(true_loss, exp_loss_new):,.0f} (new, distribution-based) | "
          f"{mean_absolute_error(true_loss, exp_loss_old):,.0f} (old, single typical delay) | {base:,.0f} for guessing the average")
    print(f"Total loss INR: actual {true_loss.sum():,.0f} | predicted new {exp_loss_new.sum():,.0f} | predicted old {exp_loss_old.sum():,.0f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"clf": clf, "reg": reg, "qmodels": qmodels, "quants": QUANTS, "cat_maps": cat_maps,
                 "mean_late_days": float(tr.Delay_Days[late_tr].mean())}, f"{MODEL_DIR}/delay_models.joblib")
    print(f"\nsaved -> {MODEL_DIR}/delay_models.joblib")


# ------------------------------------------------------------------ 4. predict delay + loss
def predict_orders(orders, hist, a, bundle):
    """orders: new orders with Shipment_ID, Supplier_ID, Item_ID, Order_Date, Promised_Date, Quantity_Ordered,
    Unit_Cost_INR, Stock_On_Hand_Days, Line_Loss_Per_Day_INR.  hist: past delivered shipments."""
    if "qmodels" not in bundle:
        raise RuntimeError("Saved models are from the old version. Re-run `train` to create models with 'qmodels'.")
    o = attach_static(orders.copy(), a)
    o = history_features(o, hist)
    X = encode(o, bundle["cat_maps"])
    p = bundle["clf"].predict_proba(X)[:, 1]
    days = np.maximum(1, bundle["reg"].predict(X))
    draws = delay_draws(X, bundle["qmodels"])
    lo, hi = draws[:, 0], draws[:, -1]                    # 10th and 90th percentile of days late
    o["prob_late"] = p
    o["exp_delay_days"] = p * days
    o["days_if_late"] = days
    o["days_if_late_range"] = [f"{l:.0f}-{h:.0f}" for l, h in zip(lo, hi)]
    o["exp_loss_inr"] = expected_loss(p, draws, o.Stock_On_Hand_Days, o.Line_Loss_Per_Day_INR)
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

    # current supplier: include its shipping cost too, if the supplier file lists this supplier-item pair
    cur = a[(a.Supplier_ID == order.Supplier_ID) & (a.Item_ID == order.Item_ID)]
    cur_ship = float(cur.Shipping_Cost_INR.iloc[0]) if len(cur) else 0.0
    current_cost = qty * order.Unit_Cost_INR + cur_ship + order.exp_loss_inr

    c = a[(a.Item_ID == order.Item_ID) & (a.Supplier_ID != order.Supplier_ID) & (a.Supplier_Status == "Active")
          & (a.Available_Quantity >= qty) & (a.Minimum_Order_Quantity <= qty) & (a.Maximum_Supply_Quantity >= qty)
          & (a.Quality_Rating >= min_quality)].copy()
    c = c[c.Lead_Time_Days <= days_left + stock]            # must still arrive in time (stock covers the gap)
    if c.empty:
        return current_cost, pd.DataFrame()

    # build one hypothetical order per candidate: same dates, quantity, stock and line, different supplier + price
    cand = pd.DataFrame({"Shipment_ID": order.Shipment_ID, "Supplier_ID": c.Supplier_ID.values, "Item_ID": order.Item_ID,
                         "Order_Date": pd.Timestamp(as_of), "Promised_Date": order.Promised_Date, "Quantity_Ordered": qty,
                         "Unit_Cost_INR": c.Unit_Cost_INR.values, "Stock_On_Hand_Days": stock,
                         "Line_Loss_Per_Day_INR": loss_day})
    scored = predict_orders(cand, hist, a, bundle)
    c["prob_late"] = scored.prob_late.values
    c["exp_loss_inr"] = scored.exp_loss_inr.values
    c["total_cost_inr"] = qty * c.Unit_Cost_INR + c.Shipping_Cost_INR + c.exp_loss_inr
    c["saving_inr"] = current_cost - c.total_cost_inr
    c = c.sort_values("total_cost_inr")
    c = c[c.saving_inr > switch_threshold * qty * order.Unit_Cost_INR]   # switch only if clearly cheaper
    cols = ["Supplier_ID", "Supplier_Location", "Unit_Cost_INR", "Shipping_Cost_INR", "Lead_Time_Days",
            "Quality_Rating", "prob_late", "exp_loss_inr", "total_cost_inr", "saving_inr"]
    return current_cost, c[cols].head(top_n)


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
    new = new.drop(columns=["Actual_Delivery_Date", "Delay_Days", "Production_Loss_Cost_INR"])
    scored = predict_orders(new, hist, a, bundle)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"Scored the last {n} orders as if they were new; showing the {show} with the highest expected loss.")
    for _, o in scored.sort_values("exp_loss_inr", ascending=False).head(show).iterrows():
        print("-" * 78)
        print(f"{o.Shipment_ID}  {o.Supplier_ID} / {o.Item_ID}  qty {o.Quantity_Ordered}")
        print(f"  P(late) {o.prob_late:.0%} | expected delay {o.exp_delay_days:.1f} d | if late {o.days_if_late_range} d "
              f"| expected loss INR {o.exp_loss_inr:,.0f} | if-late 1-in-10 worst case INR {o.worst_case_loss_inr:,.0f}")
        print(f"  (actual: delay {actual.loc[o.Shipment_ID, 'Delay_Days']} d, loss INR {actual.loc[o.Shipment_ID, 'Production_Loss_Cost_INR']:,})")
        if o.prob_late >= 0.30:
            cur, alts = recommend(o, a, hist, bundle, as_of=o.Order_Date)
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