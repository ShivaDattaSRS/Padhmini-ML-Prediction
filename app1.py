"""
Streamlit UI for the supplier delay / production loss / alternative supplier model.
Run from the same folder as delay_pipeline.py, the two CSV files and models/:
    streamlit run app.py
"""
import os
import joblib
import pandas as pd
import streamlit as st

import delay_pipeline as dp

HIGH, MEDIUM = 0.40, 0.20          # risk bands on the chance of being late (base late rate is about 16%)
MODEL_FILE = f"{dp.MODEL_DIR}/delay_models.joblib"

st.set_page_config(page_title="Supplier delay and loss predictor", layout="wide")
st.title("Supplier delay, loss and alternative supplier predictor")
st.caption("Demo running on simulated data. Replace the CSV files with the client's real shipments before presenting results.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    delivery_path = st.text_input("Delivery CSV", "automotive_supplier_delivery_data_realistic_100k.csv")
    alt_path = st.text_input("Alternative supplier CSV", "automotive_alternative_supplier_data_realistic_100k.csv")
    n_recent = st.slider("Recent orders to score", 100, 1000, 300, step=50)
    min_quality = st.slider("Minimum supplier quality rating", 3.0, 5.0, 4.0, step=0.1)
    switch_pct = st.slider("Switch only if saving is above (% of order value)", 0.0, 10.0, 2.0, step=0.5)
    always_alts = st.checkbox("Always list alternatives (even if low risk or not cheaper)", value=True)
    st.markdown("---")
    st.caption("Chance of being late: High 40% or more, Medium 20% to 40%, Low below 20%.")

for p in (delivery_path, alt_path, MODEL_FILE):
    if not os.path.exists(p):
        st.error(f"File not found: {p}. Put it in this folder"
                 + (" or run: python delay_pipeline.py train --delivery ... --alt ..." if p == MODEL_FILE else "."))
        st.stop()


# ------------------------------------------------------------------ data + models
@st.cache_resource(show_spinner="Loading data and model...")
def load_all(delivery, alt, model_mtime):
    d, a = dp.load_data(delivery, alt)
    d = d.sort_values("Order_Date").reset_index(drop=True)
    return d, a, joblib.load(MODEL_FILE)


@st.cache_data(show_spinner="Scoring recent orders...")
def score_recent(_d, _a, _bundle, n, delivery, alt, model_mtime):
    new = _d.tail(n).drop(columns=["Actual_Delivery_Date", "Delay_Days", "Production_Loss_Cost_INR"])
    scored = dp.predict_orders(new, _d.iloc[:-n], _a, _bundle)       # history = only orders before the window
    actual = _d.tail(n)[["Shipment_ID", "Delay_Days", "Production_Loss_Cost_INR"]]
    return scored.merge(actual, on="Shipment_ID", how="left")


d, a, bundle = load_all(delivery_path, alt_path, os.path.getmtime(MODEL_FILE))


# ------------------------------------------------------------------ helpers
inr = lambda x: f"INR {x:,.0f}"


def band(p):
    return "High" if p >= HIGH else "Medium" if p >= MEDIUM else "Low"


def show_table(df):
    st.dataframe(df, hide_index=True)


def render_order(o, hist):
    """Show prediction, loss, alternatives and mitigation options for one scored order."""
    b = band(o["prob_late"])
    msg = f"{b} risk: {o['prob_late']:.0%} chance this shipment is late."
    (st.error if b == "High" else st.warning if b == "Medium" else st.success)(msg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chance of being late", f"{o['prob_late']:.0%}")
    c2.metric("Expected delay", f"{o['exp_delay_days']:.1f} days", help=f"If it is late, likely {o['days_if_late_range']} days")
    c3.metric("Expected loss", inr(o["exp_loss_inr"]))
    c4.metric("Worst-case loss", inr(o["worst_case_loss_inr"]), help="Loss if the delay hits the high end of the range")
    st.caption(f"If late, the delay is likely {o['days_if_late_range']} days. Stock on hand covers "
               f"{int(o['Stock_On_Hand_Days'])} days; line loss is {inr(o['Line_Loss_Per_Day_INR'])} per day.")

    if b == "Low" and not always_alts:
        st.info("No action needed for this order.")
        return
    if b == "Low":
        st.info("Low risk. Alternatives are shown for reference only.")

    st.subheader("Alternative suppliers")
    cur, alts = dp.recommend(o, a, hist, bundle, as_of=o["Order_Date"], min_quality=min_quality,
                             switch_threshold=-100 if always_alts else switch_pct / 100)   # -100 = keep every feasible supplier
    st.write(f"Current supplier {o['Supplier_ID']}: total expected cost {inr(cur)} (purchase + shipping + expected loss).")
    if alts.empty:
        st.info("No supplier meets the rules: active, enough stock, order size within limits, arrives in time, "
                "and quality at or above the minimum. Try lowering the minimum quality in the sidebar.")
    else:
        t = alts.copy()
        t["Unit_Cost_INR"] = t.Unit_Cost_INR.map(inr)
        t["Shipping_Cost_INR"] = t.Shipping_Cost_INR.map(inr)
        t["prob_late"] = t.prob_late.map(lambda x: f"{x:.0%}")
        for c in ("exp_loss_inr", "total_cost_inr", "saving_inr"):
            t[c] = t[c].map(inr)
        t = t.rename(columns={"Supplier_ID": "Supplier", "Supplier_Location": "Location", "Unit_Cost_INR": "Unit cost",
                              "Shipping_Cost_INR": "Shipping", "Lead_Time_Days": "Lead time (days)",
                              "Quality_Rating": "Quality", "prob_late": "Chance late", "exp_loss_inr": "Expected loss",
                              "total_cost_inr": "Total expected cost", "saving_inr": "Saving vs current"})
        show_table(t)
        st.caption("Each alternative is scored by the same model under this order's dates and quantity. "
                   "Confirm approval, price and capacity with the supplier before switching.")

    st.subheader("Other ways to reduce the loss")
    m = dp.mitigation_options(o)
    for c in ("cost_inr", "loss_avoided_inr", "net_benefit_inr"):
        m[c] = m[c].map(inr)
    show_table(m.rename(columns={"option": "Option", "cost_inr": "Cost", "loss_avoided_inr": "Loss avoided",
                                 "net_benefit_inr": "Net benefit"}))


# ------------------------------------------------------------------ tabs
tab1, tab2 = st.tabs(["Risk watchlist", "Check an order"])

with tab1:
    scored = score_recent(d, a, bundle, n_recent, delivery_path, alt_path, os.path.getmtime(MODEL_FILE))
    scored = scored.sort_values("exp_loss_inr", ascending=False).reset_index(drop=True)
    high = scored[scored.prob_late >= HIGH]
    k1, k2, k3 = st.columns(3)
    k1.metric("Orders scored", f"{len(scored):,}")
    k2.metric("High-risk orders", f"{len(high):,}")
    k3.metric("Expected loss on high-risk orders", inr(high.exp_loss_inr.sum()))

    st.subheader("Highest expected loss")
    top = scored.head(25)
    st.bar_chart(top.head(10).set_index("Shipment_ID")["exp_loss_inr"])
    t = top[["Shipment_ID", "Supplier_ID", "Item_ID", "Quantity_Ordered", "prob_late", "exp_delay_days",
             "days_if_late_range", "exp_loss_inr", "worst_case_loss_inr"]].copy()
    t["Risk"] = t.prob_late.map(band)
    t["prob_late"] = t.prob_late.map(lambda x: f"{x:.0%}")
    t["exp_delay_days"] = t.exp_delay_days.map(lambda x: f"{x:.1f}")
    t["exp_loss_inr"] = t.exp_loss_inr.map(inr)
    t["worst_case_loss_inr"] = t.worst_case_loss_inr.map(inr)
    show_table(t.rename(columns={"Quantity_Ordered": "Qty", "prob_late": "Chance late", "exp_delay_days": "Expected delay (d)",
                                 "days_if_late_range": "Range if late (d)", "exp_loss_inr": "Expected loss",
                                 "worst_case_loss_inr": "Worst-case loss"}))

    st.subheader("Shipment detail")
    pick = st.selectbox("Choose a shipment", list(top.Shipment_ID))
    o = scored[scored.Shipment_ID == pick].iloc[0]
    render_order(o, d.iloc[:-n_recent])
    st.caption(f"Backtest only (demo data): actual delay was {int(o['Delay_Days'])} days, "
               f"actual loss {inr(o['Production_Loss_Cost_INR'])}.")

with tab2:
    st.write("Enter a new order to get its delay risk, expected loss and alternatives.")
    item = st.selectbox("Item", sorted(a.Item_ID.unique()))
    supplier = st.selectbox("Supplier", sorted(a[a.Item_ID == item].Supplier_ID))
    pair = a[(a.Item_ID == item) & (a.Supplier_ID == supplier)].iloc[0]
    past = d[d.Item_ID == item]
    default_loss = int(past.Line_Loss_Per_Day_INR.median()) if len(past) else 300000
    default_stock = int(round(past.Stock_On_Hand_Days.median())) if len(past) else 3

    col1, col2 = st.columns(2)
    order_date = col1.date_input("Order date", value=(d.Order_Date.max() + pd.Timedelta(days=1)).date())
    promised = col2.date_input("Promised delivery date",
                               value=(pd.Timestamp(order_date) + pd.Timedelta(days=int(pair.Lead_Time_Days) + 2)).date())
    col3, col4 = st.columns(2)
    qty = col3.number_input("Quantity", min_value=1, max_value=5000, value=200, step=10)
    unit_cost = col4.number_input("Unit cost (INR)", min_value=1.0, value=float(pair.Unit_Cost_INR), step=10.0)
    col5, col6 = st.columns(2)
    stock = col5.number_input("Stock on hand (days)", min_value=0, max_value=60, value=default_stock, step=1)
    loss_day = col6.number_input("Line loss per day (INR)", min_value=0, value=default_loss, step=10000)

    if st.button("Assess order"):
        if pd.Timestamp(promised) <= pd.Timestamp(order_date):
            st.error("The promised date must be after the order date.")
        else:
            new = pd.DataFrame([{"Shipment_ID": "NEW-ORDER", "Supplier_ID": supplier, "Item_ID": item,
                                 "Order_Date": pd.Timestamp(order_date), "Promised_Date": pd.Timestamp(promised),
                                 "Quantity_Ordered": int(qty), "Unit_Cost_INR": float(unit_cost),
                                 "Stock_On_Hand_Days": int(stock), "Line_Loss_Per_Day_INR": int(loss_day)}])
            res = dp.predict_orders(new, d, a, bundle).iloc[0]
            render_order(res, d)

with st.expander("How this works"):
    st.markdown(
        "- A gradient boosting model (LightGBM) predicts the chance a shipment is late and how many days late.\n"
        "- Loss = max(0, delay days - stock on hand days) x line loss per day, weighted by the chance of being late.\n"
        "- Alternatives are other suppliers of the same item that are active, have stock, can deliver in time and meet the "
        "quality limit. Each is scored with the same model, then ranked on purchase + shipping + expected loss.\n"
        "- Supplier history uses only shipments delivered before the order date.")