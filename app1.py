"""Streamlit dashboard for supplier delay and production-loss analysis."""

from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import delay_pipeline as dp


PROJECT_DIR = Path(__file__).resolve().parent
DELIVERY_DEFAULT = PROJECT_DIR / "automotive_supplier_delivery_data_realistic_100k.csv"
ALTERNATIVES_DEFAULT = PROJECT_DIR / "automotive_alternative_supplier_data_realistic_100k.csv"
MODEL_FILE = PROJECT_DIR / dp.MODEL_DIR / "delay_models.joblib"
HIGH_RISK = 0.40
MEDIUM_RISK = 0.20


def format_inr(value: float) -> str:
    return f"INR {value:,.0f}"


def risk_band(probability: float) -> str:
    if probability >= HIGH_RISK:
        return "High"
    if probability >= MEDIUM_RISK:
        return "Medium"
    return "Low"


@st.cache_resource(show_spinner="Loading shipment data and model...")
def load_all(delivery_path: str, alternative_path: str, model_modified: float):
    delivery, alternatives = dp.load_data(delivery_path, alternative_path)
    delivery = delivery.sort_values("Order_Date").reset_index(drop=True)
    model = joblib.load(MODEL_FILE)
    return delivery, alternatives, model


@st.cache_data(show_spinner="Scoring recent orders...")
def score_recent(_delivery, _alternatives, _model, count: int):
    recent = _delivery.tail(count).copy()
    history = _delivery.iloc[:-count]
    actual = recent[
        ["Shipment_ID", "Delay_Days", "Production_Loss_Cost_INR"]
    ].copy()
    orders = recent.drop(
        columns=["Actual_Delivery_Date", "Delay_Days", "Production_Loss_Cost_INR"]
    )
    predictions = dp.predict_orders(orders, history, _alternatives, _model)
    return predictions.merge(actual, on="Shipment_ID", how="left"), history


def show_table(data: pd.DataFrame) -> None:
    st.dataframe(data, hide_index=True, width="stretch")


def render_order(order: pd.Series, history: pd.DataFrame, alternatives: pd.DataFrame, model) -> None:
    probability = float(order["prob_late"])
    band = risk_band(probability)
    message = f"{band} risk: {probability:.0%} chance this shipment is late."
    if band == "High":
        st.error(message)
    elif band == "Medium":
        st.warning(message)
    else:
        st.success(message)

    metric_columns = st.columns(5)
    metric_columns[0].metric("Chance of being late", f"{probability:.0%}")
    metric_columns[1].metric(
        "Expected delay (overall)", f"{order['exp_delay_days']:.1f} days"
    )
    metric_columns[2].metric(
        "Estimated delay if late", f"{order['days_if_late']:.1f} days"
    )
    metric_columns[3].metric("Expected loss", format_inr(order["exp_loss_inr"]))
    metric_columns[4].metric("Worst-case loss", format_inr(order["worst_case_loss_inr"]))

    st.caption(
        f"If late, estimated delay range: {order['days_if_late_range']} days. "
        f"Stock on hand covers {int(order['Stock_On_Hand_Days'])} days; "
        f"line loss is {format_inr(order['Line_Loss_Per_Day_INR'])} per day."
    )

    if probability >= MEDIUM_RISK or st.session_state.get("always_list_alternatives", True):
        st.subheader("Alternative suppliers")
        current_cost, recommendations = dp.recommend(
            order,
            alternatives,
            history,
            model,
            as_of=order["Order_Date"],
            min_quality=st.session_state["minimum_quality"],
            switch_threshold=st.session_state["switch_percent"] / 100,
        )
        st.write(
            f"Current supplier {order['Supplier_ID']}: expected total cost "
            f"{format_inr(current_cost)} (purchase, shipping and expected loss)."
        )
        if recommendations.empty:
            st.info(
                "No alternative meets the current availability, lead-time, "
                "quality, and savings requirements."
            )
        else:
            table = recommendations.copy()
            table["prob_late"] = table["prob_late"].map(lambda value: f"{value:.0%}")
            for column in (
                "Unit_Cost_INR",
                "Shipping_Cost_INR",
                "exp_loss_inr",
                "total_cost_inr",
                "saving_inr",
            ):
                table[column] = table[column].map(format_inr)
            table = table.rename(
                columns={
                    "Supplier_ID": "Supplier",
                    "Supplier_Location": "Location",
                    "Unit_Cost_INR": "Unit cost",
                    "Shipping_Cost_INR": "Shipping",
                    "Lead_Time_Days": "Lead time (days)",
                    "Quality_Rating": "Quality",
                    "prob_late": "Chance late",
                    "exp_loss_inr": "Expected loss",
                    "total_cost_inr": "Total expected cost",
                    "saving_inr": "Saving vs current",
                }
            )
            show_table(table)

    st.subheader("Mitigation options")
    mitigation = dp.mitigation_options(order).copy()
    for column in ("cost_inr", "loss_avoided_inr", "net_benefit_inr"):
        mitigation[column] = mitigation[column].map(format_inr)
    show_table(
        mitigation.rename(
            columns={
                "option": "Option",
                "cost_inr": "Cost",
                "loss_avoided_inr": "Loss avoided",
                "net_benefit_inr": "Net benefit",
            }
        )
    )


st.set_page_config(
    page_title="Supplier Delay and Loss Predictor",
    page_icon="📦",
    layout="wide",
)
st.title("Supplier Delay, Production Loss and Alternatives")
st.caption(
    "Delay-risk estimates and alternative supplier analysis. "
    "The bundled dataset is simulated demo data."
)

with st.sidebar:
    st.header("Settings")
    delivery_path = st.text_input("Delivery CSV", str(DELIVERY_DEFAULT))
    alternative_path = st.text_input("Alternative supplier CSV", str(ALTERNATIVES_DEFAULT))
    recent_count = st.slider("Recent orders to score", 100, 1000, 300, step=50)
    st.session_state["minimum_quality"] = st.slider(
        "Minimum supplier quality rating", 3.0, 5.0, 4.0, step=0.1
    )
    st.session_state["switch_percent"] = st.slider(
        "Minimum savings to switch (% of order value)", 0.0, 10.0, 2.0, step=0.5
    )
    st.session_state["always_list_alternatives"] = st.checkbox(
        "Always list alternatives", value=True
    )
    st.caption("High risk: 40% or more · Medium: 20% to below 40% · Low: below 20%.")

missing_paths = [
    str(path)
    for path in (Path(delivery_path), Path(alternative_path), MODEL_FILE)
    if not path.is_file()
]
if missing_paths:
    st.error("Required file(s) not found:\n\n" + "\n".join(f"- {path}" for path in missing_paths))
    st.info("Place the files in the project folder or update the CSV paths in the sidebar.")
    st.stop()

try:
    delivery, alternatives, model = load_all(
        delivery_path,
        alternative_path,
        MODEL_FILE.stat().st_mtime,
    )
except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
    st.error(f"Could not load the data or model: {error}")
    st.stop()

if len(delivery) < 2:
    st.error("At least two delivery records are required to score recent orders.")
    st.stop()

recent_count = min(recent_count, len(delivery) - 1)
try:
    scored, history = score_recent(delivery, alternatives, model, recent_count)
except (ValueError, KeyError, IndexError) as error:
    st.error(f"Could not score recent orders: {error}")
    st.stop()

watchlist_tab, order_tab = st.tabs(["Risk watchlist", "Check an order"])

with watchlist_tab:
    scored = scored.sort_values("exp_loss_inr", ascending=False).reset_index(drop=True)
    high_risk = scored[scored["prob_late"] >= HIGH_RISK]

    metric_columns = st.columns(3)
    metric_columns[0].metric("Orders scored", f"{len(scored):,}")
    metric_columns[1].metric("High-risk orders", f"{len(high_risk):,}")
    metric_columns[2].metric(
        "Expected loss on high-risk orders",
        format_inr(high_risk["exp_loss_inr"].sum()),
    )

    st.subheader("Highest expected loss")
    st.bar_chart(
        scored.head(10).set_index("Shipment_ID")["exp_loss_inr"],
        x_label="Shipment",
        y_label="Expected loss (INR)",
    )

    table = scored.head(25)[
        [
            "Shipment_ID",
            "Supplier_ID",
            "Item_ID",
            "Quantity_Ordered",
            "prob_late",
            "exp_delay_days",
            "days_if_late_range",
            "exp_loss_inr",
            "worst_case_loss_inr",
        ]
    ].copy()
    table["Risk"] = table["prob_late"].map(risk_band)
    table["prob_late"] = table["prob_late"].map(lambda value: f"{value:.0%}")
    table["exp_delay_days"] = table["exp_delay_days"].map(lambda value: f"{value:.1f}")
    for column in ("exp_loss_inr", "worst_case_loss_inr"):
        table[column] = table[column].map(format_inr)
    show_table(
        table.rename(
            columns={
                "Quantity_Ordered": "Quantity",
                "prob_late": "Chance late",
                "exp_delay_days": "Expected delay (days)",
                "days_if_late_range": "Delay range if late (days)",
                "exp_loss_inr": "Expected loss",
                "worst_case_loss_inr": "Worst-case loss",
            }
        )
    )

    st.subheader("Shipment detail")
    shipment_id = st.selectbox("Choose a shipment", scored["Shipment_ID"].tolist())
    selected_order = scored[scored["Shipment_ID"] == shipment_id].iloc[0]
    render_order(selected_order, history, alternatives, model)
    st.caption(
        f"Demo backtest: actual delay was {int(selected_order['Delay_Days'])} days; "
        f"actual loss was {format_inr(selected_order['Production_Loss_Cost_INR'])}."
    )

with order_tab:
    st.write("Enter a proposed order to estimate its delay risk and expected loss.")
    item_id = st.selectbox("Item", sorted(alternatives["Item_ID"].dropna().unique()))
    item_suppliers = alternatives[alternatives["Item_ID"] == item_id]
    supplier_id = st.selectbox("Supplier", sorted(item_suppliers["Supplier_ID"].dropna().unique()))
    offer = item_suppliers[item_suppliers["Supplier_ID"] == supplier_id].iloc[0]
    item_history = delivery[delivery["Item_ID"] == item_id]

    default_loss = (
        int(item_history["Line_Loss_Per_Day_INR"].median())
        if not item_history.empty
        else 300_000
    )
    default_stock = (
        int(round(item_history["Stock_On_Hand_Days"].median()))
        if not item_history.empty
        else 3
    )

    date_column_1, date_column_2 = st.columns(2)
    order_date = date_column_1.date_input(
        "Order date",
        value=(pd.Timestamp(delivery["Order_Date"].max()) + timedelta(days=1)).date(),
    )
    promised_date = date_column_2.date_input(
        "Expected delivery date (supplier promised)",
        value=(
            pd.Timestamp(order_date)
            + timedelta(days=int(offer["Lead_Time_Days"]) + 2)
        ).date(),
    )
    quantity_column, price_column = st.columns(2)
    quantity = quantity_column.number_input(
        "Quantity", min_value=1, max_value=5000, value=200, step=10
    )
    unit_cost = price_column.number_input(
        "Unit cost (INR)", min_value=1.0, value=float(offer["Unit_Cost_INR"]), step=10.0
    )
    stock_column, loss_column = st.columns(2)
    stock_days = stock_column.number_input(
        "Stock on hand (days)", min_value=0, max_value=60, value=default_stock, step=1
    )
    loss_per_day = loss_column.number_input(
        "Line loss per day (INR)", min_value=0, value=default_loss, step=10_000
    )

    if st.button("Assess order"):
        if promised_date <= order_date:
            st.error("The promised delivery date must be after the order date.")
        else:
            new_order = pd.DataFrame(
                [
                    {
                        "Shipment_ID": "NEW-ORDER",
                        "Supplier_ID": supplier_id,
                        "Item_ID": item_id,
                        "Order_Date": pd.Timestamp(order_date),
                        "Promised_Date": pd.Timestamp(promised_date),
                        "Quantity_Ordered": int(quantity),
                        "Unit_Cost_INR": float(unit_cost),
                        "Stock_On_Hand_Days": int(stock_days),
                        "Line_Loss_Per_Day_INR": int(loss_per_day),
                    }
                ]
            )
            try:
                prediction = dp.predict_orders(new_order, delivery, alternatives, model).iloc[0]
                render_order(prediction, delivery, alternatives, model)
            except (ValueError, KeyError, IndexError) as error:
                st.error(f"Could not assess this order: {error}")

with st.expander("How this works"):
    st.markdown(
        "- A trained model estimates the probability of delay and delay duration.\n"
        "- Expected production loss is based on delay risk, stock coverage, and line loss per day.\n"
        "- Alternative suppliers are filtered by item, status, capacity, lead time, and quality, "
        "then compared by purchase cost, shipping, and expected loss.\n"
        "- Supplier-history features use shipments delivered before the order date."
    )
1