"""Streamlit dashboard for calibrated supplier-delay prediction and alternatives."""

from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Automotive Supplier Risk Dashboard",
    page_icon="📦",
    layout="wide",
)

try:
    import delay_pipeline_quantile_calibrated as dp
except Exception as import_error:
    st.error("Could not import delay_pipeline_quantile_calibrated.py")
    st.exception(import_error)
    st.stop()


PROJECT_DIR = Path(__file__).resolve().parent
DELIVERY_DEFAULT = PROJECT_DIR / "automotive_supplier_delivery_data_realistic_100k.csv"
ALTERNATIVES_DEFAULT = PROJECT_DIR / "automotive_alternative_supplier_data_realistic_100k.csv"
MODEL_FILE = PROJECT_DIR / dp.MODEL_DIR / "delay_models.joblib"


def format_inr(value: float) -> str:
    return f"₹{value:,.0f}"


def get_threshold(model) -> float:
    return float(model.get("classification_threshold", 0.5))


def risk_band(probability: float, threshold: float) -> str:
    # The model-selected validation threshold defines the High-risk boundary.
    medium_threshold = threshold * 0.65
    if probability >= threshold:
        return "High"
    if probability >= medium_threshold:
        return "Medium"
    return "Low"


@st.cache_resource(show_spinner="Loading shipment data and calibrated model...")
def load_all(delivery_path: str, alternative_path: str, model_path: str, model_modified: float):
    delivery, alternatives = dp.load_data(delivery_path, alternative_path)
    delivery = delivery.sort_values("Order_Date").reset_index(drop=True)
    model = joblib.load(model_path)
    # Guard against accidentally loading a model from a different pipeline version.
    if "qmodels" not in model or "quantile_offsets" not in model:
        raise ValueError(
            "The selected model is not the calibrated quantile model. "
            "Use the delay_models.joblib trained by delay_pipeline_quantile_calibrated.py."
        )
    if (
        "stockout_clf" not in model
        or "excess_delay_model" not in model
        or list(model.get("stockout_features", [])) != list(dp.STOCKOUT_FEATURES)
    ):
        raise ValueError(
            "The selected model bundle does not include stock-out estimates. "
            "Retrain it with delay_pipeline_quantile_calibrated.py."
        )
    if list(model.get("features", [])) != list(dp.FEATURES):
        raise ValueError("Model feature list does not match the calibrated pipeline.")
    return delivery, alternatives, model


@st.cache_data(show_spinner="Scoring recent orders...")
def score_recent(_delivery, _alternatives, _model, count: int):
    recent = _delivery.tail(count).copy()
    history = _delivery.iloc[:-count]
    actual = recent[
        [
            "Shipment_ID",
            "Actual_Delivery_Date",
            "Delay_Days",
            "Production_Loss_Cost_INR",
        ]
    ].copy()
    predictions = dp.sequential_historical_replay(
        recent, history, _alternatives, _model
    )
    return predictions.merge(actual, on="Shipment_ID", how="left"), _delivery


def show_table(data: pd.DataFrame) -> None:
    st.dataframe(data, hide_index=True, width="stretch")


def render_order(
    order: pd.Series,
    history: pd.DataFrame,
    alternatives: pd.DataFrame,
    model,
    min_quality: float,
    switch_percent: float,
    always_list_alternatives: bool,
) -> None:
    probability = float(order["prob_late"])
    threshold = get_threshold(model)
    band = risk_band(probability, threshold)
    message = (
        f"{band} risk: {probability:.1%} chance this shipment is late. "
        f"Model-selected high-risk threshold: {threshold:.1%}."
    )
    if band == "High":
        st.error(message)
    elif band == "Medium":
        st.warning(message)
    else:
        st.success(message)

    metric_columns = st.columns(5)
    metric_columns[0].metric("Chance of being late", f"{probability:.1%}")
    metric_columns[1].metric(
        "Expected delay (overall)", f"{order['exp_delay_days']:.1f} days"
    )
    metric_columns[2].metric(
        "Estimated delay if late", f"{order['days_if_late']:.1f} days"
    )
    metric_columns[3].metric("Expected production loss", format_inr(order["exp_loss_inr"]))
    metric_columns[4].metric("90th-percentile loss", format_inr(order["worst_case_loss_inr"]))

    with st.expander("Stock-out risk and experimental loss estimate"):
        stockout_columns = st.columns(3)
        stockout_columns[0].metric(
            "Chance stock runs out", f"{order['stockout_probability']:.1%}"
        )
        stockout_columns[1].metric(
            "Expected excess delay after stock runs out",
            f"{order['expected_excess_delay_days']:.1f} days",
        )
        stockout_columns[2].metric(
            "Experimental stock-out loss",
            format_inr(order["stockout_expected_loss_inr"]),
        )
        st.caption(
            "This separate stock-out model estimates the chance delivery delay exceeds available stock "
            "and the resulting excess-delay loss. It is experimental; the expected-loss estimate above "
            "continues to drive supplier rankings and mitigation calculations until rolling-period "
            "validation confirms an improvement."
        )

    st.caption(
        f"If late, estimated delay range: {order['days_if_late_range']} days. "
        f"Stock covers {int(order['Stock_On_Hand_Days'])} days; "
        f"line loss is {format_inr(order['Line_Loss_Per_Day_INR'])} per day."
    )

    if probability >= threshold or always_list_alternatives:
        st.subheader("Alternative supplier recommendations")
        current_cost, recommendations = dp.recommend(
            order,
            alternatives,
            history,
            model,
            as_of=order["Order_Date"],
            top_n=3,
            min_quality=min_quality,
            switch_threshold=switch_percent / 100,
        )
        recommendation_status = recommendations.attrs.get("status", "available")
        if pd.isna(current_cost):
            st.warning(
                "Savings cannot be calculated because the current supplier's "
                "shipping cost is missing or ambiguous in the supplier quote data."
            )
        else:
            st.write(
                f"Current supplier **{order['Supplier_ID']}**: expected total cost "
                f"**{format_inr(current_cost)}** (purchase + shipping + expected production loss)."
            )
        if recommendation_status == "alternatives_missing_shipping_cost":
            st.warning(
                "Some otherwise eligible alternatives were excluded because their "
                "shipping-cost quotes are missing or invalid."
            )
        if recommendations.empty:
            if recommendation_status == "alternatives_missing_shipping_cost":
                st.info(
                    "No cost comparison is available: eligible alternatives have missing "
                    "shipping-cost quotes."
                )
            elif recommendation_status == "missing_current_supplier_quote":
                st.info("The current supplier-item quote is missing; savings cannot be compared.")
            elif recommendation_status == "missing_current_supplier_shipping_cost":
                st.info("The current supplier shipping-cost quote is missing or invalid.")
            elif recommendation_status == "ambiguous_current_supplier_shipping_cost":
                st.info("The current supplier has conflicting shipping-cost quotes.")
            else:
                st.info(
                    "No alternative currently meets the availability, lead-time, quality, "
                    "and minimum-savings requirements."
                )
        else:
            table = recommendations.copy()
            table["prob_late"] = table["prob_late"].map(lambda value: f"{value:.1%}")
            table["stockout_probability"] = table["stockout_probability"].map(
                lambda value: f"{value:.1%}"
            )
            for column in (
                "Unit_Cost_INR",
                "Shipping_Cost_INR",
                "exp_loss_inr",
                "stockout_expected_loss_inr",
                "total_cost_inr",
                "saving_inr",
            ):
                table[column] = table[column].map(format_inr)
            table = table.rename(
                columns={
                    "Supplier_ID": "Alternative supplier",
                    "Supplier_Location": "Location",
                    "Unit_Cost_INR": "Unit cost",
                    "Shipping_Cost_INR": "Shipping cost",
                    "Lead_Time_Days": "Lead time (days)",
                    "Quality_Rating": "Quality rating",
                    "prob_late": "Chance of delay",
                    "stockout_probability": "Chance stock runs out",
                    "exp_loss_inr": "Expected loss",
                    "stockout_expected_loss_inr": "Experimental stock-out loss",
                    "total_cost_inr": "Total expected cost",
                    "saving_inr": "Estimated saving",
                }
            )
            show_table(table)
            st.caption(
                "Alternatives are ranked by purchase cost + shipping cost + expected production loss. "
                "Savings are model estimates, not guaranteed savings."
            )

    st.subheader("Mitigation options")
    mitigation = dp.mitigation_options(order).copy()
    for column in ("cost_inr", "loss_avoided_inr", "net_benefit_inr"):
        mitigation[column] = mitigation[column].map(format_inr)
    show_table(
        mitigation.rename(
            columns={
                "option": "Option",
                "cost_inr": "Estimated action cost",
                "loss_avoided_inr": "Estimated loss avoided",
                "net_benefit_inr": "Estimated net benefit",
            }
        )
    )



st.title("Automotive Supplier Delay & Production-Loss Dashboard")
st.caption(
    "Calibrated delay-risk prediction, expected production loss, and alternative supplier ranking. "
    "The supplied datasets are simulated demonstration data."
)

with st.sidebar:
    st.header("Dashboard settings")
    delivery_path = st.text_input("Delivery CSV", str(DELIVERY_DEFAULT))
    alternative_path = st.text_input("Alternative supplier CSV", str(ALTERNATIVES_DEFAULT))
    model_path = st.text_input("Calibrated model file", str(MODEL_FILE))
    recent_count = st.slider("Recent orders to score", 100, 1000, 300, step=50)
    minimum_quality = st.slider(
        "Minimum alternative supplier quality", 3.0, 5.0, 4.0, step=0.1
    )
    switch_percent = st.slider(
        "Minimum savings required to switch (%)", 0.0, 10.0, 2.0, step=0.5
    )
    always_list_alternatives = st.checkbox("Always show alternatives", value=True)

missing_paths = [
    str(path)
    for path in (Path(delivery_path), Path(alternative_path), Path(model_path))
    if not path.is_file()
]
if missing_paths:
    st.error("Required file(s) not found:\n\n" + "\n".join(f"- {path}" for path in missing_paths))
    st.info("Place the files in the project folder or correct their paths in the sidebar.")
    st.stop()

try:
    delivery, alternatives, model = load_all(
        delivery_path,
        alternative_path,
        model_path,
        Path(model_path).stat().st_mtime,
    )
except Exception as error:
    st.error(f"Could not load data or calibrated model: {error}")
    st.stop()

if len(delivery) < 2:
    st.error("At least two delivery records are required.")
    st.stop()

recent_count = min(recent_count, len(delivery) - 1)
try:
    scored, history = score_recent(delivery, alternatives, model, recent_count)
except Exception as error:
    st.error(f"Could not score orders: {error}")
    st.stop()

threshold = get_threshold(model)
medium_threshold = threshold * 0.65

watchlist_tab, order_tab, supplier_tab = st.tabs(
    ["Risk watchlist", "Check a new order", "Supplier overview"]
)

with watchlist_tab:
    scored = scored.sort_values("exp_loss_inr", ascending=False).reset_index(drop=True)
    high_risk = scored[scored["prob_late"] >= threshold]
    medium_risk = scored[
        (scored["prob_late"] >= medium_threshold) & (scored["prob_late"] < threshold)
    ]

    metric_columns = st.columns(4)
    metric_columns[0].metric("Orders scored", f"{len(scored):,}")
    metric_columns[1].metric("High-risk orders", f"{len(high_risk):,}")
    metric_columns[2].metric("Medium-risk orders", f"{len(medium_risk):,}")
    metric_columns[3].metric(
        "Expected loss across scored orders", format_inr(scored["exp_loss_inr"].sum())
    )
    st.caption(
        f"High risk ≥ {threshold:.1%} (selected on validation data); "
        f"medium risk ≥ {medium_threshold:.1%} and below the high-risk threshold."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Top 10 shipments by expected loss")
        st.bar_chart(
            scored.head(10).set_index("Shipment_ID")["exp_loss_inr"],
            x_label="Shipment",
            y_label="Expected loss (INR)",
        )
    with right:
        st.subheader("Risk distribution")
        risk_counts = pd.Series(
            {
                "High": int((scored["prob_late"] >= threshold).sum()),
                "Medium": int(
                    (
                        (scored["prob_late"] >= medium_threshold)
                        & (scored["prob_late"] < threshold)
                    ).sum()
                ),
                "Low": int((scored["prob_late"] < medium_threshold).sum()),
            }
        )
        st.bar_chart(risk_counts)

    st.subheader("Priority shipment list")
    table = scored.head(30)[
        [
            "Shipment_ID",
            "Supplier_ID",
            "Item_ID",
            "Quantity_Ordered",
            "prob_late",
            "stockout_probability",
            "exp_delay_days",
            "days_if_late_range",
            "exp_loss_inr",
            "stockout_expected_loss_inr",
            "worst_case_loss_inr",
        ]
    ].copy()
    table["Risk"] = table["prob_late"].map(lambda p: risk_band(p, threshold))
    table["prob_late"] = table["prob_late"].map(lambda value: f"{value:.1%}")
    table["stockout_probability"] = table["stockout_probability"].map(
        lambda value: f"{value:.1%}"
    )
    table["exp_delay_days"] = table["exp_delay_days"].map(lambda value: f"{value:.1f}")
    for column in (
        "exp_loss_inr",
        "stockout_expected_loss_inr",
        "worst_case_loss_inr",
    ):
        table[column] = table[column].map(format_inr)
    show_table(
        table.rename(
            columns={
                "Quantity_Ordered": "Quantity",
                "prob_late": "Chance of delay",
                "stockout_probability": "Chance stock runs out",
                "exp_delay_days": "Expected delay (days)",
                "days_if_late_range": "Delay range if late (days)",
                "exp_loss_inr": "Expected loss",
                "stockout_expected_loss_inr": "Experimental stock-out loss",
                "worst_case_loss_inr": "90th-percentile loss",
            }
        )
    )

    st.subheader("Shipment detail")
    shipment_id = st.selectbox("Choose a shipment", scored["Shipment_ID"].tolist())
    selected_order = scored[scored["Shipment_ID"] == shipment_id].iloc[0]
    render_order(
        selected_order,
        history,
        alternatives,
        model,
        minimum_quality,
        switch_percent,
        always_list_alternatives,
    )
    st.caption(
        f"Historical replay only: actual delay was {int(selected_order['Delay_Days'])} days; "
        f"actual production loss was {format_inr(selected_order['Production_Loss_Cost_INR'])}."
    )

with order_tab:
    st.write("Enter a proposed order to estimate delay risk, expected loss, and supplier alternatives.")
    item_id = st.selectbox("Item", sorted(alternatives["Item_ID"].dropna().unique()))
    item_suppliers = alternatives[alternatives["Item_ID"] == item_id]
    supplier_id = st.selectbox(
        "Current supplier",
        sorted(item_suppliers["Supplier_ID"].dropna().unique()),
    )
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
        "Supplier-promised delivery date",
        value=(
            pd.Timestamp(order_date) + timedelta(days=int(offer["Lead_Time_Days"]) + 2)
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
        "Production line loss per day (INR)",
        min_value=0,
        value=default_loss,
        step=10_000,
    )

    if st.button("Assess order", type="primary"):
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
                render_order(
                    prediction,
                    delivery,
                    alternatives,
                    model,
                    minimum_quality,
                    switch_percent,
                    True,
                )
            except Exception as error:
                st.error(f"Could not assess this order: {error}")

with supplier_tab:
    st.subheader("Supplier performance in the scored historical window")
    supplier_summary = (
        scored.groupby("Supplier_ID")
        .agg(
            Shipments=("Shipment_ID", "count"),
            Average_delay_risk=("prob_late", "mean"),
            Average_stockout_risk=("stockout_probability", "mean"),
            Expected_loss_INR=("exp_loss_inr", "sum"),
            Average_expected_delay_days=("exp_delay_days", "mean"),
        )
        .reset_index()
        .sort_values("Expected_loss_INR", ascending=False)
    )
    supplier_summary["Average_delay_risk"] = supplier_summary["Average_delay_risk"].map(
        lambda value: f"{value:.1%}"
    )
    supplier_summary["Average_stockout_risk"] = supplier_summary[
        "Average_stockout_risk"
    ].map(lambda value: f"{value:.1%}")
    supplier_summary["Expected_loss_INR"] = supplier_summary["Expected_loss_INR"].map(format_inr)
    supplier_summary["Average_expected_delay_days"] = supplier_summary[
        "Average_expected_delay_days"
    ].map(lambda value: f"{value:.1f}")
    show_table(
        supplier_summary.rename(
            columns={
                "Average_delay_risk": "Average chance of delay",
                "Average_stockout_risk": "Average chance stock runs out",
                "Expected_loss_INR": "Total expected loss",
                "Average_expected_delay_days": "Average expected delay (days)",
            }
        ).head(50)
    )

with st.expander("How the dashboard works and important limitations"):
    st.markdown(
        "- The calibrated classifier estimates the probability that a shipment will be late.\n"
        "- Quantile regression estimates the delay distribution; calibration offsets are applied from a separate calibration period.\n"
        "- Expected production loss combines delay probability, stock coverage, and production-line loss per day.\n"
        "- Alternative suppliers are filtered by item, active status, available quantity, order limits, lead time, and quality. They are ranked by purchase cost + shipping + expected production loss.\n"
        "- Supplier-history features use only shipments delivered before the order date.\n"
        "- The risk threshold comes from the saved model's validation-selected threshold; the medium band is a dashboard display band.\n"
        "- The supplied datasets are simulated demo data. Historical replay results and estimated savings are not guaranteed future outcomes.\n"
        "- Before production use, validate on later real-world periods and confirm current supplier quotes, stock, capacity, and delivery commitments."
    )