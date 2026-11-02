"""Balanced alternative-supplier recommender for the automotive delay project.

Runs a leakage-conscious historical replay on the newest N shipments:
- earlier shipments only are used as history for the replay window;
- actual delay/loss columns are excluded from prediction inputs;
- alternatives are filtered by item, active status, quantity, quality and lead time;
- alternatives are ranked by purchase + shipping + expected production-loss cost.

Run from the project folder:
python run_supplier_recommendations.py --delivery automotive_supplier_delivery_data_realistic_100k.csv --alt automotive_alternative_supplier_data_realistic_100k.csv --model models/delay_models.joblib --recent-n 300
"""
import argparse
import importlib.util
import os
import joblib
import numpy as np
import pandas as pd


def load_pipeline(path):
    spec = importlib.util.spec_from_file_location("calibrated_delay_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", required=True)
    parser.add_argument("--alt", required=True)
    parser.add_argument("--model", default="models/delay_models.joblib")
    parser.add_argument("--pipeline", default="delay_pipeline_quantile_calibrated.py")
    parser.add_argument("--recent-n", type=int, default=300)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--min-quality", type=float, default=4.0)
    parser.add_argument("--switch-threshold", type=float, default=0.02,
                        help="Minimum saving as a fraction of current purchase value")
    parser.add_argument("--out", default="recommendation_output")
    args = parser.parse_args()

    if args.recent_n < 1:
        parser.error("--recent-n must be at least 1")
    for path in [args.delivery, args.alt, args.model, args.pipeline]:
        if not os.path.isfile(path):
            parser.error(f"File not found: {path}")

    pipe = load_pipeline(args.pipeline)
    delivery, alternatives = pipe.load_data(args.delivery, args.alt)
    delivery = delivery.sort_values("Order_Date").reset_index(drop=True)
    if args.recent_n >= len(delivery):
        parser.error("--recent-n must be smaller than the number of delivery records")

    bundle = joblib.load(args.model)
    if "qmodels" not in bundle or "quantile_offsets" not in bundle:
        raise RuntimeError("Model bundle is not the calibrated quantile model expected by this pipeline.")
    if "stockout_clf" not in bundle or "excess_delay_model" not in bundle:
        raise RuntimeError("Model bundle is missing the stock-out models; retrain it with the selected pipeline.")
    if list(bundle.get("features", [])) != list(pipe.FEATURES):
        raise RuntimeError("Model feature list does not match the selected calibrated pipeline. Use the matching pipeline/model pair.")
    if list(bundle.get("stockout_features", [])) != list(pipe.STOCKOUT_FEATURES):
        raise RuntimeError("Stock-out model feature list does not match the selected pipeline.")

    replay = delivery.tail(args.recent_n).copy()
    history = delivery.iloc[:-args.recent_n].copy()
    actual = replay[
        ["Shipment_ID", "Actual_Delivery_Date", "Delay_Days", "Production_Loss_Cost_INR"]
    ].copy()

    scored = pipe.sequential_historical_replay(
        replay, history, alternatives, bundle
    )
    actual_by_id = actual.set_index("Shipment_ID")
    scored["actual_delivery_date_for_replay_only"] = scored["Shipment_ID"].map(
        actual_by_id["Actual_Delivery_Date"]
    )
    scored["actual_delay_days"] = scored["Shipment_ID"].map(actual_by_id["Delay_Days"])
    scored["actual_loss_inr"] = scored["Shipment_ID"].map(actual_by_id["Production_Loss_Cost_INR"])
    scored["actual_stockout"] = (
        scored["actual_delay_days"] > scored["Stock_On_Hand_Days"]
    ).astype(int)

    threshold = float(bundle.get("classification_threshold", 0.5))
    high_risk = scored[scored["prob_late"] >= threshold].sort_values("exp_loss_inr", ascending=False).copy()

    summary_rows, recommendation_rows = [], []
    for _, order in high_risk.iterrows():
        current_total, ranked = pipe.recommend(
            order, alternatives, delivery, bundle, as_of=order["Order_Date"],
            top_n=args.top_n, switch_threshold=args.switch_threshold,
            min_quality=args.min_quality
        )
        best = ranked.iloc[0] if not ranked.empty else None
        summary_rows.append({
            "Shipment_ID": order["Shipment_ID"],
            "Current_Supplier_ID": order["Supplier_ID"],
            "Item_ID": order["Item_ID"],
            "Quantity_Ordered": order["Quantity_Ordered"],
            "Order_Date": order["Order_Date"],
            "prob_late": order["prob_late"],
            "stockout_probability": order["stockout_probability"],
            "expected_delay_days": order["exp_delay_days"],
            "delay_range_if_late": order["days_if_late_range"],
            "expected_production_loss_inr": order["exp_loss_inr"],
            "experimental_stockout_loss_inr": order["stockout_expected_loss_inr"],
            "actual_delay_days_for_retro_test_only": order["actual_delay_days"],
            "actual_loss_inr_for_retro_test_only": order["actual_loss_inr"],
            "current_total_expected_cost_inr": current_total,
            "current_supplier_cost_status": ranked.attrs.get(
                "status", "available"
            ),
            "eligible_alternative_count_returned": len(ranked),
            "recommended_supplier_id": best["Supplier_ID"] if best is not None else None,
            "recommended_total_expected_cost_inr": best["total_cost_inr"] if best is not None else np.nan,
            "estimated_saving_inr": best["saving_inr"] if best is not None else np.nan,
            "recommended_supplier_late_probability": best["prob_late"] if best is not None else np.nan,
            "recommended_supplier_stockout_probability": best["stockout_probability"] if best is not None else np.nan,
        })
        for rank, (_, candidate) in enumerate(ranked.iterrows(), start=1):
            recommendation_rows.append({
                "Shipment_ID": order["Shipment_ID"],
                "Current_Supplier_ID": order["Supplier_ID"],
                "Item_ID": order["Item_ID"],
                "rank": rank,
                **candidate.to_dict(),
            })

    os.makedirs(args.out, exist_ok=True)
    scored.to_csv(os.path.join(args.out, "all_scored_shipments.csv"), index=False)
    high_risk.to_csv(os.path.join(args.out, "high_risk_shipments.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out, "recommendation_summary.csv"), index=False)
    pd.DataFrame(recommendation_rows).to_csv(os.path.join(args.out, "alternative_supplier_options.csv"), index=False)

    actual_total = float(scored["actual_loss_inr"].sum())
    pred_total = float(scored["exp_loss_inr"].sum())
    high_actual = float(high_risk["actual_loss_inr"].sum())
    high_pred = float(high_risk["exp_loss_inr"].sum())
    stockout_probability = scored["stockout_probability"].to_numpy(dtype=float)
    actual_stockout = scored["actual_stockout"].to_numpy(dtype=int)
    reliability = pipe.stockout_reliability_table(
        actual_stockout, stockout_probability
    )
    calibration = pipe.stockout_calibration_metrics(reliability)
    replay_metrics = {
        "replay_rows": len(scored),
        "actual_stockout_rate": float(actual_stockout.mean()),
        "predicted_stockout_rate": float(stockout_probability.mean()),
        "stockout_brier_score": float(
            pipe.brier_score_loss(actual_stockout, stockout_probability)
        ),
        "stockout_expected_calibration_error": calibration[
            "expected_calibration_error"
        ],
        "stockout_max_calibration_error": calibration["max_calibration_error"],
        "stockout_roc_auc": (
            float(pipe.roc_auc_score(actual_stockout, stockout_probability))
            if len(np.unique(actual_stockout)) == 2 else np.nan
        ),
        "stockout_average_precision": (
            float(pipe.average_precision_score(actual_stockout, stockout_probability))
            if actual_stockout.sum() else np.nan
        ),
        "quantile_expected_loss_mae_inr": float(
            pipe.mean_absolute_error(
                scored["actual_loss_inr"], scored["exp_loss_inr"]
            )
        ),
        "quantile_expected_loss_rmse_inr": float(
            np.sqrt(
                pipe.mean_squared_error(
                    scored["actual_loss_inr"], scored["exp_loss_inr"]
                )
            )
        ),
        "quantile_expected_loss_total_bias_pct": (
            (pred_total / actual_total - 1) * 100 if actual_total else np.nan
        ),
    }

    os.makedirs("evaluation", exist_ok=True)
    pd.DataFrame([replay_metrics]).to_csv(
        "evaluation/sequential_replay_metrics.csv", index=False
    )
    reliability.to_csv(
        "evaluation/sequential_replay_stockout_reliability.csv", index=False
    )
    print("\n=== BALANCED ALTERNATIVE SUPPLIER RECOMMENDATIONS ===")
    print(f"Historical replay rows: {len(scored):,}")
    print(f"Risk threshold from saved model: {threshold:.4f}")
    print(f"High-risk shipments: {len(high_risk):,}")
    print(f"High-risk shipments with at least one returned alternative: {sum(x['eligible_alternative_count_returned'] > 0 for x in summary_rows):,}")
    print(f"High-risk shipments with no returned alternative: {sum(x['eligible_alternative_count_returned'] == 0 for x in summary_rows):,}")
    print(f"Replay actual loss (retrospective only): INR {actual_total:,.0f}")
    print(f"Replay predicted expected loss: INR {pred_total:,.0f}")
    print(f"High-risk group actual loss (retrospective only): INR {high_actual:,.0f}")
    print(f"High-risk group predicted expected loss: INR {high_pred:,.0f}")
    print(
        f"Sequential replay stock-out Brier: "
        f"{replay_metrics['stockout_brier_score']:.4f} | "
        f"ECE: {replay_metrics['stockout_expected_calibration_error']:.4f}"
    )
    print(
        "High-risk orders with unavailable current-supplier shipping cost: "
        f"{sum(row['current_supplier_cost_status'] != 'available' for row in summary_rows):,}"
    )
    if not pd.DataFrame(recommendation_rows).empty:
        rec = pd.DataFrame(recommendation_rows)
        print(f"Alternative options exported: {len(rec):,}")
        print("\nTop 10 recommended alternatives by estimated saving:")
        print(rec.sort_values("saving_inr", ascending=False)[
            ["Shipment_ID", "Current_Supplier_ID", "Supplier_ID", "Item_ID",
             "prob_late", "stockout_probability", "total_cost_inr", "saving_inr"]
        ].head(10).round(2).to_string(index=False))
    print(f"\nFiles saved in: {os.path.abspath(args.out)}")
    for name in ["all_scored_shipments.csv", "high_risk_shipments.csv", "recommendation_summary.csv", "alternative_supplier_options.csv"]:
        print(f" - {name}")
    print("Sequential replay evaluation:")
    print(" - evaluation/sequential_replay_metrics.csv")
    print(" - evaluation/sequential_replay_stockout_reliability.csv")
    print("\nImportant: this is a historical replay, not proof of future savings. Validate recommendations with live quotes, inventory confirmation, and operational approval before switching suppliers.")


if __name__ == "__main__":
    main()
