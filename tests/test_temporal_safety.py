import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import delay_pipeline_quantile_calibrated as pipeline


class ConstantProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, _features):
        return np.column_stack(
            [1 - self.probabilities, self.probabilities]
        )


class TemporalSafetyTests(unittest.TestCase):
    def test_replay_admits_only_previously_delivered_outcomes(self):
        base_history = pd.DataFrame(
            [
                {
                    "Shipment_ID": "BASE",
                    "Supplier_ID": "SUP-1",
                    "Order_Date": pd.Timestamp("2026-01-01"),
                    "Promised_Date": pd.Timestamp("2026-01-03"),
                    "Actual_Delivery_Date": pd.Timestamp("2026-01-02"),
                    "Delay_Days": 1,
                    "Production_Loss_Cost_INR": 0.0,
                }
            ]
        )
        replay = pd.DataFrame(
            [
                {
                    "Shipment_ID": "A",
                    "Supplier_ID": "SUP-1",
                    "Order_Date": pd.Timestamp("2026-01-03"),
                    "Promised_Date": pd.Timestamp("2026-01-05"),
                    "Actual_Delivery_Date": pd.Timestamp("2026-01-10"),
                    "Delay_Days": 7,
                    "Production_Loss_Cost_INR": 100.0,
                },
                {
                    "Shipment_ID": "B",
                    "Supplier_ID": "SUP-1",
                    "Order_Date": pd.Timestamp("2026-01-05"),
                    "Promised_Date": pd.Timestamp("2026-01-08"),
                    "Actual_Delivery_Date": pd.Timestamp("2026-01-06"),
                    "Delay_Days": 1,
                    "Production_Loss_Cost_INR": 0.0,
                },
                {
                    "Shipment_ID": "C",
                    "Supplier_ID": "SUP-1",
                    "Order_Date": pd.Timestamp("2026-01-12"),
                    "Promised_Date": pd.Timestamp("2026-01-14"),
                    "Actual_Delivery_Date": pd.Timestamp("2026-01-15"),
                    "Delay_Days": 3,
                    "Production_Loss_Cost_INR": 0.0,
                },
            ]
        )
        observed_prior_counts = []

        def fake_predict(prepared, _bundle):
            self.assertFalse(
                {
                    "Actual_Delivery_Date",
                    "Delay_Days",
                    "Production_Loss_Cost_INR",
                }.intersection(prepared.columns)
            )
            observed_prior_counts.extend(prepared["h_n"].astype(int).tolist())
            return pd.DataFrame(
                {
                    "Shipment_ID": prepared["Shipment_ID"],
                    "h_n": prepared["h_n"],
                }
            )

        with (
            patch.object(pipeline, "attach_static", side_effect=lambda orders, _: orders),
            patch.object(
                pipeline, "_predict_prepared_orders", side_effect=fake_predict
            ),
        ):
            predictions = pipeline.sequential_historical_replay(
                replay, base_history, pd.DataFrame(), {}
            )

        self.assertEqual(predictions["Shipment_ID"].tolist(), ["A", "B", "C"])
        self.assertEqual(observed_prior_counts, [1, 1, 3])

    def test_stockout_probability_is_bounded_by_late_probability(self):
        predicted = pipeline.stockout_probabilities(
            ConstantProbabilityModel([0.8, 0.2]),
            pd.DataFrame(index=[0, 1]),
            [0.4, 0.7],
        )
        np.testing.assert_allclose(predicted, [0.4, 0.2])
        self.assertTrue(np.all((predicted >= 0) & (predicted <= 1)))

    def test_reliability_bins_compare_predictions_and_observed_rates(self):
        table = pipeline.stockout_reliability_table(
            [0, 1, 1, 1],
            [0.05, 0.15, 0.75, 0.85],
            bins=2,
        )
        self.assertEqual(table["shipment_count"].tolist(), [2, 2])
        self.assertEqual(
            table["observed_stockout_frequency"].tolist(), [0.5, 1.0]
        )
        calibration = pipeline.stockout_calibration_metrics(table)
        self.assertAlmostEqual(calibration["expected_calibration_error"], 0.3)
        self.assertAlmostEqual(calibration["max_calibration_error"], 0.4)

    def test_sklearn_fallback_selects_iteration_using_validation(self):
        X_train = pd.DataFrame({"x": np.arange(100, dtype=float)})
        y_train = (X_train["x"] > 50).astype(int)
        X_valid = pd.DataFrame({"x": np.arange(100, 120, dtype=float)})
        y_valid = (X_valid["x"] > 110).astype(int)
        model = HistGradientBoostingClassifier(
            max_iter=30,
            learning_rate=0.1,
            random_state=7,
        )
        best_iteration = pipeline.fit_with_early_stopping(
            model, X_train, y_train, X_valid, y_valid
        )
        self.assertGreaterEqual(best_iteration, 10)
        self.assertLessEqual(best_iteration, 30)
        self.assertEqual(model.n_iter_, best_iteration)
        self.assertFalse(model.early_stopping)

    def test_missing_current_shipping_cost_prevents_savings(self):
        order = pd.Series(
            {
                "Supplier_ID": "CURRENT",
                "Item_ID": "PART",
                "Quantity_Ordered": 10,
                "Stock_On_Hand_Days": 2,
                "Line_Loss_Per_Day_INR": 100.0,
                "Order_Date": pd.Timestamp("2026-01-01"),
                "Promised_Date": pd.Timestamp("2026-01-06"),
                "Unit_Cost_INR": 50.0,
                "exp_loss_inr": 25.0,
            }
        )
        offers = pd.DataFrame(
            [
                {
                    "Supplier_ID": "CURRENT",
                    "Item_ID": "PART",
                    "Shipping_Cost_INR": np.nan,
                },
                {
                    "Supplier_ID": "ALT",
                    "Item_ID": "PART",
                    "Shipping_Cost_INR": 1.0,
                    "Supplier_Status": "Active",
                    "Available_Quantity": 100,
                    "Minimum_Order_Quantity": 1,
                    "Maximum_Supply_Quantity": 100,
                    "Quality_Rating": 5.0,
                    "Lead_Time_Days": 2,
                },
            ]
        )
        current_cost, recommendations = pipeline.recommend(
            order, offers, pd.DataFrame(), {}, as_of=order.Order_Date
        )
        self.assertTrue(np.isnan(current_cost))
        self.assertTrue(recommendations.empty)
        self.assertEqual(
            recommendations.attrs["status"],
            "missing_current_supplier_shipping_cost",
        )

    def test_missing_alternative_shipping_cost_is_not_ranked(self):
        order = pd.Series(
            {
                "Shipment_ID": "ORDER",
                "Supplier_ID": "CURRENT",
                "Item_ID": "PART",
                "Quantity_Ordered": 10,
                "Stock_On_Hand_Days": 2,
                "Line_Loss_Per_Day_INR": 100.0,
                "Order_Date": pd.Timestamp("2026-01-01"),
                "Promised_Date": pd.Timestamp("2026-01-06"),
                "Unit_Cost_INR": 50.0,
                "exp_loss_inr": 25.0,
            }
        )
        offers = pd.DataFrame(
            [
                {
                    "Supplier_ID": "CURRENT",
                    "Item_ID": "PART",
                    "Shipping_Cost_INR": 5.0,
                },
                {
                    "Supplier_ID": "ALT",
                    "Item_ID": "PART",
                    "Shipping_Cost_INR": np.nan,
                    "Supplier_Status": "Active",
                    "Available_Quantity": 100,
                    "Minimum_Order_Quantity": 1,
                    "Maximum_Supply_Quantity": 100,
                    "Quality_Rating": 5.0,
                    "Lead_Time_Days": 2,
                },
            ]
        )
        current_cost, recommendations = pipeline.recommend(
            order, offers, pd.DataFrame(), {}, as_of=order.Order_Date
        )
        self.assertEqual(current_cost, 530.0)
        self.assertTrue(recommendations.empty)
        self.assertEqual(
            recommendations.attrs["status"],
            "alternatives_missing_shipping_cost",
        )


if __name__ == "__main__":
    unittest.main()
