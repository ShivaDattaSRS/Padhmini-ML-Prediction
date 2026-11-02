# Supplier Delay Prediction: Error Analysis and Business Impact

## Scope and guardrails

- Uses the saved chronological test predictions and the latest sequential replay outputs; no model architecture, features, or ranking formula were changed.
- The replay's 300 shipments are contained in the chronological test period. Distribution tests compare the replay with the non-overlapping remainder of that test period; replay-vs-full-test comparisons are not treated as independent.
- Conditional delay MAE is calculated only for actually late shipments because the regressor estimates delay given late.
- High-error flags use minimum group counts and 90th-percentile screening thresholds. The overlap means repeated supplier flags are candidates for review, not independent proof of persistent error.
- The replay has 0 supplier groups with at least five actual late orders. Therefore it does not support a reliable claim that any supplier has consistently high delay error across both scopes.
- Estimated supplier savings are not realized savings. No switched-order counterfactual outcomes are available.

## Aggregate financial error

| Scope | Shipments | Actual loss (INR) | Predicted quantile loss (INR) | Underestimation (INR) | Overestimation (INR) | MAE (INR) | RMSE (INR) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chronological test | 19,989 | 4,334,555,000 | 4,031,573,617 | 3,299,789,731 | 2,996,808,348 | 315,003 | 865,642 |
| Sequential replay | 300 | 82,219,000 | 73,240,908 | 59,115,154 | 50,137,062 | 364,174 | 889,759 |

Stock-out Brier score: chronological test 0.0756; sequential replay 0.0928.

## Business impact

- High-risk orders identified: 5,243 in the test set and 78 in the latest replay, using the saved validation threshold.
- See `business_impact_summary.csv` for actual late orders identified, missed late orders, precision/recall, recommendation feasibility, estimated potential savings, and the explicit not-estimable realized-savings entry.
- Supplier alternatives remain costed using purchase cost + shipping + quantile-based expected production loss.

## Drift comparison

The comparison uses 19,689 earlier test orders versus 300 recent replay orders. Input distribution drift is distinguished from outcome and model-performance changes.

- unit_cost_inr: reference 2926, recent 3154, p=0.07 (two-sample Kolmogorov-Smirnov).
- actual_stockout_rate: reference 0.0995, recent 0.1267, p=0.119 (two-proportion z test).
- supplier_lead_time_days: reference 15.03, recent 15.54, p=0.256 (two-sample Kolmogorov-Smirnov).
- actual_late_rate: reference 0.162, recent 0.1833, p=0.321 (two-proportion z test).
- order_quantity: reference 415.5, recent 437.2, p=0.688 (two-sample Kolmogorov-Smirnov).

See `recent_data_drift_report.csv` for all distributions, sample counts, tests, and performance metrics.
For the numeric input and binary actual-outcome comparisons above, no test had
`p < 0.05`; this is not evidence that the distributions are identical.

Categorical distribution summaries (descriptive total-variation distance; no significance test because cells are sparse):

- supplier_distribution: total-variation distance 0.017; leading shifts: [{"category": "SUP-02407", "recent_minus_reference_share": 0.006361927980090406}, {"category": "SUP-00589", "recent_minus_reference_share": 0.006311138198994363}, {"category": "SUP-00897", "recent_minus_reference_share": 0.006311138198994363}, {"category": "SUP-01170", "recent_minus_reference_share": 0.006260348417898319}, {"category": "SUP-00002", "recent_minus_reference_share": 0.006260348417898319}, {"category": "SUP-01026", "recent_minus_reference_share": 0.006209558636802275}, {"category": "SUP-01791", "recent_minus_reference_share": 0.006107979074610189}, {"category": "SUP-02453", "recent_minus_reference_share": 0.006107979074610189}, {"category": "SUP-01063", "recent_minus_reference_share": 0.006006399512418102}, {"category": "SUP-00245", "recent_minus_reference_share": 0.006006399512418102}]
- item_distribution: total-variation distance 0.057; leading shifts: [{"category": "AUTO-00397", "recent_minus_reference_share": 0.006514297323378536}, {"category": "AUTO-02130", "recent_minus_reference_share": 0.006514297323378536}, {"category": "AUTO-02481", "recent_minus_reference_share": 0.006361927980090406}, {"category": "AUTO-02043", "recent_minus_reference_share": 0.006361927980090406}, {"category": "AUTO-01118", "recent_minus_reference_share": 0.006361927980090406}, {"category": "AUTO-01956", "recent_minus_reference_share": 0.006311138198994363}, {"category": "AUTO-01958", "recent_minus_reference_share": 0.006311138198994363}, {"category": "AUTO-01128", "recent_minus_reference_share": 0.006260348417898319}, {"category": "AUTO-01141", "recent_minus_reference_share": 0.006260348417898319}, {"category": "AUTO-01108", "recent_minus_reference_share": 0.006209558636802275}]

## Highest error screens

Supplier delay MAE screens (chronological test; at least five actual late orders; top of the test-set 90th percentile):

- SUP-00936: 8.671 (n=9).
- SUP-01557: 8.559 (n=6).
- SUP-01948: 8.518 (n=5).
- SUP-00497: 8.106 (n=5).
- SUP-00795: 7.902 (n=6).

Supplier stock-out false-negative-rate screens (at least three actual stock-outs):

- SUP-00535: 1.000 (n=3).
- SUP-00067: 1.000 (n=3).
- SUP-00400: 1.000 (n=3).
- SUP-02461: 1.000 (n=4).
- SUP-00171: 1.000 (n=4).

Item delay MAE screens:

- AUTO-01931: 10.607 (n=5).
- AUTO-01370: 9.196 (n=5).
- AUTO-01542: 8.005 (n=5).
- AUTO-01430: 6.834 (n=5).
- AUTO-00485: 6.048 (n=6).

Item stock-out false-negative-rate screens:

- AUTO-01997: 1.000 (n=4).
- AUTO-01612: 1.000 (n=3).
- AUTO-02280: 1.000 (n=3).
- AUTO-00791: 1.000 (n=3).
- AUTO-00102: 1.000 (n=3).

## Largest financial-error contributors

These are ranked by the sum of absolute shipment-level loss errors in each
group, not by supplier-caused loss.

Chronological test — suppliers:

- SUP-00885: gross absolute error INR 50,583,071; net error INR -16,307,824 (n=19).
- SUP-00435: gross absolute error INR 39,645,529; net error INR -13,359,564 (n=21).
- SUP-00936: gross absolute error INR 36,077,142; net error INR -8,747,644 (n=17).
- SUP-01557: gross absolute error INR 30,320,626; net error INR -25,769,227 (n=11).
- SUP-00484: gross absolute error INR 25,596,760; net error INR -1,303,323 (n=21).

Chronological test — items:

- AUTO-01772: gross absolute error INR 34,660,831; net error INR 9,695,068 (n=26).
- AUTO-00032: gross absolute error INR 28,908,555; net error INR -5,664,273 (n=12).
- AUTO-01123: gross absolute error INR 27,514,220; net error INR -20,558,268 (n=12).
- AUTO-01931: gross absolute error INR 25,752,756; net error INR -16,735,760 (n=11).
- AUTO-02241: gross absolute error INR 24,483,477; net error INR -6,196,205 (n=13).

Sequential replay — suppliers:

- SUP-01560: gross absolute error INR 8,213,272; net error INR -8,213,272 (n=1).
- SUP-00884: gross absolute error INR 6,502,237; net error INR -6,502,237 (n=1).
- SUP-00020: gross absolute error INR 4,512,437; net error INR -4,512,437 (n=1).
- SUP-01541: gross absolute error INR 3,361,593; net error INR -3,361,593 (n=1).
- SUP-01528: gross absolute error INR 3,247,633; net error INR -3,247,633 (n=1).

Sequential replay — items:

- AUTO-01790: gross absolute error INR 8,278,726; net error INR -8,147,819 (n=2).
- AUTO-00298: gross absolute error INR 6,502,237; net error INR -6,502,237 (n=1).
- AUTO-00513: gross absolute error INR 4,512,437; net error INR -4,512,437 (n=1).
- AUTO-01076: gross absolute error INR 3,361,593; net error INR -3,361,593 (n=1).
- AUTO-00155: gross absolute error INR 3,247,633; net error INR -3,247,633 (n=1).

## Error analysis files

- `supplier_error_analysis.csv`: per-supplier conditional delay MAE and stock-out precision, recall, and false-negative rate, with sample counts for test and replay.
- `item_error_analysis.csv`: equivalent item-level results and minimum-count high-error flags.
- `financial_error_analysis.csv`: supplier/item financial MAE, RMSE, under/overestimation totals, and ranked error contribution.

All metrics are retrospective diagnostics on the supplied simulated/demo data; they are not guarantees of future supplier performance or savings.
