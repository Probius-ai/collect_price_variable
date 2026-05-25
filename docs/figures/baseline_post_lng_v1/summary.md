# Baseline snapshot — `baseline_post_lng_v1`

Captured 2026-05-25.

This snapshot freezes the **current monthly SMP forecast results** before
any LNG-price-forecast integration. The intended use is before/after
comparison: re-run `save_baseline_plots --tag baseline_post_lng_v1` (or
similar) after the LNG model lands, then diff the two folders.

## Forecast contract in effect

* forecast_origin_month = period_month
* information_cutoff    = end-of-day of forecast_origin_month
* target_month          = forecast_origin_month + 1 month
* horizon               = 1M

The persistence baseline (`persistence_monthly`) predicts
`SMP(M+1) = SMP(M)`. On any plot indexed by forecast_origin_month, its
line **trails the actual by one month** — that is the lag this snapshot
documents and which a leading-indicator model is expected to compress.

## Model classification

* Baselines: naive, naive_lag_1m, persistence_monthly, seasonal_naive, seasonal_naive_lag_12m
* Trainable: delta_ar_ridge, delta_lightgbm, delta_ridge, lightgbm, monthly_ar_ridge, ridge

## Test-split metrics (sorted by MAE)

| model | kind | MAE | MAPE [%] | R² | delta_MAE | delta_dir_acc |
|---|---|---|---|---|---|---|
| persistence_monthly | baseline | 6.427 | 5.464 | 0.556 | 6.427 | 0.000 |
| delta_ar_ridge | trainable | 6.594 | 5.466 | 0.464 | 6.594 | 0.500 |
| monthly_ar_ridge | trainable | 7.445 | 6.224 | 0.424 | 7.445 | 0.500 |
| delta_ridge | trainable | 7.724 | 6.576 | 0.369 | 7.724 | 0.308 |
| ridge | trainable | 8.853 | 7.535 | 0.260 | 8.853 | 0.385 |
| lightgbm | trainable | 9.163 | 7.861 | 0.209 | 9.163 | 0.615 |
| naive_lag_1m | baseline | 9.397 | 8.213 | 0.063 | 9.397 | 0.500 |
| delta_lightgbm | trainable | 9.973 | 8.674 | 0.049 | 9.973 | 0.385 |
| seasonal_naive_lag_12m | baseline | 22.201 | 18.688 | -6.607 | 22.201 | 0.500 |

## Files in this snapshot

| file | what it shows |
|---|---|
| plot_01_predictions_test_all_models.png | Actual + top trainable + persistence overlay (test split) |
| plot_02_persistence_lag_zoom.png        | The one-month lag of persistence visualised explicitly |
| plot_03_residuals_per_model.png         | Per-model residual (pred − actual) over time |
| plot_04_test_mae_r2_comparison.png      | Bar chart of MAE / R² by model |
| plot_05_walk_forward_long_range.png     | Walk-forward CV over the full 150-month eval window |
| plot_06_feature_group_ablation_heatmap.png | Feature-group × model MAE heatmap |
| predictions_test_<m>.csv                | Raw test-split predictions per spotlight model |
| walk_forward_predictions_<m>.csv        | Raw walk-forward predictions per spotlight model |
| metrics_comparison_snapshot.csv         | Copy of `outputs/metrics/comparison.csv` |
| walk_forward_comparison_snapshot.csv    | Copy of `outputs/walk_forward/comparison.csv` |
| feature_group_ablation_snapshot.csv     | Copy of the ablation result table |

## Reproduce a comparable snapshot

```bash
# 1. rebuild monthly features (or run the file pipeline)
python -m src.pipelines.build_monthly_features --area mainland --horizon-months 1

# 2. retrain all models you want spotlighted
F=data/processed/smp_monthly_mainland_h1m.parquet
for M in persistence_monthly monthly_ar_ridge delta_ar_ridge delta_ridge ridge; do
  python -m src.pipelines.train --features-path $F --model $M \
      --target target_smp_t_plus_h_months --timestamp-col period_month \
      --min-train-rows 24
done

# 3. (optional) walk-forward + ablation for plots 5/6
for M in persistence_monthly monthly_ar_ridge delta_ar_ridge delta_ridge ridge; do
  python -m src.pipelines.walk_forward main --features-path $F --model $M --min-train-rows 24
done
python -m src.pipelines.walk_forward compare
python -m src.pipelines.feature_group_ablation --features-path $F

# 4. snapshot under a new tag
python -m src.pipelines.evaluate
python -m src.pipelines.save_baseline_plots --tag baseline_post_lng_v1
```
