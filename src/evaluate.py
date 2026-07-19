import sys
import os
sys.path.insert(0, ".")

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.features import load_raw_data
from src.preprocess import clean_data, fit_target_encoding, apply_target_encoding

# ── Configuration (mirrors train.py so results are comparable) ──
DATA_PATH = os.environ.get("DATA_PATH", "data/raw/uae_used_cars_10k.csv")

FEATURES = [
    "Car_Age",
    "Log_Mileage",
    "Cylinders",
    "Make_Encoded_Log",
    "Model_Encoded_Log",
    "Body Type_Encoded_Log"
]

TARGET = "Log_Price"
PRICE_CAP = 400000

N_ESTIMATORS = 90
MAX_DEPTH = 9
RANDOM_STATE = 42
TEST_SIZE = 0.2


def evaluate_by_price_range(y_actual, y_predicted):
    """Evaluate model performance across different price ranges.

    y_actual / y_predicted are real AED prices (already reversed from log space),
    aligned positionally (e.g. both numpy arrays or both reset-index Series).
    """
    y_actual = np.asarray(y_actual)
    y_predicted = np.asarray(y_predicted)

    buckets = [0, 50000, 150000, 400000, np.inf]
    bucket_labels = ['0-50K', '50K-150K', '150K-400K', '400K+']

    print("\n" + "=" * 70)
    print("EVALUATION BY PRICE RANGE")
    print("=" * 70)

    for i in range(len(buckets) - 1):
        lower, upper = buckets[i], buckets[i + 1]

        mask = (y_actual >= lower) & (y_actual < upper)
        n_cars = mask.sum()

        if n_cars == 0:
            continue

        actual_range = y_actual[mask]
        pred_range = y_predicted[mask]

        mae = mean_absolute_error(actual_range, pred_range)
        mape = mean_absolute_percentage_error(actual_range, pred_range)

        print(f"\n{bucket_labels[i]} AED ({n_cars} cars):")
        print(f"  MAE:  {mae:,.0f} AED")
        print(f"  MAPE: {mape:.2%}")
        print(f"  Avg Price: {actual_range.mean():,.0f} AED")
        print(f"  Error as % of avg: {(mae / actual_range.mean() * 100):.1f}%")


def main():
    # 1. Load and clean (no leakage risk in cleaning itself)
    df = load_raw_data(DATA_PATH)
    df = clean_data(df)

    # 2. Filter to standard market (same cap train.py uses)
    df = df[df["Price"] <= PRICE_CAP].copy()
    print(f"✅ Market filtered: {len(df)} cars under {PRICE_CAP:,} AED")

    # 3. Split FIRST - before anything learns from the data
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"✅ Train size: {len(train_df)}, Test size: {len(test_df)}")

    # 4. Fit encoding on TRAIN ONLY, then apply to both (leakage-safe)
    categorical_cols = ["Make", "Model", "Body Type"]
    encoding_mappings = fit_target_encoding(train_df, categorical_cols, alpha=10)

    train_df = apply_target_encoding(train_df, categorical_cols, encoding_mappings)
    test_df = apply_target_encoding(test_df, categorical_cols, encoding_mappings)

    # 5. Assemble X/y from the already-encoded frames
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # 6. Train (same hyperparameters as train.py, for a comparable model)
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)

    # 7. Overall metrics (in real AED, for context before the breakdown)
    real_actuals = np.expm1(y_test)
    real_preds = np.expm1(y_pred_log)

    overall_r2 = r2_score(y_test, y_pred_log)
    overall_mae = mean_absolute_error(real_actuals, real_preds)
    overall_mape = mean_absolute_percentage_error(real_actuals, real_preds)

    print(f"\n── Overall (Test Set) ───────────────────────")
    print(f"✅ R2 Score:  {overall_r2:.4f}")
    print(f"✅ MAE:       {overall_mae:,.0f} AED")
    print(f"✅ MAPE:      {overall_mape:.2%}")

    # 8. Segment-level breakdown
    evaluate_by_price_range(real_actuals, real_preds)


if __name__ == "__main__":
    main()