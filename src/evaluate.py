import sys
import os
sys.path.insert(0, ".")

import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

from src.features import load_raw_data
from src.preprocess import clean_data, apply_target_encoding

# ── Configuration (mirrors train.py so the split is identical) ──
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
    # Load the SAME model and mappings the API actually serves, instead of
    # training a second, separate copy that can silently drift out of sync
    # with whatever hyperparameters train.py currently uses.
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoding_mappings.pkl", "rb") as f:
        encoding_mappings = pickle.load(f)

    # Rebuild the exact same test split train.py used, so we're evaluating
    # on the same held-out rows the model has never seen during training.
    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    df = df[df["Price"] <= PRICE_CAP].copy()
    print(f"✅ Market filtered: {len(df)} cars under {PRICE_CAP:,} AED")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"✅ Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Apply the ALREADY-LEARNED encoding to the test set (no fitting here -
    # fitting only ever happens once, inside train.py)
    categorical_cols = ["Make", "Model", "Body Type"]
    test_df = apply_target_encoding(test_df, categorical_cols, encoding_mappings)

    X_test, y_test = test_df[FEATURES], test_df[TARGET]
    y_pred_log = model.predict(X_test)

    real_actuals = np.expm1(y_test)
    real_preds = np.expm1(y_pred_log)

    overall_r2 = r2_score(y_test, y_pred_log)
    overall_mae = mean_absolute_error(real_actuals, real_preds)
    overall_mape = mean_absolute_percentage_error(real_actuals, real_preds)

    print(f"\n── Overall (Test Set) ───────────────────────")
    print(f"✅ R2 Score:  {overall_r2:.4f}")
    print(f"✅ MAE:       {overall_mae:,.0f} AED")
    print(f"✅ MAPE:      {overall_mape:.2%}")

    evaluate_by_price_range(real_actuals, real_preds)


if __name__ == "__main__":
    main()