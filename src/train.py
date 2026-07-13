import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)

from src.features import load_raw_data
from src.preprocess import clean_data, fit_target_encoding, apply_target_encoding

# ── Configuration ──────────────────────────────────────────
DATA_PATH = "data/raw/uae_used_cars_10k.csv"

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

# Model parameters
N_ESTIMATORS = 90
MAX_DEPTH = 9
RANDOM_STATE = 42
TEST_SIZE = 0.2


def train():
    # 1. Load and clean (no leakage risk in cleaning)
    df = load_raw_data(DATA_PATH)
    df = clean_data(df)

    # 2. Filter to standard market (remove luxury outliers)
    df = df[df["Price"] <= PRICE_CAP].copy()
    print(f"✅ Market filtered: {len(df)} cars under {PRICE_CAP:,} AED")

    # 3. Split FIRST - before anything learns from the data
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"✅ Train size: {len(train_df)}, Test size: {len(test_df)}")

    # 4. Fit encoding on TRAIN ONLY, then apply to both
    categorical_cols = ["Make", "Model", "Body Type"]
    encoding_mappings = fit_target_encoding(train_df, categorical_cols, alpha=10)

    train_df = apply_target_encoding(train_df, categorical_cols, encoding_mappings)
    test_df = apply_target_encoding(test_df, categorical_cols, encoding_mappings)

    # 5. Now assemble X/y from the already-encoded frames
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # 6. MLflow tracking
    mlflow.set_experiment("gulf-auto-price")

    with mlflow.start_run():

        # Train model
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        # Predict in log space
        preds_log = model.predict(X_test)

        # Convert back to real AED
        real_actuals = np.expm1(y_test)
        real_preds = np.expm1(preds_log)

        # Calculate metrics
        r2 = r2_score(y_test, preds_log)
        mae = mean_absolute_error(real_actuals, real_preds)
        mape = mean_absolute_percentage_error(real_actuals, real_preds)

        # Log parameters to MLflow
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("price_cap", PRICE_CAP)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("encoding_alpha", 10)

        # Log metrics to MLflow
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae_aed", mae)
        mlflow.log_metric("mape", mape)

        # Save the model
        mlflow.sklearn.log_model(model, name="random-forest-model")

        # Also save a plain copy for the API to load directly
        
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("encoding_mappings.pkl", "wb") as f:
            pickle.dump(encoding_mappings, f)
        print("✅ Saved model.pkl and encoding_mappings.pkl for serving")

        print(f"\n── Results ───────────────────────")
        print(f"✅ R2 Score:  {r2:.4f}")
        print(f"✅ MAE:       {mae:,.0f} AED")
        print(f"✅ MAPE:      {mape:.2%}")
        print(f"✅ Model saved to MLflow!")


if __name__ == "__main__":
    train()