import pandas as pd
import numpy as np
from features import load_raw_data


def preprocess(df: pd.DataFrame) -> pd.DataFrame:


    # Replace "Unknown" string with actual NaN so fillna can fix it
    df["Cylinders"] = df["Cylinders"].replace("Unknown", np.nan)
    df["Cylinders"] = df["Cylinders"].replace("unknown", np.nan)

    # 1. Fix missing Cylinders by Body Type group
    df["Cylinders"] = df.groupby("Body Type")["Cylinders"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 6)
    )

    # 2. Remove outliers
    df = df[df["Price"] >= 2000]
    df = df[df["Mileage"] <= 500000]
    df = df.reset_index(drop=True)

    # 3. Log transform Price and Mileage
    df["Log_Price"] = np.log1p(df["Price"])
    df["Log_Mileage"] = np.log1p(df["Mileage"])

    # 4. Calculate Age (prevent zero with max 1)
    df["Car_Age"] = (2026 - df["Year"]).apply(lambda x: max(x, 1))

    # 5. Calculate Km Per Year
    df["Km_Per_Year"] = df["Mileage"] / df["Car_Age"]

    # 6. Smoothed Log-Target Encoding
    df = apply_smoothed_encoding(
        df,
        target_col="Price",
        categorical_cols=["Make", "Model", "Body Type"],
        alpha=10
    )

    print(f"✅ Cylinders fixed by Body Type")
    print(f"✅ Outliers removed")
    print(f"✅ Log transforms applied")
    print(f"✅ Car_Age and Km_Per_Year calculated")
    print(f"✅ Encoding applied")
    print(f"✅ Final shape: {df.shape}")

    return df


def apply_smoothed_encoding(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: list,
    alpha: int = 10
) -> pd.DataFrame:

    # Step 1 — Calculate log of price temporarily
    df["temp_log_price"] = np.log1p(df[target_col])

    # Step 2 — Calculate overall market average
    global_mean = df["temp_log_price"].mean()

    # Step 3 — For each categorical column
    for col in categorical_cols:
        # Count and mean per category
        agg = df.groupby(col)["temp_log_price"].agg(["count", "mean"])
        counts = agg["count"]
        means = agg["mean"]

        # Apply smoothing formula
        smooth = (counts * means + alpha * global_mean) / (counts + alpha)

        # Map back to dataframe
        df[f"{col}_Encoded_Log"] = df[col].map(smooth)

    # Step 4 — Remove temporary column
    df.drop(columns=["temp_log_price"], inplace=True)

    return df


if __name__ == "__main__":
    df = load_raw_data("data/raw/uae_used_cars_10k.csv")
    df = preprocess(df)
    print("\n--- Sample Output ---")
    print(df[["Make", "Make_Encoded_Log",
              "Model", "Model_Encoded_Log",
              "Car_Age", "Km_Per_Year"]].head())