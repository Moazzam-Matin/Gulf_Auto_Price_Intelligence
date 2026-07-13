import pandas as pd
import numpy as np
from src.features import load_raw_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Safe to run on the full dataset - no target leakage risk here."""

    df["Cylinders"] = df["Cylinders"].replace("Unknown", np.nan)
    df["Cylinders"] = df["Cylinders"].replace("unknown", np.nan)

    df["Cylinders"] = df.groupby("Body Type")["Cylinders"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 6)
    )

    df = df[df["Price"] >= 2000]
    df = df[df["Mileage"] <= 500000]
    df = df.reset_index(drop=True)

    df["Log_Price"] = np.log1p(df["Price"])
    df["Log_Mileage"] = np.log1p(df["Mileage"])
    df["Car_Age"] = (2026 - df["Year"]).apply(lambda x: max(x, 1))
    df["Km_Per_Year"] = df["Mileage"] / df["Car_Age"]

    print(f"✅ Cylinders fixed by Body Type")
    print(f"✅ Outliers removed")
    print(f"✅ Log transforms applied")
    print(f"✅ Car_Age and Km_Per_Year calculated")
    print(f"✅ Clean shape: {df.shape}")

    return df


def fit_target_encoding(train_df: pd.DataFrame, categorical_cols: list, alpha: int = 10) -> dict:
    """LEARN encoding mappings from TRAINING data only. Returns a dict of mappings."""
    global_mean = train_df["Log_Price"].mean()
    mappings = {"global_mean": global_mean}

    for col in categorical_cols:
        agg = train_df.groupby(col)["Log_Price"].agg(["count", "mean"])
        smooth = (agg["count"] * agg["mean"] + alpha * global_mean) / (agg["count"] + alpha)
        mappings[col] = smooth

    return mappings


def apply_target_encoding(df: pd.DataFrame, categorical_cols: list, mappings: dict) -> pd.DataFrame:
    """APPLY already-learned mappings to any dataframe (train or test)."""
    df = df.copy()
    global_mean = mappings["global_mean"]

    for col in categorical_cols:
        smooth = mappings[col]
        df[f"{col}_Encoded_Log"] = df[col].map(smooth).fillna(global_mean)

    return df