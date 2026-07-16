import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

from features import load_raw_data
import preprocess
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def evaluate_by_price_range(df, y_test, y_pred_log):
    """Evaluate model performance across different price ranges"""
    
    # Convert back from log to real AED
    y_actual = np.expm1(y_test)
    y_predicted = np.expm1(y_pred_log)
    
    # Define price buckets
    buckets = [0, 50000, 150000, 400000, np.inf]
    bucket_labels = ['0-50K', '50K-150K', '150K-400K', '400K+']
    
    print("\n" + "="*70)
    print("EVALUATION BY PRICE RANGE")
    print("="*70)
    
    for i in range(len(buckets)-1):
        lower, upper = buckets[i], buckets[i+1]
        
        # Filter cars in this price range
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

if __name__ == "__main__":
    # Load and prepare data
    df = load_raw_data("data/raw/uae_used_cars_10k.csv")
    df = preprocess(df)
    df = df[df["Price"] <= 400000].copy()
    
    # Train model (same as train.py)
    FEATURES = [
        "Car_Age",
        "Log_Mileage",
        "Cylinders",
        "Make_Encoded_Log",
        "Model_Encoded_Log",
        "Body Type_Encoded_Log"
    ]
    
    X = df[FEATURES]
    y = df["Log_Price"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    
    # Evaluate
    evaluate_by_price_range(df, y_test, y_pred_log)