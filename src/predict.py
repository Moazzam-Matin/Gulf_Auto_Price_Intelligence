import numpy as np
import mlflow.sklearn

def predict_car_price(car_details):
    """
    Predict price for a car given its engineered features
    
    car_details: dict with preprocessed features
    Returns: predicted price in AED
    """
    
    # Load best model from MLflow (Run ID: 73a16f44d0c9412c8af27dbc78e1a11b)
    model_uri = "runs:/73a16f44d0c9412c8af27dbc78e1a11b/random-forest-model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Extract features in the correct order
    features = np.array([[
        car_details['Car_Age'],
        car_details['Log_Mileage'],
        car_details['Cylinders'],
        car_details['Make_Encoded_Log'],
        car_details['Model_Encoded_Log'],
        car_details['Body Type_Encoded_Log']
    ]])
    
    # Predict in log space
    log_price = model.predict(features)[0]
    
    # Convert back to real AED
    predicted_price = np.expm1(log_price)
    
    return predicted_price


if __name__ == "__main__":
    # Example: Predict price for a Toyota Camry, 6 years old, 120K km
    example_car = {
        'Car_Age': 6,
        'Log_Mileage': 11.70,  # log(120000)
        'Cylinders': 4,
        'Make_Encoded_Log': 11.32,  # Toyota average from encoding
        'Model_Encoded_Log': 10.82,  # Camry average from encoding
        'Body Type_Encoded_Log': 11.0  # Sedan average from encoding
    }
    
    price = predict_car_price(example_car)
    print(f"✅ Predicted Car Price: {price:,.0f} AED")
    print(f"\nModel Used:")
    print(f"  Run ID: 73a16f44d0c9412c8af27dbc78e1a11b")
    print(f"  n_estimators: 100")
    print(f"  max_depth: 10")
    print(f"  R2 Score: 0.5206")