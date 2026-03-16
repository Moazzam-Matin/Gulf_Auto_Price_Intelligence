import numpy as np
import pandas as pd

def predict_car_price(model, make_mapping, model_mapping, body_mapping, input_details):
 
    try:
        # 1. Extract inputs
        year = input_details['year']
        mileage = input_details['mileage']
        make = input_details['make']
        car_model = input_details['model']
        body_type = input_details['body_type']
        cylinders = input_details['cylinders']

        # 2. Feature Engineering 
        age = 2026 - year # Using current year
        log_mileage = np.log1p(mileage)

        # 3. Apply the Target Encoding logic (using the mappings from training)
        # We use .get() to handle car models the model hasn't seen before
        make_enc = make_mapping.get(make, make_mapping.mean())
        model_enc = model_mapping.get(car_model, model_mapping.mean())
        body_enc = body_mapping.get(body_type, body_mapping.mean())

        # 4. Create the Input Vector (Must match the 6 features we settled on)
        features = pd.DataFrame([[
            age, 
            log_mileage, 
            cylinders, 
            make_enc, 
            model_enc, 
            body_enc
        ]], columns=['Age', 'Log_Mileage', 'Cylinders', 'Make_Encoded_Log', 
                     'Model_Encoded_Log', 'Body Type_Encoded_Log'])

        # 5. Predict and Reverse the Log-Price
        log_pred = model.predict(features)[0]
        final_price = np.expm1(log_pred)

        return round(final_price, -2) # Round to nearest 100 for a 'cleaner' look

    except Exception as e:
        return f"Error in prediction: {str(e)}"