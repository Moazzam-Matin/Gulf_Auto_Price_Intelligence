import numpy as np
import pandas as pd

FEATURES = [
    "Car_Age",
    "Log_Mileage",
    "Cylinders",
    "Make_Encoded_Log",
    "Model_Encoded_Log",
    "Body Type_Encoded_Log",
]


def predict_price(model, mappings: dict, year: int, mileage: int, make: str,
                   car_model: str, body_type: str, cylinders: float) -> float:

    car_age = max(2026 - year, 1)
    log_mileage = np.log1p(mileage)

    global_mean = mappings["global_mean"]
    make_enc = mappings["Make"].get(make, global_mean)
    model_enc = mappings["Model"].get(car_model, global_mean)
    body_enc = mappings["Body Type"].get(body_type, global_mean)

    features = pd.DataFrame(
        [[car_age, log_mileage, cylinders, make_enc, model_enc, body_enc]],
        columns=FEATURES,
    )

    log_pred = model.predict(features)[0]
    return round(float(np.expm1(log_pred)), -2)