import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_price

app = FastAPI(title="Gulf Auto Price API")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoding_mappings.pkl", "rb") as f:
    mappings = pickle.load(f)


class CarDetails(BaseModel):
    year: int = Field(..., ge=1990, le=2026, description="Manufacture year")
    mileage: int = Field(..., ge=0, description="Mileage in km")
    make: str
    model: str
    body_type: str
    cylinders: float = Field(..., ge=1, le=16)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(car: CarDetails):
    price = predict_price(
        model=model,
        mappings=mappings,
        year=car.year,
        mileage=car.mileage,
        make=car.make,
        car_model=car.model,
        body_type=car.body_type,
        cylinders=car.cylinders,
    )
    return {"predicted_price_aed": price}