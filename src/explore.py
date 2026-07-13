import sys
sys.path.insert(0, ".")
from src.features import load_raw_data

df = load_raw_data("data/raw/uae_used_cars_10k.csv")   
print("Shape", df.shape)
print("Data Type: ", df.dtypes)
print("Any Missing value", df.isna().sum())