from src import ML_Backend
import pandas as pd

df = pd.read_csv("data/train_data.csv", names=["id", "property_description", "area_jhb", "listing_agency", "rental_amount", "street_address", "occupation_date", "Bedrooms_1", "Bedrooms_2", "bathrooms", "area", "Feature_1", "Feature_2", "additional"])
print(df.head())