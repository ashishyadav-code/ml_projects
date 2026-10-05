import os
import sys
import pandas as pd
import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.components.data_transformation import clean_data


class PredictPipeline:
    def __init__(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.model_path = os.path.join(base_dir, "Artifacts", "model.pkl")
        self.preprocessor_path = os.path.join(base_dir, "Artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at: {self.preprocessor_path}")

            with open(self.preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)

            with open(self.model_path, "rb") as f:
                model = pickle.load(f)

            cleaned_features = clean_data(features)
            data_scaled = preprocessor.transform(cleaned_features)

            log_preds = model.predict(data_scaled)
            rupee_preds = np.expm1(log_preds)
            rupee_preds = np.clip(rupee_preds, a_min=15000, a_max=None)

            return rupee_preds

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e


class CustomData:
    def __init__(
        self,
        name: str,
        year: int,
        km_driven: int,
        fuel: str,
        seller_type: str,
        transmission: str,
        owner: str,
        mileage: str,
        engine: str,
        max_power: str,
        seats: float,
    ):
        self.name = name
        self.year = year
        self.km_driven = km_driven
        self.fuel = fuel
        self.seller_type = seller_type
        self.transmission = transmission
        self.owner = owner
        self.mileage = mileage
        self.engine = engine
        self.max_power = max_power
        self.seats = seats

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            data_dict = {
                "name": [self.name],
                "year": [self.year],
                "km_driven": [self.km_driven],
                "fuel": [self.fuel],
                "seller_type": [self.seller_type],
                "transmission": [self.transmission],
                "owner": [self.owner],
                "mileage": [self.mileage],
                "engine": [self.engine],
                "max_power": [self.max_power],
                "seats": [self.seats],
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to create DataFrame: {e}") from e


if __name__ == "__main__":
    sample = CustomData(
        name="Maruti Swift Dzire VDI",
        year=2014,
        km_driven=145500,
        fuel="Diesel",
        seller_type="Individual",
        transmission="Manual",
        owner="First Owner",
        mileage="23.4 kmpl",
        engine="1248 CC",
        max_power="74 bhp",
        seats=5.0
    )

    df_test = sample.get_data_as_data_frame()
    pipeline = PredictPipeline()
    estimated_price = pipeline.predict(df_test)
    print(f"Sample Car: Maruti Swift Dzire VDI (2014)")
    print(f"Estimated Market Valuation: INR {estimated_price[0]:,.2f}")
