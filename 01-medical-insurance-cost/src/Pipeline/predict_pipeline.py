import os
import sys
import pandas as pd
import numpy as np
import pickle


class PredictPipeline:
    def __init__(self):
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.model_path = os.path.join(base_dir, "Artifacts", "model.pkl")
        self.preprocessor_path = os.path.join(base_dir, "Artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found at: {self.model_path}. Please train the model first."
                )
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(
                    f"Preprocessor file not found at: {self.preprocessor_path}."
                )

            with open(self.preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)

            with open(self.model_path, "rb") as f:
                model = pickle.load(f)

            data_scaled = preprocessor.transform(features)

            log_predicts = model.predict(data_scaled)

            dollor_predicts = np.expm1(log_predicts)

            return dollor_predicts
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e

class CustomData:
    """
    Maps user input values from Web UI into a standardized Pandas DataFrame
    compatible with the preprocessor pipeline.
    """

    def __init__(
        self,
        age: int,
        sex: str,
        bmi: float,
        children: int,
        smoker: str,
        region: str,
    ):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def get_data_as_data_frame(self)->pd.DataFrame:
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "bmi": [self.bmi],
                "children": [self.children],
                "smoker": [self.smoker],
                "region": [self.region],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to create DataFrame from input: {e}") from e

if __name__ =="__main__":
    print("Creating dataset..")
    sample = CustomData(
        age=20,sex="male",bmi=21.0,children=0,smoker="yes",region="southwest"
    )
    df = sample.get_data_as_data_frame()
    print(f"Sample Applicant Input:\n{df}\n")
    print("Creating pipeline and predicting..")
    pipeline = PredictPipeline()
    pred=pipeline.predict(df)

    print(f"Predicted Annual Premium: ${pred[0]:,.2f}")