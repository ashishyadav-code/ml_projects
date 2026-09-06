import os
import sys
import pandas as pd
import pickle


class PredictPipeline:
    """
    Loads saved preprocessor and model artifacts to produce churn predictions
    and probability confidence scores.
    """
    def __init__(self):
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
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

            data_scaled = preprocessor.transform(features)

            # Class prediction (0 = Stay, 1 = Churn)
            prediction = model.predict(data_scaled)

            # Probability score (0.0 to 1.0 risk of churning)
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(data_scaled)[:, 1]
            else:
                probability = prediction

            return prediction, probability

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e


class CustomData:
    """
    Encapsulates all 19 raw customer features entered from the UI
    and converts them into a DataFrame formatted for the preprocessor.
    """
    def __init__(
        self,
        gender: str,
        SeniorCitizen: int,
        Partner: str,
        Dependents: str,
        tenure: int,
        PhoneService: str,
        MultipleLines: str,
        InternetService: str,
        OnlineSecurity: str,
        OnlineBackup: str,
        DeviceProtection: str,
        TechSupport: str,
        StreamingTV: str,
        StreamingMovies: str,
        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str,
        MonthlyCharges: float,
        TotalCharges: float,
    ):
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Converts the instance attributes to a pandas DataFrame matching raw dataset columns.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise RuntimeError(f"Failed to convert custom data to DataFrame: {e}") from e


if __name__ == "__main__":
    # Quick sanity test
    test_customer = CustomData(
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=1,
        PhoneService="No",
        MultipleLines="No phone service",
        InternetService="DSL",
        OnlineSecurity="No",
        OnlineBackup="Yes",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=29.85,
        TotalCharges=29.85,
    )

    df_test = test_customer.get_data_as_data_frame()
    print("Test Customer Input DataFrame:\n", df_test)

    pipeline = PredictPipeline()
    pred, proba = pipeline.predict(df_test)
    print(f"\nPrediction: {'CHURN (Leave)' if pred[0] == 1 else 'STAY (Active)'}")
    print(f"Churn Risk Probability: {proba[0] * 100:.2f}%")
