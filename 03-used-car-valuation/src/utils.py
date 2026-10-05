import os
import sys
import pickle
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.exception import CustomException


def save_object(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def eval_model(X_train, y_train, X_test, y_test, models: dict):
    try:
        report = {}

        y_train_orig = np.expm1(y_train)
        y_test_orig = np.expm1(y_test)

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            y_train_pred_log = model.predict(X_train)
            y_test_pred_log = model.predict(X_test)

            y_train_pred_orig = np.expm1(y_train_pred_log)
            y_test_pred_orig = np.expm1(y_test_pred_log)

            y_test_pred_orig = np.clip(y_test_pred_orig, a_min=10000, a_max=None)

            train_r2 = r2_score(y_train_orig, y_train_pred_orig)
            test_r2 = r2_score(y_test_orig, y_test_pred_orig)
            test_mae = mean_absolute_error(y_test_orig, y_test_pred_orig)
            test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred_orig))
            test_mape = mean_absolute_percentage_error(y_test_orig, y_test_pred_orig) * 100

            report[model_name] = {
                "Model": model,
                "TRAIN_R2_SCORE": train_r2,
                "TEST_R2_SCORE": test_r2,
                "MAE": test_mae,
                "RMSE": test_rmse,
                "MAPE_PCT": test_mape
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)