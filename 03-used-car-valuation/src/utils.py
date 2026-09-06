import os
import sys
import pickle
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.exception import CustomException


def save_object(file_path: str, obj):
    """
    Saves any Python object into a pickle file safely.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Loads a pickled Python object safely.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
