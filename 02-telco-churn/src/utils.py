import os
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score


def save_object(file_path: str, obj):
    """Saves a python object to a pickle file."""
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def eval_model(X_train, y_train, X_test, y_test, models: dict):
    """
    Evaluate multiple classification models on ROC-AUC, Recall, Precision, and F1.
    """
    try:
        report = {}

        for model_name, model in models.items():
            # 1. Model train karo
            model.fit(X_train, y_train)

            # 2. Discrete class predictions (0 or 1) for Precision, Recall, F1
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # 3. Probabilities for ROC-AUC
            if hasattr(model, "predict_proba"):
                y_train_proba = model.predict_proba(X_train)[:, 1]
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_train_proba = y_train_pred
                y_test_proba = y_test_pred

            # 4. Calculate classification metrics
            train_auc = roc_auc_score(y_train, y_train_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)
            test_recall = recall_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred)

            report[model_name] = {
                "Model": model,
                "Train_ROC_AUC": train_auc,
                "Test_ROC_AUC": test_auc,
                "Recall": test_recall,
                "Precision": test_precision,
                "F1": test_f1
            }

        return report

    except Exception as e:
        raise e