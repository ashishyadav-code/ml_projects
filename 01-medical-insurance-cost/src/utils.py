from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def eval_model(X_train, y_train, X_test, y_test, models: dict):
    """
    Evaluate multiple regression models using R2, MSE, and MAE.

    Parameters:
    X_train : Training features
    y_train : Training target values (log-transformed)
    X_test : Testing features
    y_test : Testing target values (log-transformed)
    models : Dictionary containing model names and model objects

    Returns:
    report : Dictionary containing trained models and their evaluation metrics.
    """
    try:
        report = {}

        for model_name, model in models.items():

            # Train model on log-transformed target
            model.fit(X_train, y_train)

            # Predictions are also in log scale
            y_train_pred_log = model.predict(X_train)
            y_test_pred_log = model.predict(X_test)

            # Convert back to original charges
            y_train_original = np.expm1(y_train)
            y_test_original = np.expm1(y_test)

            y_train_pred_original = np.expm1(y_train_pred_log)
            y_test_pred_original = np.expm1(y_test_pred_log)

            # Metrics on original charges
            train_r2 = r2_score(
                y_train_original,
                y_train_pred_original
            )

            test_r2 = r2_score(
                y_test_original,
                y_test_pred_original
            )

            test_mse = mean_squared_error(
                y_test_original,
                y_test_pred_original
            )

            test_mae = mean_absolute_error(
                y_test_original,
                y_test_pred_original
            )

            report[model_name] = {
                "Model": model,
                "TRAIN_R2_SCORE": train_r2,
                "TEST_R2_SCORE": test_r2,
                "MSE": test_mse,
                "MAE": test_mae
            }

        return report

    except Exception as e:
        return e