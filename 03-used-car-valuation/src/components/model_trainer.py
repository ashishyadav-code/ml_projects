import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.components.data_transformation import main as data_main
from src.utils import eval_model, save_object

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

def main():
    train_arr, test_arr = data_main()
    X_train = train_arr[:, :-1]
    X_test = test_arr[:, :-1]

    y_train = train_arr[:, -1]
    y_test = test_arr[:, -1]

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        "AdaBoostRegressor": AdaBoostRegressor(random_state=42)
    }

    reports = eval_model(X_train, y_train, X_test, y_test, models)
    return reports


if __name__ == "__main__":
    reports = main()

    for model_name, report in reports.items():
        print(f"\n==============================")
        print(f"Model: {model_name}")
        print(f"Train R2: {report['TRAIN_R2_SCORE']:.4f}")
        print(f"Test R2:  {report['TEST_R2_SCORE']:.4f}")
        print(f"MAE:      INR {report['MAE']:,.2f}")
        print(f"RMSE:     INR {report['RMSE']:,.2f}")
        print(f"MAPE:     {report['MAPE_PCT']:.2f}%")

    best_model_name = max(reports, key=lambda x: reports[x]["TEST_R2_SCORE"])
    best_model = reports[best_model_name]["Model"]
    best_score = reports[best_model_name]["TEST_R2_SCORE"]
    print(f"\nBest Model: {best_model_name} with Test R2: {best_score:.4f}")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    artifact_path = os.path.join(base_dir, "Artifacts", "model.pkl")
    save_object(artifact_path, best_model)
    print(f"Model saved successfully at: {artifact_path}")