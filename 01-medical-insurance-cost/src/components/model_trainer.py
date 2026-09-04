from src.utils import eval_model
from src.components.data_transformation import main as data_transformation_main
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import os
import pickle

def main():
    train_arr,test_arr,_ = data_transformation_main()

    models = {
        "Linear regression":LinearRegression(),
        "Ridge":Ridge(),
        "Lasso":Lasso(),
        "RandomForestRegressor":RandomForestRegressor(),
        "XGBRegressor":XGBRegressor()
    }

    X_train = train_arr[:,:-1]
    y_train = train_arr[:,-1]

    X_test = test_arr[:,:-1]
    y_test = test_arr[:,-1]

    reports = eval_model(X_train,y_train,X_test,y_test,models)
    return reports

if __name__=="__main__":
    reports = main()

    for model_name, report in reports.items():
        print(f"\n{model_name}")
        print(f"Train R2: {report['TRAIN_R2_SCORE']}")
        print(f"Test R2: {report['TEST_R2_SCORE']}")
        print(f"MSE: {report['MSE']}")
        print(f"MAE: {report['MAE']}")

    best_model_name = max(reports,key=lambda x: reports[x]["TEST_R2_SCORE"])
    best_model = reports[best_model_name]["Model"]
    best_score = reports[best_model_name]["TEST_R2_SCORE"]
    print(f"\n🏆 Best Model: {best_model_name} with Test R2 Score: {best_score:.4f}")

    os.makedirs("Artifacts",exist_ok=True)
    with open("Artifacts/model.pkl","wb") as f:
        pickle.dump(best_model,f)
    
