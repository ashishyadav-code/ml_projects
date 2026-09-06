from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from src.components.data_transformation import main as data_main
from src.utils import eval_model,save_object
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import os

def main():
    train_arr,test_arr=data_main()
    X_train = train_arr[:,:-1]
    X_test = test_arr[:,:-1]

    y_train = train_arr[:,-1]
    y_test = test_arr[:,-1]

    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(class_weight="balanced", random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
        "DecisionTreeClassifier": DecisionTreeClassifier(class_weight="balanced", random_state=42)
    }

    reports = eval_model(X_train, y_train, X_test, y_test, models)
    return reports

if __name__ =="__main__":
    reports = main()

    for model_name, report in reports.items():
        print(f"\n{model_name}")
        print(f"Train AUC: {report['Train_ROC_AUC']}")
        print(f"Test AUC: {report['Test_ROC_AUC']}")
        print(f"Recall: {report['Recall']}")
        print(f"Precision: {report['Precision']}")
        print(f"F1: {report['F1']}")

    best_model_name = max(reports,key=lambda x: reports[x]["F1"])
    best_model = reports[best_model_name]["Model"]
    best_score = reports[best_model_name]["F1"]
    print(f"\n🏆 Best Model: {best_model_name} with Test F1 Score: {best_score:.4f}")

    os.makedirs("Artifacts",exist_ok=True)
    save_object("Artifacts/model.pkl",best_model)