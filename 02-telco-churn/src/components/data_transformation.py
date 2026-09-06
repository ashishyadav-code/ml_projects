import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import save_object
import os

def get_cat_num_cols(df: pd.DataFrame):
    df = df.drop(columns=['Churn'])
    numerical_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    return categorical_cols, numerical_cols

def clean_data(df:pd.DataFrame)->pd.DataFrame:
    if 'customerID' in df.columns: df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
    return df

def get_data_transformer_object(categorical_cols,numerical_cols):
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num_pipeline", num_pipeline, numerical_cols),
            ("cat_pipeline", cat_pipeline, categorical_cols)
        ]
    )
    return preprocessor

def initiate_data_transformation(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    X_train = train_df.drop(columns=['Churn'])
    X_test = test_df.drop(columns=['Churn'])

    y_train = train_df['Churn'].map({'Yes': 1, 'No': 0})
    y_test = test_df['Churn'].map({'Yes': 1, 'No': 0})

    categorical_cols,numerical_cols = get_cat_num_cols(train_df)
    preprocessor = get_data_transformer_object(categorical_cols,numerical_cols)

    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    train_arr = np.c_[X_train_arr,np.array(y_train)]
    test_arr = np.c_[X_test_arr,np.array(y_test)]

    os.makedirs("Artifacts", exist_ok=True)
    preprocessor_path = "Artifacts/preprocessor.pkl"
    save_object(preprocessor_path,preprocessor)

    return train_arr,test_arr,preprocessor_path

def main():
    train_path = "Data/Processed/train.csv"
    test_path = "Data/Processed/test.csv"
    train_arr, test_arr, pkl_path = initiate_data_transformation(train_path, test_path)
    print(f"Train Array Shape: {train_arr.shape}")
    print(f"Test Array Shape: {test_arr.shape}")
    return train_arr,test_arr

if __name__=="__main__":
    train_arr,test_arr = main()