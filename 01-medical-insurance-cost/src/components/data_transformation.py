import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

def get_cat_num_cols(df: pd.DataFrame):
    df = df.drop(columns=['charges'])
    numerical_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns.tolist()
    return categorical_cols, numerical_cols

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

def main():
    train_df = pd.read_csv("Data/Processed/train.csv")
    test_df = pd.read_csv("Data/Processed/test.csv")

    X_train = train_df.drop(columns=["charges"])
    y_train = train_df["charges"]

    X_test = test_df.drop(columns=["charges"])
    y_test = test_df["charges"]

    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    categorical_cols,numerical_cols = get_cat_num_cols(train_df)

    # print(f"Categorical coloumns: {categorical_cols}")
    # print(f"Numerical coloumns: {numerical_cols}")

    preprocessor = get_data_transformer_object(categorical_cols,numerical_cols)
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    train_arr = np.c_[X_train_arr,np.array(y_train)]
    test_arr = np.c_[X_test_arr,np.array(y_test)]

    os.makedirs("Artifacts", exist_ok=True)
    preprocessor_path = "Artifacts/preprocessor.pkl"

    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)

    return train_arr, test_arr, preprocessor_path

if __name__ == "__main__":
    train_arr, test_arr, preprocessor_path = main()
    
    print(f"Transformed Train Shape: {train_arr.shape}")
    print(f"Transformed Test Shape:  {test_arr.shape}")
    print(f"Preprocessor saved at:   {preprocessor_path}")