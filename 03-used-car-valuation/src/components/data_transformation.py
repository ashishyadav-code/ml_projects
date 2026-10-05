import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils import save_object

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'mileage' in df.columns:
        df['mileage_kmpl'] = pd.to_numeric(
            df['mileage'].str.extract(r'([0-9.]+)', expand=False), 
            errors='coerce'
        )
    if 'engine' in df.columns:
        df['engine_cc'] = pd.to_numeric(
            df['engine'].str.extract(r'([0-9.]+)', expand=False), 
            errors='coerce'
        )
    if 'max_power' in df.columns:
        df['max_power_bhp'] = pd.to_numeric(
            df['max_power'].str.extract(r'([0-9.]+)', expand=False), 
            errors='coerce'
        )
    if 'name' in df.columns:
        df['brand'] = df['name'].apply(lambda x: str(x).split()[0].lower().strip())

    if 'year' in df.columns:
        df['car_age'] = 2024 - df['year']
        if 'km_driven' in df.columns:
            df['km_per_year'] = df['km_driven'] / (df['car_age'] + 0.5)
        df = df.drop(columns=['year'])

    cols_to_drop = ['name', 'mileage', 'engine', 'max_power', 'torque']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    return df

def get_num_cat_col(df: pd.DataFrame):
    if 'selling_price' in df.columns:
        df = df.drop(columns=['selling_price'])
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
    return numerical_columns, categorical_columns

def get_data_transformation_object(categorical_columns, numerical_columns):
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
            ("num_pipeline", num_pipeline, numerical_columns),
            ("cat_pipeline", cat_pipeline, categorical_columns)
        ]
    )
    return preprocessor

def initiate_data_transformation(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    X_train = train_df.drop(columns=['selling_price'])
    X_test = test_df.drop(columns=['selling_price'])

    y_train = np.log1p(train_df['selling_price'])
    y_test = np.log1p(test_df['selling_price'])

    numerical_cols, categorical_cols = get_num_cat_col(train_df)
    preprocessor = get_data_transformation_object(categorical_cols, numerical_cols)

    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    train_arr = np.c_[X_train_arr, np.array(y_train)]
    test_arr = np.c_[X_test_arr, np.array(y_test)]

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    artifacts_dir = os.path.join(base_dir, "Artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    save_object(preprocessor_path, preprocessor)

    return train_arr, test_arr, preprocessor_path

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    train_path = os.path.join(base_dir, "Data", "Processed", "train.csv")
    test_path = os.path.join(base_dir, "Data", "Processed", "test.csv")
    train_arr, test_arr, pkl_path = initiate_data_transformation(train_path, test_path)
    print(f"Train Array Shape: {train_arr.shape}")
    print(f"Test Array Shape: {test_arr.shape}")
    print(f"Preprocessor saved at: {pkl_path}")
    return train_arr, test_arr

if __name__ == "__main__":
    main()