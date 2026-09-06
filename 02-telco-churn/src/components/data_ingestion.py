import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os

def load_data(file_path:str)->pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def split_data(df:pd.DataFrame):
    """
    Takes a dataset as pandas dataframe.
    splits the dataset into train and test data.

    Args:
        df:pd.DataFrame
    """
    train_data,test_data = train_test_split(
        df,
        stratify=df['Churn'],
        test_size=0.2,
        random_state=42
    )
    return train_data,test_data

def save_data(train_data,test_data,output_dir:str):
    os.makedirs(output_dir,exist_ok=True)

    train_path = os.path.join(output_dir,"train.csv")
    test_path = os.path.join(output_dir,"test.csv")

    train_data.to_csv(train_path,index=False)
    test_data.to_csv(test_path,index=False)

    print(f"Train data saved at: {train_path}")
    print(f"Test data saved at: {test_path}")

    return train_path,test_path

if __name__=="__main__":
    file_path = "Data/Raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_dir = "Data/Processed"

    df = load_data(file_path)
    train_data,test_data = split_data(df)
    save_data(train_data,test_data,output_dir)
