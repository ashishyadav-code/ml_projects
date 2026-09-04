import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def splitting_data(df:pd.DataFrame):

    # Splitting Data
    train_data, test_data = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    return train_data,test_data

def save_data(train_data,test_data,output_dir:str):
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir,"train.csv")
    test_path = os.path.join(output_dir,"test.csv")

    train_data.to_csv(train_path,index=False)
    test_data.to_csv(test_path,index=False)

    print(f"Train data saved at: {train_path}")
    print(f"Test data saved at: {test_path}")

    return train_path, test_path

if __name__ == "__main__":
    file_path = "../../Data/Raw_data.csv"
    output_dir = "../../Data/Processed"
    df = load_data(file_path)
    train_data, test_data = splitting_data(df)
    save_data(train_data,test_data,output_dir)
