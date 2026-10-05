import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data(file_path:str)->pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def split_data(df:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    """
    Splits data into train and test data.

    Args:
        df (pd.Dataframe): Dataframe for performing the split
    """
    train_data,test_data = train_test_split(
        df,
        random_state=42,
        test_size=0.2
    )
    return train_data,test_data

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,output_dir:str):
    os.makedirs(output_dir,exist_ok=True)

    train_path = os.path.join(output_dir,"train.csv")
    test_path = os.path.join(output_dir,"test.csv")

    train_data.to_csv(train_path,index=False)
    test_data.to_csv(test_path,index=False)

    print(f"Train data saved at {train_path}")
    print(f"Test data saved at {test_path}")

if __name__=="__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
    file_path = os.path.join(BASE_DIR, "Data", "Raw", "Car details v3.csv")
    output_dir = os.path.join(BASE_DIR, "Data", "Processed")
    df = load_data(file_path)
    train_data,test_data=split_data(df)
    save_data(train_data,test_data,output_dir)