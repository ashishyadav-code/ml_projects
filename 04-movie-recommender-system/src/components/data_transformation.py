import os
import sys
import ast
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.exception import CustomException
from src.logger import logging


def parse_names(obj):
    try:
        names = []
        for item in ast.literal_eval(obj):
            names.append(item["name"])
        return names
    except Exception:
        return []


def parse_top3_cast(obj):
    try:
        cast_list = []
        for count, item in enumerate(ast.literal_eval(obj)):
            if count < 3:
                cast_list.append(item["name"])
            else:
                break
        return cast_list
    except Exception:
        return []


def parse_director(obj):
    try:
        for item in ast.literal_eval(obj):
            if item.get("job") == "Director":
                return [item["name"]]
        return []
    except Exception:
        return []


def collapse_spaces(lst):
    return [str(i).replace(" ", "") for i in lst] if isinstance(lst, list) else []


def transform_movie_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Selecting key columns and parsing JSON strings...")
        selected_cols = ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
        df = df[selected_cols].copy()

        df.dropna(inplace=True)

        df["genres"] = df["genres"].apply(parse_names)
        df["keywords"] = df["keywords"].apply(parse_names)
        df["cast"] = df["cast"].apply(parse_top3_cast)
        df["crew"] = df["crew"].apply(parse_director)

        df["genres"] = df["genres"].apply(collapse_spaces)
        df["keywords"] = df["keywords"].apply(collapse_spaces)
        df["cast"] = df["cast"].apply(collapse_spaces)
        df["crew"] = df["crew"].apply(collapse_spaces)

        df["overview"] = df["overview"].apply(lambda x: str(x).split())

        df["tags"] = df["overview"] + df["genres"] + df["keywords"] + df["cast"] + df["crew"]

        new_df = df[["movie_id", "title", "tags"]].copy()
        new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())

        logging.info(f"Data Transformation completed. Shape: {new_df.shape}")
        return new_df

    except Exception as e:
        raise CustomException(e, sys)


def initiate_data_transformation(input_csv: str, output_csv: str) -> str:
    try:
        logging.info(f"Reading merged dataset from: {input_csv}")
        df = pd.read_csv(input_csv)
        transformed_df = transform_movie_data(df)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        transformed_df.to_csv(output_csv, index=False)
        logging.info(f"Saved transformed dataset at: {output_csv}")

        return output_csv
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_path = os.path.join(base_dir, "Data", "Processed", "raw_merged.csv")
    out_path = os.path.join(base_dir, "Data", "Processed", "movies_processed.csv")
    initiate_data_transformation(raw_path, out_path)
    print(f"Data Transformation Complete! File saved at: {out_path}")
