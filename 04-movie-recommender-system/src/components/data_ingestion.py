import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.exception import CustomException
from src.logger import logging


def load_and_merge_data(movies_path: str, credits_path: str) -> pd.DataFrame:
    try:
        logging.info("Reading raw movies and credits datasets...")
        movies = pd.read_csv(movies_path)
        credits = pd.read_csv(credits_path)

        logging.info(f"Merging datasets on 'title'... Movies: {movies.shape}, Credits: {credits.shape}")
        merged_df = movies.merge(credits, on="title")
        logging.info(f"Merged Dataset shape: {merged_df.shape}")
        return merged_df
    except Exception as e:
        raise CustomException(e, sys)


def initiate_data_ingestion(output_dir: str):
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        movies_path = os.path.join(base_dir, "Data", "Raw", "tmdb_5000_movies.csv")
        credits_path = os.path.join(base_dir, "Data", "Raw", "tmdb_5000_credits.csv")

        merged_df = load_and_merge_data(movies_path, credits_path)

        os.makedirs(output_dir, exist_ok=True)
        raw_merged_path = os.path.join(output_dir, "raw_merged.csv")
        merged_df.to_csv(raw_merged_path, index=False)
        logging.info(f"Saved merged dataset at: {raw_merged_path}")

        return raw_merged_path
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(base_dir, "Data", "Processed")
    saved_path = initiate_data_ingestion(out_dir)
    print(f"Data Ingestion Complete! Saved at: {saved_path}")
