import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


def build_similarity_matrix(df: pd.DataFrame, max_features: int = 5000):
    try:
        logging.info(f"Vectorizing movie tags with CountVectorizer (max_features={max_features})...")
        cv = CountVectorizer(max_features=max_features, stop_words="english")
        vectors = cv.fit_transform(df["tags"]).toarray()

        logging.info(f"Vectors Shape: {vectors.shape}. Computing Cosine Similarity Matrix...")
        similarity = cosine_similarity(vectors)
        logging.info(f"Cosine Similarity Matrix shape: {similarity.shape}")

        return similarity
    except Exception as e:
        raise CustomException(e, sys)


def initiate_model_trainer(processed_csv: str):
    try:
        logging.info(f"Reading processed movies from: {processed_csv}")
        df = pd.read_csv(processed_csv)

        similarity = build_similarity_matrix(df)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        artifacts_dir = os.path.join(base_dir, "Artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        movies_pkl_path = os.path.join(artifacts_dir, "movies_dict.pkl")
        similarity_pkl_path = os.path.join(artifacts_dir, "similarity.pkl")

        logging.info("Saving artifacts: movies_dict.pkl and similarity.pkl...")
        save_object(movies_pkl_path, df.to_dict())
        save_object(similarity_pkl_path, similarity)

        print(f"Artifacts successfully created in: {artifacts_dir}")
        return movies_pkl_path, similarity_pkl_path

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    processed_path = os.path.join(base_dir, "Data", "Processed", "movies_processed.csv")
    initiate_model_trainer(processed_path)
