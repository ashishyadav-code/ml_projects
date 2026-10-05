import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.exception import CustomException
from src.utils import load_object


class RecommendPipeline:
    def __init__(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.movies_dict_path = os.path.join(base_dir, "Artifacts", "movies_dict.pkl")
        self.similarity_path = os.path.join(base_dir, "Artifacts", "similarity.pkl")

        if not os.path.exists(self.movies_dict_path) or not os.path.exists(self.similarity_path):
            raise FileNotFoundError("Artifacts not found! Please run the training pipeline first.")

        self.movies = pd.DataFrame(load_object(self.movies_dict_path))
        self.similarity = load_object(self.similarity_path)

    def get_all_titles(self):
        return sorted(self.movies["title"].tolist())

    def recommend(self, movie_title: str, top_k: int = 5):
        try:
            if movie_title not in self.movies["title"].values:
                raise ValueError(f"Movie '{movie_title}' not found in database.")

            movie_index = self.movies[self.movies["title"] == movie_title].index[0]
            distances = list(enumerate(self.similarity[movie_index]))
            sorted_movies = sorted(distances, reverse=True, key=lambda x: x[1])[1 : top_k + 1]

            recommendations = []
            for i in sorted_movies:
                rec_id = int(self.movies.iloc[i[0]]["movie_id"])
                rec_title = str(self.movies.iloc[i[0]]["title"])
                rec_score = float(i[1])
                recommendations.append({
                    "movie_id": rec_id,
                    "title": rec_title,
                    "similarity_score": round(rec_score, 4)
                })

            return recommendations

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = RecommendPipeline()
    sample_movie = "Avatar"
    recs = pipeline.recommend(sample_movie, top_k=5)
    print(f"\nRecommended Movies for '{sample_movie}':")
    for r in recs:
        print(f"• {r['title']} (Similarity: {r['similarity_score']})")
