import os
import sys
import streamlit as st
import requests

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.pipeline.recommend_pipeline import RecommendPipeline

st.set_page_config(
    page_title="Cinematic AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        color: #e50914;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .movie-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        border: 1px solid #334155;
        transition: transform 0.2s;
    }
    .movie-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 8px;
        min-height: 42px;
    }
    .score-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        background: rgba(229, 9, 20, 0.2);
        color: #e50914;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🎬 CineMatch: AI Movie Recommendation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Content-Based Filtering • Cosine Similarity • Real-Time Discovery</div>', unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return RecommendPipeline()

try:
    engine = load_engine()
    movie_list = engine.get_all_titles()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Helper to fetch movie poster via free public TMDB API fallback
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except Exception:
        pass
    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=60"

with st.sidebar:
    st.header("⚙️ Recommendation Controls")
    top_k = st.slider("Number of Recommendations:", min_value=3, max_value=10, value=5)
    st.markdown("---")
    st.info("**Model Type:** Vector Cosine Similarity\n\n**Feature Vectors:** 5,000 Bag of Words\n\n**Catalog:** TMDB 5000 Movies")

selected_movie = st.selectbox("🔎 Select or Type a Movie You Love:", movie_list, index=movie_list.index("Avatar") if "Avatar" in movie_list else 0)

if st.button("🚀 Recommend Similar Movies", type="primary", use_container_width=True):
    with st.spinner("Analyzing genre, plot keywords, actors, and director vectors..."):
        recommendations = engine.recommend(selected_movie, top_k=top_k)

        st.subheader(f"🍿 Because you watched '{selected_movie}':")
        cols = st.columns(top_k)

        for idx, rec in enumerate(recommendations):
            poster_url = fetch_poster(rec["movie_id"])
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <img src="{poster_url}" style="width:100%; border-radius:8px; aspect-ratio: 2/3; object-fit: cover;">
                        <div class="movie-title">{rec['title']}</div>
                        <span class="score-badge">Match: {int(rec['similarity_score'] * 100)}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
