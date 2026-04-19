import streamlit as st
import numpy as np
import pandas as pd
import ast
import pickle
import os
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; }

    h1 { color: #e50914 !important; font-size: 2.8rem !important; }
    h2, h3 { color: #ffffff !important; }
    p, label, .stMarkdown { color: #cccccc !important; }

    .movie-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #333;
        border-radius: 14px;
        padding: 0;
        overflow: hidden;
        transition: transform 0.25s, border-color 0.25s;
    }
    .movie-card:hover {
        transform: scale(1.04);
        border-color: #e50914;
    }
    .movie-card img {
        width: 100%;
        aspect-ratio: 2/3;
        object-fit: cover;
        display: block;
        border-radius: 14px 14px 0 0;
    }
    .movie-card-body {
        padding: 10px 10px 14px 10px;
        text-align: center;
    }
    .rank-badge {
        display: inline-block;
        background: #e50914;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        line-height: 24px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-bottom: 6px;
    }
    .movie-title {
        color: #ffffff;
        font-size: 0.88rem;
        font-weight: 600;
        margin: 0;
        line-height: 1.3;
    }
    .no-poster {
        width: 100%;
        aspect-ratio: 2/3;
        background: #1a1a2e;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        border-radius: 14px 14px 0 0;
    }

    div[data-testid="stSelectbox"] > div > div {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
        border: 1px solid #e50914 !important;
        border-radius: 8px !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #e50914, #b20710);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .status-box {
        background: #1a1a2e;
        border-left: 4px solid #e50914;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 12px 0;
        color: #cccccc;
    }

    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
PKL_MOVIES     = "movie_list.pkl"
PKL_SIMILARITY = "similarity.pkl"
TMDB_API_KEY   = "29031496149dcf0e763e3c16fc98edee"   # public demo key
TMDB_IMG_BASE  = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER    = "https://placehold.co/300x450/1a1a2e/555?text=No+Poster"


# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def fetch_director(text):
    return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']

def collapse(L):
    return [i.replace(" ", "") for i in L]


# ─────────────────────────────────────────────
# Train & save pkl (only called when pkl missing)
# ─────────────────────────────────────────────
def train_and_save(movies_path: str, credits_path: str):
    progress = st.progress(0, text="Loading CSVs…")

    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    progress.progress(15, text="Merging datasets…")

    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)
    progress.progress(30, text="Parsing genres & keywords…")

    movies['genres']   = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast']     = movies['cast'].apply(convert).apply(lambda x: x[:3])
    movies['crew']     = movies['crew'].apply(fetch_director)
    progress.progress(50, text="Collapsing tags…")

    for col in ['cast', 'crew', 'genres', 'keywords']:
        movies[col] = movies[col].apply(collapse)

    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['tags'] = (movies['overview'] + movies['genres'] +
                      movies['keywords'] + movies['cast'] + movies['crew'])
    progress.progress(65, text="Vectorising with CountVectorizer…")

    new = movies[['movie_id', 'title']].copy()
    new['tags'] = movies['tags'].apply(lambda x: " ".join(x))

    cv     = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(new['tags']).toarray()
    progress.progress(82, text="Computing cosine similarity…")

    similarity = cosine_similarity(vector)
    progress.progress(95, text="Saving model to disk…")

    pickle.dump(new, open(PKL_MOVIES, 'wb'))
    pickle.dump(similarity, open(PKL_SIMILARITY, 'wb'))
    progress.progress(100, text="Done!")

    return new, similarity


# ─────────────────────────────────────────────
# Load model — pkl if available, else train
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(movies_path: str, credits_path: str):
    if os.path.exists(PKL_MOVIES) and os.path.exists(PKL_SIMILARITY):
        new        = pickle.load(open(PKL_MOVIES, 'rb'))
        similarity = pickle.load(open(PKL_SIMILARITY, 'rb'))
        return new, similarity, False   # False = loaded from pkl, no training
    else:
        new, similarity = train_and_save(movies_path, credits_path)
        return new, similarity, True    # True  = freshly trained


# ─────────────────────────────────────────────
# Fetch poster from TMDB using movie_id
# ─────────────────────────────────────────────
def fetch_poster(movie_id: int) -> str:
    try:
        url  = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        data = requests.get(url, timeout=5).json()
        path = data.get('poster_path')
        return TMDB_IMG_BASE + path if path else PLACEHOLDER
    except Exception:
        return PLACEHOLDER


# ─────────────────────────────────────────────
# Get top-5 recommendations
# ─────────────────────────────────────────────
def get_recommendations(movie: str, new_df: pd.DataFrame, similarity):
    idx       = new_df[new_df['title'] == movie].index[0]
    distances = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    results   = []
    for i, _ in distances[1:6]:
        row = new_df.iloc[i]
        results.append({"title": row['title'], "movie_id": int(row['movie_id'])})
    return results


# ═════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════
st.markdown("# 🎬 Movie Recommender")
st.markdown("*Content-based recommendations · genres · keywords · cast · director*")
st.markdown("---")

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Setup")
    movies_path  = st.text_input("Movies CSV",  value="tmdb_5000_movies.csv")
    credits_path = st.text_input("Credits CSV", value="tmdb_5000_credits.csv")

    pkl_exists = os.path.exists(PKL_MOVIES) and os.path.exists(PKL_SIMILARITY)
    st.markdown("---")
    if pkl_exists:
        st.success("✅ Saved model found — skipping training")
        if st.button("🔄 Force retrain"):
            os.remove(PKL_MOVIES)
            os.remove(PKL_SIMILARITY)
            st.cache_resource.clear()
            st.rerun()
    else:
        st.warning("⚠️ No saved model — will train on first run")

    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown("""
- Merges movie metadata with cast & crew
- Builds a **tag** from genres, keywords, cast, director & overview
- Vectorises with **CountVectorizer** (5 000 features)
- Ranks movies by **cosine similarity**
- Training result saved to **pkl** files
- Posters fetched live from **TMDB API**
""")

# ── Load / train model ────────────────────────
model_ready = False
new_df, sim = None, None

data_ok = (
    os.path.exists(PKL_MOVIES) and os.path.exists(PKL_SIMILARITY)
) or (
    os.path.exists(movies_path) and os.path.exists(credits_path)
)

if data_ok:
    try:
        with st.spinner("Loading model…"):
            new_df, sim, trained = load_model(movies_path, credits_path)
        if trained:
            st.success("✅ Model trained and saved to disk. Future runs will load instantly.")
        model_ready = True
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.markdown("""
<div class="status-box">
    📂 <strong>Data not found.</strong> Place
    <code>tmdb_5000_movies.csv</code> and <code>tmdb_5000_credits.csv</code>
    in the same folder as <code>app.py</code>, or update the paths in the sidebar.
</div>
""", unsafe_allow_html=True)

# ── Inference UI ──────────────────────────────
if model_ready:
    st.markdown("## 🔍 Find Similar Movies")

    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        movie_list = sorted(new_df['title'].tolist())
        selected   = st.selectbox("Pick a movie", movie_list, label_visibility="collapsed")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("Recommend ▶")

    if go or selected:
        recs = get_recommendations(selected, new_df, sim)

        # Show selected movie poster + info
        sel_id     = int(new_df[new_df['title'] == selected]['movie_id'].values[0])
        sel_poster = fetch_poster(sel_id)

        st.markdown("---")
        hero_col, info_col = st.columns([1, 3])
        with hero_col:
            st.image(sel_poster, use_container_width=True)
        with info_col:
            st.markdown(f"### {selected}")
            st.markdown("**Top 5 similar movies:**")
            for i, r in enumerate(recs, 1):
                st.markdown(f"`#{i}` &nbsp; {r['title']}")

        st.markdown("---")
        st.markdown(f"### 🎯 Recommendations for **{selected}**")
        st.markdown("")

        cols = st.columns(5)
        for rank, (col, rec) in enumerate(zip(cols, recs), start=1):
            poster_url = fetch_poster(rec['movie_id'])
            with col:
                st.markdown(f"""
<div class="movie-card">
    <img src="{poster_url}"
         alt="{rec['title']}"
         onerror="this.style.display='none';this.nextSibling.style.display='flex'"/>
    <div class="no-poster" style="display:none">🎬</div>
    <div class="movie-card-body">
        <div class="rank-badge">#{rank}</div>
        <p class="movie-title">{rec['title']}</p>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f"<p style='color:#555;font-size:0.8rem;'>Model trained on "
        f"<strong style='color:#888'>{len(new_df):,}</strong> movies · "
        f"5 000-feature bag-of-words · cosine similarity · "
        f"Posters via TMDB API</p>",
        unsafe_allow_html=True,
    )
