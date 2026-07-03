import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = "outputs/recommendation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def load_dataset(filepath):
    print("\nLoading Dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset Loaded Successfully ({len(df)} movies)")
    return df


def create_content_features(df):
    print("\nCreating Content Features...")
    columns = [
        "Genre",
        "Lead_Star",
        "Director",
        "Music_Director",
        "Release_Period",
        "Whether_Remake",
        "Whether_Franchise"
    ]
    for col in columns:
        df[col] = df[col].astype(str)
    df["content"] = (
        df["Genre"] + " " +
        df["Lead_Star"] + " " +
        df["Director"] + " " +
        df["Music_Director"] + " " +
        df["Release_Period"] + " " +
        df["Whether_Remake"] + " " +
        df["Whether_Franchise"]
    )
    print("Content Features Created")

    return df

# TF-IDF MATRIX
def build_tfidf(df):
    print("\nBuilding TF-IDF Matrix...")
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(
        df["content"]
    )
    joblib.dump(
        vectorizer,
        "models/tfidf_vectorizer.pkl"
    )
    print("TF-IDF Shape:", tfidf_matrix.shape)
    return tfidf_matrix

# COSINE SIMILARITY
def compute_similarity(tfidf_matrix):
    print("\nComputing Cosine Similarity...")
    similarity = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )
    print("Similarity Matrix Created")
    return similarity

# SAVE MATRIX
def save_similarity(similarity):

    joblib.dump(
        similarity,
        "models/movie_similarity.pkl"
    )
    print("Similarity Matrix Saved")

def recommend_movies(
        movie_name,
        df,
        similarity,
        top_n=10
):

    movie_name = movie_name.strip().lower()
    movie_index = None
    for i, movie in enumerate(df["Movie_Name"]):
        if movie.lower() == movie_name:
            movie_index = i
            break

    if movie_index is None:
        print(f"\nMovie '{movie_name}' not found.")
        return None

    similarity_scores = list(
        enumerate(similarity[movie_index])
    )
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )
    similarity_scores = similarity_scores[1:top_n+1]
    recommendations = []
    for index, score in similarity_scores:
        recommendations.append({
            "Movie": df.iloc[index]["Movie_Name"],
            "Genre": df.iloc[index]["Genre"],
            "Lead Star": df.iloc[index]["Lead_Star"],
            "Similarity": round(score,3)
        })
    recommendation_df = pd.DataFrame(
        recommendations
    )
    return recommendation_df

# SEARCH MOVIES
def search_movie(
        keyword,
        df
):
    keyword = keyword.lower()
    movies = df[
        df["Movie_Name"]
        .str.lower()
        .str.contains(keyword)
    ]
    return movies["Movie_Name"].tolist()

# SAVE RECOMMENDATIONS
def save_recommendations(
        recommendation_df
):
    recommendation_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "sample_recommendations.csv"
        ),
        index=False
    )
    print("Recommendations Saved.")

# COMPLETE PIPELINE
def recommendation_pipeline(filepath):
    print("\n")
    print("MOVIE RECOMMENDATION SYSTEM")
    df = load_dataset(filepath)
    df = create_content_features(df)
    tfidf_matrix = build_tfidf(df)
    similarity = compute_similarity(
        tfidf_matrix
    )
    save_similarity(similarity)
    joblib.dump(
        df,
        "models/recommendation_dataset.pkl"
    )
    print("\nRecommendation System Ready.")
    return df, similarity

# INTERACTIVE RECOMMENDER
def start_recommendation(
        df,
        similarity
):
    print("\n")
    print("BOLLYWOOD MOVIE RECOMMENDER")

    while True:
        movie = input(
            "\nEnter Movie Name (or 'exit'): "
        )
        if movie.lower() == "exit":
            break

        recommendations = recommend_movies(
            movie,
            df,
            similarity
        )
        if recommendations is None:
            print("\nMovie not found.")
            matches = search_movie(
                movie,
                df
            )
            if len(matches):
                print("\nDid you mean:")
                for m in matches[:10]:
                    print("-", m)
            continue

        print("\nTop Recommended Movies\n")
        print(recommendations)
        save_recommendations(
            recommendations
        )