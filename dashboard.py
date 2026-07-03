import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from src.recommendation import recommend_movies

# PAGE CONFIG
st.set_page_config(
    page_title="Bollywood Movie Analytics",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Bollywood Movie Analytics Dashboard")
st.markdown("---")

@st.cache_data
def load_data():
    return pd.read_csv("data/movies.csv")

df = load_data()

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "EDA",
        "Machine Learning",
        "Recommendation",
        "Statistics",
        "About"
    ]
)

if page == "Home":
    st.header("Dataset Overview")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric(
        "Movies",
        len(df)
    )

    col2.metric(
        "Genres",
        df["Genre"].nunique()
    )

    col3.metric(
        "Average Budget",
        f"₹{df['Budget(INR)'].mean():,.0f}"
    )

    col4.metric(
        "Average Revenue",
        f"₹{df['Revenue(INR)'].mean():,.0f}"
    )

    st.write(df.head())

# EDA PAGE
elif page == "EDA":
    st.header("📊 Exploratory Data Analysis")
    chart = st.selectbox(
        "Choose Visualization",
        [
            "Genre Distribution",
            "Revenue Distribution",
            "Budget Distribution",
            "Release Period Distribution",
            "Budget vs Revenue",
            "Top Lead Stars",
            "Top Directors"
        ]
    )

    # Genre Distribution
    if chart == "Genre Distribution":
        genre = df["Genre"].value_counts().reset_index()
        genre.columns = ["Genre","Count"]
        fig = px.bar(
            genre,
            x="Genre",
            y="Count",
            color="Count",
            title="Genre Distribution"
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Revenue Distribution
    elif chart == "Revenue Distribution":
        fig = px.histogram(
            df,
            x="Revenue(INR)",
            nbins=30,
            title="Revenue Distribution"
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Budget Distribution
    elif chart == "Budget Distribution":
        fig = px.histogram(
            df,
            x="Budget(INR)",
            nbins=30,
            title="Budget Distribution"
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Release Period
    elif chart == "Release Period Distribution":
        period = df["Release_Period"].value_counts().reset_index()
        period.columns = ["Release Period","Count"]
        fig = px.pie(
            period,
            values="Count",
            names="Release Period",
            title="Release Period Distribution"
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Budget vs Revenue
    elif chart == "Budget vs Revenue":
        fig = px.scatter(
            df,
            x="Budget(INR)",
            y="Revenue(INR)",
            color="Genre",
            hover_data=["Movie_Name"],
            title="Budget vs Revenue"
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Lead Stars
    elif chart == "Top Lead Stars":
        star = df["Lead_Star"].value_counts().head(10).reset_index()
        star.columns = ["Lead Star","Movies"]
        fig = px.bar(
            star,
            x="Lead Star",
            y="Movies",
            color="Movies",
            title="Top 10 Lead Stars"
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Directors
    elif chart == "Top Directors":
        director = df["Director"].value_counts().head(10).reset_index()
        director.columns = ["Director","Movies"]
        fig = px.bar(
            director,
            x="Director",
            y="Movies",
            color="Movies",
            title="Top Directors"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# MACHINE LEARNING PAGE
elif page == "Machine Learning":
    st.header("Machine Learning Results")

    # Load Regression Results
    try:
        regression = pd.read_csv(
            "outputs/ml/regression_results.csv"
            )
        st.subheader("Regression Model Comparison")
        st.dataframe(
            regression,
            use_container_width=True
            )
        best = regression.sort_values(
            by="R2 Score",
            ascending=False
            ).iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Best Model",
            best["Model"]
            )
        col2.metric(
            "Best R² Score",
            round(best["R2 Score"],4)
            )
        col3.metric(
            "RMSE",
            round(best["RMSE"],2)
            )
    except:
        st.warning("Regression results not found.")
        st.markdown("---")

    # Feature Importance
    try:
        importance = pd.read_csv(
            "outputs/ml/feature_importance.csv"
        )
        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            title="Feature Importance"
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )
    except:
        st.warning("Feature importance file not found.")
    st.markdown("---")

    # Model Performance Chart
    try:
        fig = px.bar(
            regression,
            x="Model",
            y="R2 Score",
            color="Model",
            title="R² Score Comparison"
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )
    except:
        pass
    st.markdown("---")

    # Revenue Prediction
    st.subheader("Revenue Prediction")

    model = joblib.load(
        "models/best_regressor.pkl"
    )
    genre = st.selectbox(
        "Genre",
        sorted(df["Genre"].unique())
    )
    actor = st.selectbox(
        "Lead Star",
        sorted(df["Lead_Star"].unique())
    )
    director = st.selectbox(
        "Director",
        sorted(df["Director"].unique())
    )
    music = st.selectbox(
        "Music Director",
        sorted(df["Music_Director"].unique())
    )
    release = st.selectbox(
        "Release Period",
        sorted(df["Release_Period"].unique())
    )
    remake = st.selectbox(
        "Remake",
        sorted(df["Whether_Remake"].unique())
    )
    franchise = st.selectbox(
        "Franchise",
        sorted(df["Whether_Franchise"].unique())
    )
    new_actor = st.selectbox(
        "New Actor",
        sorted(df["New_Actor"].unique())
    )
    new_director = st.selectbox(
        "New Director",
        sorted(df["New_Director"].unique())
    )
    new_music = st.selectbox(
        "New Music Director",
        sorted(df["New_Music_Director"].unique())
    )
    screens = st.number_input(
        "Number of Screens",
        min_value=1,
        value=500
    )
    budget = st.number_input(
        "Budget (INR)",
        min_value=100000,
        value=50000000
    )

    if st.button("Predict Revenue"):
        st.info(
            "Prediction module will be connected after preprocessing pipeline is saved."
        )
        st.success(
            "Model Loaded Successfully"
        )
    

    # MOVIE RECOMMENDATION PAGE
elif page == "Recommendation":
    st.header("🎬 Bollywood Movie Recommendation System")
    st.write(
        "Select a movie and get the Top 10 similar Bollywood movies."
    )

    try:
        recommendation_df = joblib.load(
            "models/recommendation_dataset.pkl"
        )
        similarity = joblib.load(
            "models/movie_similarity.pkl"
        )
    except:
        st.error(
            "Recommendation model not found.\nRun recommendation_pipeline() first."
        )
        st.stop()

    movie = st.selectbox(
        "Select Movie",
        sorted(
            recommendation_df["Movie_Name"].unique()
        )
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

    if st.button("Recommend Movies"):
        recommendations = recommend_movies(
            movie,
            recommendation_df,
            similarity,
            top_n
        )

        if recommendations is not None:
            st.success("Recommendations Generated Successfully")
            st.dataframe(
                recommendations,
                use_container_width=True
            )
            fig = px.bar(
                recommendations,
                x="Similarity",
                y="Movie",
                orientation="h",
                color="Similarity",
                title="Similarity Scores"
            )
            st.plotly_chart(
                fig,
                use_container_width=True
            )
            csv = recommendations.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Recommendations",
                csv,
                "recommendations.csv",
                "text/csv"
            )

elif page == "Statistics":
    st.header("📈 Statistical Analysis")
    files = {
        "T-Test": "outputs/statistics/t_test.csv",
        "Chi Square": "outputs/statistics/chi_square.csv",
        "ANOVA": "outputs/statistics/anova.csv"
    }

    for title, path in files.items():
        st.subheader(title)
        try:
            data = pd.read_csv(path)
            st.dataframe(
                data,
                use_container_width=True
            )
        except:
            st.warning(f"{title} results not found.")

elif page == "About":
    st.header("About the Project")
    st.markdown("""
                # Bollywood Movie Revenue Prediction & Recommendation System
                ### Features
                - Data Cleaning
                - Feature Engineering
                - Exploratory Data Analysis
                - Statistical Analysis
                - Revenue Prediction
                - Hyperparameter Tuning
                - Feature Importance
                - Movie Recommendation
                - Interactive Dashboard

                ### Algorithms Used
                - Linear Regression
                - Decision Tree
                - Random Forest
                - TF-IDF
                - Cosine Similarity

                #Developed Using
                - Python
                - Pandas
                - NumPy
                - Scikit-Learn
                - Plotly
                - Streamlit
                """)