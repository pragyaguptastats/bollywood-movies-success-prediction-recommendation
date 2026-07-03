# Bollywood Movie Revenue Prediction & Recommendation System

An end-to-end Machine Learning and Statistics project that predicts Bollywood movie revenue, performs comprehensive statistical analysis, explains model predictions, and recommends similar movies through an interactive Streamlit dashboard.

---

## Project Overview

The Indian film industry produces hundreds of movies every year, making it difficult to estimate commercial success before release. This project uses statistical analysis and machine learning techniques to analyze Bollywood movie data, predict box office revenue, and recommend similar movies based on their characteristics.

The project demonstrates the complete Data Science workflow:

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Machine Learning
- Model Tuning
- Explainable AI
- Movie Recommendation System
- Interactive Dashboard

---

## Objectives

- Clean and preprocess Bollywood movie data.
- Perform exploratory and statistical analysis.
- Identify important factors affecting movie revenue.
- Predict movie revenue using Machine Learning models.
- Tune models for improved performance.
- Explain model predictions using Explainable AI.
- Recommend similar Bollywood movies.
- Visualize results through a Streamlit dashboard.

---

# Project Structure

```
Bollywood-Movie-Revenue-Prediction-Recommendation-System
│
├── data/
│   ├── movies.csv
│   └── processed_movies.csv
│
├── models/
│   ├── best_regressor.pkl
│   ├── final_best_model.pkl
│   ├── tuned_random_forest.pkl
│   ├── tuned_decision_tree.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── movie_similarity.pkl
│   └── recommendation_dataset.pkl
│
├── outputs/
│   ├── explainability/
│   ├── ml/
│   ├── model_tuning/
│   ├── recommendation/
│   └── statistics/
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── statistical_analysis.py
│   ├── machine_learning.py
│   ├── model_tuning.py
│   ├── explainability.py
│   ├── recommendation.py
│   └── predict.py
│
├── dashboard.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset Features

The dataset contains information such as:

- Movie Name
- Genre
- Director
- Lead Star
- Music Director
- Release Period
- Budget
- Revenue
- Number of Screens
- Franchise Status
- Remake Status
- New Actor
- New Director
- New Music Director

---

# Data Preprocessing

The preprocessing stage includes:

- Duplicate removal
- Missing value handling
- Data type conversion
- Label Encoding
- Feature Scaling
- Revenue categorization
- ROI calculation
- Hit/Flop classification

---

# Exploratory Data Analysis

EDA includes:

- Revenue Distribution
- Budget Distribution
- ROI Distribution
- Genre Distribution
- Budget vs Revenue Scatter Plot
- Correlation Heatmap
- Boxplots
- Histograms

---

# Statistical Analysis

The following statistical techniques were performed:

- Descriptive Statistics
- Correlation Analysis
- Independent T-Test
- One-Way ANOVA
- Chi-Square Test
- Normality Test
- Outlier Detection

---

# Machine Learning Models

## Regression Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Evaluation Metrics:

- MAE
- MSE
- RMSE
- R² Score

---

# Model Tuning

Hyperparameter tuning performed using:

- GridSearchCV

Models Tuned:

- Decision Tree
- Random Forest

---

# Explainable AI

Model explainability includes:

- Feature Importance
- Feature Ranking
- Prediction Interpretation

---

# Movie Recommendation System

The recommendation engine uses:

- TF-IDF Vectorization
- Cosine Similarity

Users can:

- Search any movie
- Receive Top 10 similar Bollywood movie recommendations

---

# Interactive Dashboard

Developed using **Streamlit**.

Dashboard Features:

- Dataset Overview
- Exploratory Data Analysis
- Revenue Prediction
- Movie Recommendation
- Model Performance
- Statistical Analysis

Run using:

```bash
streamlit run dashboard.py
```

---

# Technologies Used

### Programming

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Joblib
- Streamlit

---

# Results

Among all regression models:

- Random Forest achieved the highest prediction accuracy.
- Feature importance analysis identified the most influential variables affecting movie revenue.
- Statistical tests confirmed significant relationships between movie characteristics and revenue.
- The recommendation engine successfully suggests similar Bollywood movies based on movie metadata.

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Bollywood-Movie-Revenue-Prediction-Recommendation-System.git
```

Move into the project folder

```bash
cd Bollywood-Movie-Revenue-Prediction-Recommendation-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the main analysis

```bash
python main.py
```

Launch the dashboard

```bash
streamlit run dashboard.py
```

---

# Future Improvements

- Deep Learning Models
- XGBoost & LightGBM
- IMDb API Integration
- Real-time Movie Search
- Poster Recommendations
- Cloud Deployment
- User Authentication
- Advanced Recommendation Algorithms

---

# Author

**Pragya Gupta**

Master's Student in Statistics
University of Delhi

Interested in:

- Machine Learning
- Data Science
- Statistical Modelling
- Artificial Intelligence
- Recommendation Systems

---

# ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub.
