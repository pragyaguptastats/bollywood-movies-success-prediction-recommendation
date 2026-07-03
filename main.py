from src.preprocessing import (
    load_data,
    basic_information,
    remove_duplicates,
    handle_missing_values,
    encode_categorical
)

from src.feature_engineering import (
    create_features,
    create_hit_flop
)

# Load Dataset

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)
df = load_data("data/movies.csv")

# Basic Information

print("\nDATASET INFORMATION")
print("-" * 60)
basic_information(df)

# Remove Duplicate Rows

print("\nRemoving duplicate rows...")
before = len(df)
df = remove_duplicates(df)
after = len(df)
print(f"Removed {before-after} duplicate rows.")


import pandas as pd

def handle_missing_values(df):
    """
    Fill missing values:
    - Numeric columns -> median
    - Categorical/String columns -> mode
    """

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

# Feature Engineering
print("\nCreating new features...")
df = create_features(df)
print("Feature Engineering Completed.")

# Create Hit/Flop Target
print("\nCreating Hit/Flop column...")
df = create_hit_flop(df)
print("Hit/Flop column created.")

# Encode Categorical Variables
print("\nEncoding categorical variables...")
df = encode_categorical(df)
print("Encoding completed.")

# Final Dataset Information
print("\nFINAL DATASET SHAPE")
print(df.shape)

print("\nFIRST FIVE ROWS")
print(df.head())

print("\nDATA TYPES")
print(df.dtypes)
print("\nMISSING VALUES")
print(df.isnull().sum())

# Save Processed Dataset

df.to_csv(
    "data/processed_movies.csv",
    index=False
)

print("\nProcessed dataset saved")
print("\nLocation : data/processed_movies.csv")

from src.eda import *

create_output_folder()
dataset_summary(df)
revenue_distribution(df)
budget_distribution(df)
roi_distribution(df)
correlation_heatmap(df)
budget_vs_revenue(df)
genre_analysis(df)

print("\nEDA Completed")

from src.statistical_analysis import (
    descriptive_statistics,
    residual_plot,
    statistical_summary,
    pearson_correlation,
    spearman_correlation,
    covariance_matrix,
    distribution_plots,
    boxplots,
    shapiro_tests,
    independent_t_test,
    mann_whitney_test,
    levene_test,
    chi_square_test,
    anova_test,
    revenue_genre_boxplot,
    roi_genre_boxplot,
    qq_plots,
    ols_regression,
    calculate_vif,
    residual_analysis,
    feature_importance,
    model_comparison
)
print("STATISTICAL ANALYSIS")

descriptive_statistics(df)
pearson_correlation(df)
spearman_correlation(df)
covariance_matrix(df)
distribution_plots(df)
boxplots(df)
shapiro_tests(df)
qq_plots(df)
independent_t_test(df)
mann_whitney_test(df)
levene_test(df)
chi_square_test(df)
anova_test(df)
revenue_genre_boxplot(df)
roi_genre_boxplot(df)
statistical_summary(df)

print("ADVANCED STATISTICAL ANALYSIS")
ols_regression(df)
calculate_vif(df)
residual_analysis(df)
residual_plot(df)
feature_importance(df)
model_comparison()

from src.machine_learning import regression_pipeline
print("\nStarting Machine Learning...\n")
regression_pipeline(df)

from src.machine_learning import *
from src.model_tuning import *
print("\n")
print("MODEL TUNING")

X_train, X_test, y_train, y_test, feature_names = prepare_regression_data(df)

model_tuning_pipeline(
    X_train,
    X_test,
    y_train,
    y_test
)

from src.explainability import *
import joblib

best_model = joblib.load(
    "models/best_regressor.pkl"
)

explainability_pipeline(
    best_model,
    X_test,
    y_test,
    feature_names
)

from src.recommendation import *
print("\n")
print("MOVIE RECOMMENDATION SYSTEM")

df_rec, similarity = recommendation_pipeline(
    "data/movies.csv"
)

start_recommendation(
    df_rec,
    similarity
)