import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    pearsonr,
    spearmanr,
    shapiro,
    ttest_ind,
    f_oneway,
    chi2_contingency,
    mannwhitneyu,
    levene,
    skew,
    kurtosis
)

import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor

OUTPUT_DIR = "outputs/statistics"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

def save_plot(name):

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            name
        ),
        dpi=300
    )

    plt.close()

# Descriptive Statistics

def descriptive_statistics(df):

    print("\n")
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    numeric = df.select_dtypes(include=np.number)

    summary = pd.DataFrame()

    summary["Mean"] = numeric.mean()
    summary["Median"] = numeric.median()
    summary["Std"] = numeric.std()
    summary["Variance"] = numeric.var()
    summary["Minimum"] = numeric.min()
    summary["Maximum"] = numeric.max()
    summary["Skewness"] = numeric.apply(skew)
    summary["Kurtosis"] = numeric.apply(kurtosis)

    print(summary.round(3))

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "descriptive_statistics.csv"
        )
    )

    return summary

# Pearson Correlation


def pearson_correlation(df):

    print("\n")
    print("=" * 60)
    print("PEARSON CORRELATION")
    print("=" * 60)

    numeric = df.select_dtypes(include=np.number)

    corr = numeric.corr(method="pearson")

    print(corr.round(2))

    corr.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "pearson_correlation.csv"
        )
    )

    plt.figure(figsize=(12,8))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Pearson Correlation")

    save_plot("pearson_heatmap.png")

# Spearman Correlation

def spearman_correlation(df):

    print("\n")
    print("=" * 60)
    print("SPEARMAN CORRELATION")
    print("=" * 60)

    numeric = df.select_dtypes(include=np.number)

    corr = numeric.corr(method="spearman")

    print(corr.round(2))

    corr.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "spearman_correlation.csv"
        )
    )

    plt.figure(figsize=(12,8))

    sns.heatmap(
        corr,
        annot=True,
        cmap="viridis",
        fmt=".2f"
    )

    plt.title("Spearman Correlation")

    save_plot("spearman_heatmap.png")

# Covariance Matrix

def covariance_matrix(df):

    print("\n")
    print("=" * 60)
    print("COVARIANCE MATRIX")
    print("=" * 60)

    numeric = df.select_dtypes(include=np.number)

    cov = numeric.cov()

    print(cov.round(2))

    cov.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "covariance_matrix.csv"
        )
    )

# Distribution Plots

def distribution_plots(df):

    numeric = df.select_dtypes(include=np.number)

    for column in numeric.columns:

        plt.figure(figsize=(7,4))

        sns.histplot(
            numeric[column],
            kde=True
        )

        plt.title(column)

        save_plot(
            f"{column}_distribution.png"
        )

# Boxplots

def boxplots(df):

    numeric = df.select_dtypes(include=np.number)

    for column in numeric.columns:

        plt.figure(figsize=(6,4))

        sns.boxplot(
            x=numeric[column]
        )

        plt.title(column)

        save_plot(
            f"{column}_boxplot.png"
        )

# Shapiro Test

def shapiro_tests(df):

    print("\n")
    print("=" * 60)
    print("SHAPIRO NORMALITY TEST")
    print("=" * 60)

    numeric = df.select_dtypes(include=np.number)

    results = []

    for column in numeric.columns:

        if len(numeric[column]) < 5000:

            stat, p = shapiro(
                numeric[column]
            )

            results.append(
                [
                    column,
                    stat,
                    p
                ]
            )

            print(
                f"{column:25} p-value = {p:.5f}"
            )

    pd.DataFrame(
        results,
        columns=[
            "Column",
            "Statistic",
            "P-value"
        ]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "shapiro_results.csv"
        ),
        index=False
    )

# QQ Plots

def qq_plots(df):

    numeric = df.select_dtypes(include=np.number)

    for column in numeric.columns:

        plt.figure(figsize=(5,5))

        qqplot(
            numeric[column],
            line="45",
            fit=True
        )

        plt.title(column)

        save_plot(
            f"{column}_qqplot.png"
        )

# Independent T-Test

def independent_t_test(df):

    print("\n")
    print("=" * 60)
    print("INDEPENDENT T-TEST")
    print("=" * 60)

    if "Hit_Flop" not in df.columns:
        print("Hit_Flop column not found.")
        return

    hit = df[df["Hit_Flop"] == 1]["Revenue(INR)"]
    flop = df[df["Hit_Flop"] == 0]["Revenue(INR)"]

    stat, p = ttest_ind(
        hit,
        flop,
        equal_var=False
    )

    print(f"T Statistic : {stat:.4f}")
    print(f"P-value     : {p:.6f}")

    if p < 0.05:
        print("Conclusion : Significant difference exists.")
    else:
        print("Conclusion : No significant difference.")

    result = pd.DataFrame({
        "Statistic":[stat],
        "P-value":[p]
    })

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "t_test.csv"
        ),
        index=False
    )

# Mann Whitney U Test

def mann_whitney_test(df):

    print("\n")
    print("=" * 60)
    print("MANN-WHITNEY U TEST")
    print("=" * 60)

    if "Hit_Flop" not in df.columns:
        print("Hit_Flop column not found.")
        return

    hit = df[df["Hit_Flop"] == 1]["Revenue(INR)"]
    flop = df[df["Hit_Flop"] == 0]["Revenue(INR)"]

    stat, p = mannwhitneyu(
        hit,
        flop,
        alternative="two-sided"
    )

    print(f"Statistic : {stat:.4f}")
    print(f"P-value   : {p:.6f}")

    result = pd.DataFrame({
        "Statistic":[stat],
        "P-value":[p]
    })

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "mann_whitney.csv"
        ),
        index=False
    )

# Levene Test

def levene_test(df):

    print("\n")
    print("=" * 60)
    print("LEVENE TEST")
    print("=" * 60)

    if "Hit_Flop" not in df.columns:
        return

    hit = df[df["Hit_Flop"] == 1]["Revenue(INR)"]
    flop = df[df["Hit_Flop"] == 0]["Revenue(INR)"]

    stat, p = levene(
        hit,
        flop
    )

    print(f"Statistic : {stat:.4f}")
    print(f"P-value   : {p:.6f}")

    pd.DataFrame({
        "Statistic":[stat],
        "P-value":[p]
    }).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "levene_test.csv"
        ),
        index=False
    )

# Chi Square Test

def chi_square_test(df):

    print("\n")
    print("=" * 60)
    print("CHI-SQUARE TEST")
    print("=" * 60)

    if "Genre" not in df.columns:
        return

    if "Hit_Flop" not in df.columns:
        return

    table = pd.crosstab(
        df["Genre"],
        df["Hit_Flop"]
    )

    chi2, p, dof, expected = chi2_contingency(table)

    print(f"Chi Square : {chi2:.4f}")
    print(f"P-value    : {p:.6f}")
    print(f"DOF        : {dof}")

    pd.DataFrame(expected).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "chi_expected.csv"
        ),
        index=False
    )

# ANOVA

def anova_test(df):

    print("\n")
    print("=" * 60)
    print("ONE WAY ANOVA")
    print("=" * 60)

    if "Genre" not in df.columns:
        return

    groups = []

    for genre in df["Genre"].unique():

        groups.append(
            df[df["Genre"] == genre]["Revenue(INR)"]
        )

    stat, p = f_oneway(*groups)

    print(f"F Statistic : {stat:.4f}")
    print(f"P-value     : {p:.6f}")

    pd.DataFrame({
        "Statistic":[stat],
        "P-value":[p]
    }).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "anova.csv"
        ),
        index=False
    )

# Revenue by Genre

def revenue_genre_boxplot(df):

    if "Genre" not in df.columns:
        return

    plt.figure(figsize=(12,6))

    sns.boxplot(
        data=df,
        x="Genre",
        y="Revenue(INR)"
    )

    plt.xticks(rotation=45)

    plt.title("Revenue by Genre")

    save_plot("genre_revenue_boxplot.png")

# ROI by Genre

def roi_genre_boxplot(df):

    if "ROI" not in df.columns:
        return

    plt.figure(figsize=(12,6))

    sns.boxplot(
        data=df,
        x="Genre",
        y="ROI"
    )

    plt.xticks(rotation=45)

    plt.title("ROI by Genre")

    save_plot("genre_roi_boxplot.png")

# Statistical Summary Report

def statistical_summary(df):

    report = pd.DataFrame()

    numeric = df.select_dtypes(include="number")

    report["Mean"] = numeric.mean()
    report["Median"] = numeric.median()
    report["Std"] = numeric.std()
    report["Minimum"] = numeric.min()
    report["Maximum"] = numeric.max()

    report.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "summary_report.csv"
        )
    )

    print("\nSummary Report Saved.")

# Multiple Linear Regression (OLS)

import statsmodels.api as sm

def ols_regression(df):

    print("\n" + "="*60)
    print("MULTIPLE LINEAR REGRESSION")
    print("="*60)

    if "Revenue(INR)" not in df.columns:
        return

    numeric_df = df.select_dtypes(include="number")

    X = numeric_df.drop("Revenue(INR)", axis=1)
    y = numeric_df["Revenue(INR)"]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print(model.summary())

    with open(
        os.path.join(
            OUTPUT_DIR,
            "ols_summary.txt"
        ),
        "w"
    ) as f:

        f.write(model.summary().as_text())

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):

    print("\n" + "="*60)
    print("VARIANCE INFLATION FACTOR")
    print("="*60)

    numeric = df.select_dtypes(include="number")

    if "Revenue(INR)" in numeric.columns:
        numeric = numeric.drop(
            "Revenue(INR)",
            axis=1
        )

    vif = pd.DataFrame()

    vif["Feature"] = numeric.columns

    vif["VIF"] = [

        variance_inflation_factor(
            numeric.values,
            i
        )

        for i in range(
            numeric.shape[1]
        )

    ]

    print(vif)

    vif.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "vif.csv"
        ),
        index=False
    )

def residual_analysis(df):

    numeric = df.select_dtypes(include="number")

    X = numeric.drop(
        "Revenue(INR)",
        axis=1
    )

    y = numeric["Revenue(INR)"]

    X = sm.add_constant(X)

    model = sm.OLS(
        y,
        X
    ).fit()

    residuals = model.resid

    plt.figure(figsize=(8,5))

    sns.histplot(
        residuals,
        kde=True
    )

    plt.title("Residual Distribution")

    save_plot("residual_distribution.png")

def residual_plot(df):
    numeric = df.select_dtypes(include="number")
    X = numeric.drop(
        "Revenue(INR)",
        axis=1
    )
    y = numeric["Revenue(INR)"]
    X = sm.add_constant(X)
    model = sm.OLS(
        y,
        X
    ).fit()

    prediction = model.predict(X)
    residual = model.resid
    plt.figure(figsize=(8,6))
    plt.scatter(
        prediction,
        residual
    )

    plt.axhline(
        y=0,
        color="red"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    save_plot("residual_plot.png")

from sklearn.ensemble import RandomForestRegressor
def feature_importance(df):
    numeric = df.select_dtypes(include="number")
    X = numeric.drop(
        "Revenue(INR)",
        axis=1
    )
    y = numeric["Revenue(INR)"]

    rf = RandomForestRegressor(
        random_state=42
    )

    rf.fit(
        X,
        y
    )

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print(importance)

    importance.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    plt.figure(figsize=(10,6))
    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature"
    )

    plt.title("Feature Importance")
    save_plot(
        "feature_importance.png"
    )

def model_comparison():

    comparison = pd.DataFrame({
        "Model":[
            "Linear Regression",
            "Random Forest",
            "Decision Tree"
        ],
        "Purpose":[
            "Regression",
            "Regression",
            "Classification"
        ]
    })

    comparison.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "model_comparison.csv"
        ),
        index=False
    )

