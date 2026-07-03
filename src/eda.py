import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def create_output_folder():

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

def dataset_summary(df):

    print("="*60)
    print("DATASET SUMMARY")
    print("="*60)

    print(df.describe(include="all"))

def revenue_distribution(df):

    plt.figure()

    sns.histplot(
        df["Revenue(INR)"],
        bins=30,
        kde=True
    )

    plt.title("Revenue Distribution")

    plt.savefig(
        "outputs/revenue_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def budget_distribution(df):

    plt.figure()

    sns.histplot(
        df["Budget(INR)"],
        bins=30,
        kde=True
    )

    plt.title("Budget Distribution")

    plt.savefig(
        "outputs/budget_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def roi_distribution(df):

    plt.figure()

    sns.histplot(
        df["ROI"],
        bins=30,
        kde=True
    )

    plt.title("ROI Distribution")

    plt.savefig(
        "outputs/roi_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def correlation_heatmap(df):

    plt.figure(figsize=(12,8))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    plt.savefig(
        "outputs/correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def budget_vs_revenue(df):

    plt.figure()

    sns.scatterplot(
        data=df,
        x="Budget(INR)",
        y="Revenue(INR)"
    )

    plt.title("Budget vs Revenue")

    plt.savefig(
        "outputs/budget_vs_revenue.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def genre_analysis(df):

    plt.figure(figsize=(10,6))

    sns.countplot(
        data=df,
        x="Genre"
    )

    plt.xticks(rotation=45)

    plt.title("Genre Distribution")

    plt.savefig(
        "outputs/genre_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

