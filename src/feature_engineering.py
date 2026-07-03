import pandas as pd


def create_features(df):

    # Budget Categories

    df["Budget_Category"] = pd.qcut(
        df["Budget(INR)"],
        q=3,
        labels=["Low", "Medium", "High"]
    )

    # Revenue Categories

    df["Revenue_Category"] = pd.qcut(
        df["Revenue(INR)"],
        q=3,
        labels=["Low", "Medium", "High"]
    )

    # ROI

    df["ROI"] = (
        df["Revenue(INR)"] -
        df["Budget(INR)"]
    ) / df["Budget(INR)"]

    # Revenue per Screen

    df["Revenue_per_screen"] = (
        df["Revenue(INR)"] /
        df["Number_of_Screens"]
    )

    # Combined Actor Director Feature

    df["combined"] = (
        df["Lead_Star"].astype(str)
        +
        "_"
        +
        df["Director"].astype(str)
    )

    return df


def create_hit_flop(df):

    median_revenue = df["Revenue(INR)"].median()

    df["Hit_Flop"] = (
        df["Revenue(INR)"] >= median_revenue
    ).astype(int)

    return df