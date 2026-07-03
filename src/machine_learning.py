import os
import time
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# OUTPUT FOLDERS
OUTPUT_DIR = "outputs/ml"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)

def prepare_regression_data(df):
    print("Preparing Regression Dataset")
    data = df.copy()
    # Remove columns not needed
    data.drop(
        columns=[
            "Movie_Name",
            "Revenue_Category"
        ],
        errors="ignore",
        inplace=True
    )
    # Encode ALL non-numeric columns
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(
                data[col].astype(str)
            )
    X = data.drop("Revenue(INR)", axis=1)
    y = data["Revenue(INR)"]
    feature_names = X.columns
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
)

    return X_train, X_test, y_train, y_test, feature_names

# CLASSIFICATION DATA
def prepare_classification_data(df):
    print("Preparing Classification Dataset")
    data = df.copy()

    if "Movie_Name" in data.columns:
        data.drop(
            "Movie_Name",
            axis=1,
            inplace=True
        )

    if "Hit_Flop" not in data.columns:
        median = data["Revenue(INR)"].median()
        data["Hit_Flop"] = np.where(
            data["Revenue(INR)"]>=median,
            1,
            0
        )

    for col in data.columns:

        if not pd.api.types.is_numeric_dtype(data[col]):

            encoder = LabelEncoder()

            data[col] = encoder.fit_transform(
                data[col].astype(str)
        )

    X = data.drop(
        ["Revenue(INR)","Hit_Flop"],
        axis=1
    )

    y = data["Hit_Flop"]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Classification Dataset Ready")
    return X_train,X_test,y_train,y_test

# REGRESSION EVALUATION
def regression_metrics(
    model_name,
    y_test,
    prediction
):

    mae = mean_absolute_error(
        y_test,
        prediction
    )
    mse = mean_squared_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(mse)
    r2 = r2_score(
        y_test,
        prediction
    )

    print()
    print(model_name)
    print("-"*40)
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return [
        model_name,
        mae,
        mse,
        rmse,
        r2
    ]

# ACTUAL VS PREDICTED
def plot_prediction(
    y,
    pred,
    model_name
):
    plt.figure(
        figsize=(8,6)
    )
    plt.scatter(
        y,
        pred
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(
        model_name
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            model_name+"_prediction.png"
        )
    )
    plt.close()

# LINEAR REGRESSION
def linear_regression_model(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n")
    print("LINEAR REGRESSION")

    start = time.time()
    model = LinearRegression()
    model.fit(
        X_train,
        y_train
    )
    pred = model.predict(
        X_test
    )

    training_time = time.time() - start
    result = regression_metrics(
        "Linear Regression",
        y_test,
        pred
    )
    plot_prediction(
        y_test,
        pred,
        "Linear_Regression"
    )
    return model, result, training_time

# DECISION TREE REGRESSOR
def decision_tree_regressor(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n")
    print("DECISION TREE REGRESSOR")

    start = time.time()
    model = DecisionTreeRegressor(
        random_state=42,
        max_depth=10
    )
    model.fit(
        X_train,
        y_train
    )
    pred = model.predict(
        X_test
    )
    training_time = time.time() - start
    result = regression_metrics(
        "Decision Tree",
        y_test,
        pred
    )
    plot_prediction(
        y_test,
        pred,
        "Decision_Tree"
    )
    return model, result, training_time

# RANDOM FOREST REGRESSOR

def random_forest_regressor(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n")
    print("RANDOM FOREST REGRESSOR")

    start = time.time()

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=12
    )

    model.fit(
        X_train,
        y_train
    )
    pred = model.predict(
        X_test
    )

    training_time = time.time() - start
    result = regression_metrics(
        "Random Forest",
        y_test,
        pred
    )
    plot_prediction(
        y_test,
        pred,
        "Random_Forest"
    )
    return model, result, training_time

# MODEL COMPARISON

def compare_regression_models(
    lr_result,
    dt_result,
    rf_result
):

    comparison = pd.DataFrame(
        [
            lr_result,
            dt_result,
            rf_result
        ],

        columns=[
            "Model",
            "MAE",
            "MSE",
            "RMSE",
            "R2 Score"
        ]
    )
    comparison.sort_values(
        by="R2 Score",
        ascending=False,
        inplace=True
    )

    print("\n")
    print("REGRESSION MODEL COMPARISON")
    print(comparison)
    comparison.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "regression_results.csv"
        ),
        index=False
    )
    return comparison

# SAVE BEST MODEL
def save_best_regressor(
    comparison,
    lr,
    dt,
    rf
):
    best = comparison.iloc[0]["Model"]
    if best == "Linear Regression":
        model = lr
    elif best == "Decision Tree":
        model = dt
    else:
        model = rf
    joblib.dump(
        model,
        "models/best_regressor.pkl"
    )
    print("\nBest Regression Model Saved")
    print(best)

# FEATURE IMPORTANCE
def feature_importance(
    model,
    feature_names
):
    if not hasattr(
        model,
        "feature_importances_"
    ):
        return

    importance = pd.DataFrame({
        "Feature":feature_names,
        "Importance":model.feature_importances_
    })
    importance.sort_values(
        by="Importance",
        ascending=False,
        inplace=True
    )
    plt.figure(
        figsize=(10,6)
    )

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        )
    )
    plt.close()
    importance.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

# RUN COMPLETE REGRESSION PIPELINE

def regression_pipeline(df):
    X_train,X_test,y_train,y_test,feature_names = prepare_regression_data(df)
    lr,lr_result,lr_time = linear_regression_model(
        X_train,
        X_test,
        y_train,
        y_test
    )
    dt,dt_result,dt_time = decision_tree_regressor(
        X_train,
        X_test,
        y_train,
        y_test
    )
    rf,rf_result,rf_time = random_forest_regressor(
        X_train,
        X_test,
        y_train,
        y_test
    )
    comparison = compare_regression_models(
        lr_result,
        dt_result,
        rf_result
    )
    save_best_regressor(
        comparison,
        lr,
        dt,
        rf
    )
    feature_importance(
        rf,
        feature_names
    )
    print("\nRegression Pipeline Completed")
    return comparison