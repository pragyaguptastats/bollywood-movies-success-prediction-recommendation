import os
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

OUTPUT_DIR = "outputs/model_tuning"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

def cross_validation(model, X_train, y_train):

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    print("\n")
    print(type(model).__name__)
    print("Cross Validation Scores")
    print(scores)
    print()
    print(f"Average R² : {scores.mean():.4f}")

    result = pd.DataFrame({
        "Fold":range(1,6),
        "R2":scores
    })

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{type(model).__name__}_cross_validation.csv"
        ),
        index=False
    )
    return scores.mean()

def random_forest_tuning(X_train, y_train):

    print("\n")
    print("GRID SEARCH : RANDOM FOREST")
    parameters = {
        "n_estimators":[100,200,300],
        "max_depth":[8,10,12,None],
        "min_samples_split":[2,5,10]
    }

    rf = RandomForestRegressor(
        random_state=42
    )
    grid = GridSearchCV(
        estimator=rf,
        param_grid=parameters,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(
        X_train,
        y_train
    )
    print()
    print("Best Parameters")
    print(grid.best_params_)
    print()
    print("Best Score")
    print(grid.best_score_)

    joblib.dump(
        grid.best_estimator_,
        "models/tuned_random_forest.pkl"
    )
    result = pd.DataFrame(
        [grid.best_params_]
    )

    result["Best Score"] = grid.best_score_
    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "random_forest_best_parameters.csv"
        ),
        index=False
    )

    return grid.best_estimator_

# DECISION TREE HYPERPARAMETER TUNING
def decision_tree_tuning(X_train, y_train):

    print("\n")
    print("=" * 60)
    print("GRID SEARCH : DECISION TREE")
    print("=" * 60)

    parameters = {
        "max_depth": [5, 8, 10, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    dt = DecisionTreeRegressor(
        random_state=42
    )

    grid = GridSearchCV(
        estimator=dt,
        param_grid=parameters,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )
    grid.fit(
        X_train,
        y_train
    )
    print("\nBest Parameters")
    print(grid.best_params_)

    print("\nBest Score")
    print(grid.best_score_)

    joblib.dump(
        grid.best_estimator_,
        "models/tuned_decision_tree.pkl"
    )
    result = pd.DataFrame(
        [grid.best_params_]
    )
    result["Best Score"] = grid.best_score_

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "decision_tree_best_parameters.csv"
        ),
        index=False
    )
    return grid.best_estimator_

# LINEAR REGRESSION CROSS VALIDATION

def linear_regression_validation(
        X_train,
        y_train
):
    print("\n")
    print("LINEAR REGRESSION VALIDATION")
    model = LinearRegression()
    score = cross_validation(
        model,
        X_train,
        y_train
    )
    return model, score

# EVALUATE MODEL

def evaluate_model(
        model,
        X_test,
        y_test,
        model_name

):
    prediction = model.predict(
        X_test
    )
    mse = mean_squared_error(
        y_test,
        prediction
    )
    r2 = r2_score(
        y_test,
        prediction
    )
    print()
    print(model_name)
    print(f"MSE : {mse:.2f}")
    print(f"R²  : {r2:.4f}")
    return [
        model_name,
        mse,
        r2
    ]

# MODEL COMPARISON

def compare_models(results):
    comparison = pd.DataFrame(
        results,
        columns=[
            "Model",
            "MSE",
            "R2 Score"
        ]
    )

    comparison.sort_values(
        by="R2 Score",
        ascending=False,
        inplace=True
    )
    print("\n")
    print("FINAL MODEL COMPARISON")
    print(comparison)
    comparison.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "final_model_comparison.csv"
        ),
        index=False
    )
    return comparison

# SAVE BEST MODEL
def save_best_model(
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
        "models/final_best_model.pkl"
    )
    print("\nBest Tuned Model Saved Successfully")
    print(best)

# COMPLETE MODEL TUNING PIPELINE

def model_tuning_pipeline(
        X_train,
        X_test,
        y_train,
        y_test
):

    print("\n")
    print("MODEL TUNING & CROSS VALIDATION")

    lr, cv_score = linear_regression_validation(
        X_train,
        y_train
    )

    lr.fit(
        X_train,
        y_train
    )

    lr_result = evaluate_model(
        lr,
        X_test,
        y_test,
        "Linear Regression"
    )

    dt = decision_tree_tuning(
        X_train,
        y_train
    )

    dt_result = evaluate_model(
        dt,
        X_test,
        y_test,
        "Decision Tree"
    )

    rf = random_forest_tuning(
        X_train,
        y_train
    )

    rf_result = evaluate_model(
        rf,
        X_test,
        y_test,
        "Random Forest"
    )
    comparison = compare_models(
        [
            lr_result,
            dt_result,
            rf_result
        ]
    )

    save_best_model(
        comparison,
        lr,
        dt,
        rf
    )
    print("\n")
    print("MODEL TUNING COMPLETED")
    return comparison