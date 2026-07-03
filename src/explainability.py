import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance

# OUTPUT DIRECTORY

OUTPUT_DIR = "outputs/explainability"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# FEATURE IMPORTANCE

def feature_importance(
        model,
        feature_names
):

    print("\n")
    print("FEATURE IMPORTANCE")

    if not hasattr(model, "feature_importances_"):
        print("Selected model does not support feature importance.")
        return
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })
    importance.sort_values(
        by="Importance",
        ascending=False,
        inplace=True
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
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        )
    )
    plt.close()
    print("\nFeature Importance Saved Successfully.")

# PERMUTATION IMPORTANCE
def permutation_feature_importance(
        model,
        X_test,
        y_test,
        feature_names
):

    print("\n")
    print("PERMUTATION FEATURE IMPORTANCE")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="r2"
    )
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": result.importances_mean,
        "Std": result.importances_std
    })
    importance.sort_values(
        by="Importance",
        ascending=False,
        inplace=True
    )
    print(importance)
    importance.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "permutation_importance.csv"
        ),
        index=False
    )
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature"
    )
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "permutation_importance.png"
        )
    )
    plt.close()
    print("\nPermutation Importance Saved Successfully.")

# COMPLETE EXPLAINABILITY PIPELINE

def explainability_pipeline(
        model,
        X_test,
        y_test,
        feature_names
):
    print("\n")
    print("MODEL EXPLAINABILITY")

    feature_importance(
        model,
        feature_names
    )
    permutation_feature_importance(
        model,
        X_test,
        y_test,
        feature_names
    )
    print("\n")
    print("EXPLAINABILITY COMPLETED")