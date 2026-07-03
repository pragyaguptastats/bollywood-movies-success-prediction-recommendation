import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    """
    Load dataset.
    """
    df = pd.read_csv(filepath)

    print("=" * 60)
    print("Dataset Loaded Successfully")
    print("=" * 60)
    print(df.shape)

    return df


def basic_information(df):

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())


def remove_duplicates(df):

    df = df.drop_duplicates()

    return df


def handle_missing_values(df):

    for col in df.columns:

        if df[col].dtype == "object":

            df[col].fillna(
                df[col].mode()[0],
                inplace=True
            )

        else:

            df[col].fillna(
                df[col].median(),
                inplace=True
            )

    return df


def encode_categorical(df):

    encoder = LabelEncoder()

    for col in df.select_dtypes(include=["object"]).columns:

        df[col] = encoder.fit_transform(df[col])

    return df