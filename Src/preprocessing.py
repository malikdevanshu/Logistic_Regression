"""preprocessing.py
Handles all data preprocessing steps:
scaling the data
Spliting the data into train test
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def validate_data(df: pd.Dataframe):
    df1 = df.drop_duplicates()
    print(
        f"Number of rows before: {df.shape[0]}\nAfter removing duplicates rows are: {df1.shape[0]}"
    )

    null_counts = df1.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls == 0:
        print("  Missing values   : None ✓")
    else:
        print(f"  Missing values   : {total_nulls} total")
        print(null_counts[null_counts > 0].to_string())

    print("\n  Column Dtypes:")
    print(df1.dtypes.to_string())

    if "target" in df1.columns:
        counts = df1["target"].value_counts().sort_index()
        print("\n  Target distribution:")
        print(f"    No Disease  (0) : {counts.get(0, 0)} samples")
        print(f"    Heart Disease(1): {counts.get(1, 0)} samples")

    return df1


def splitting(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print("\n Shape after Splitting")
    print(
        f"\nX_train:{X_train.shape} | X_test:{X_test.shape} \n y_train:{y_train.shape} | y_test:{y_test.shape}"
    )
    return (X_train, X_test, y_train, y_test)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    model_scal = StandardScaler()
    X_train_scaled = model_scal.fit_transform(X_train)
    X_test_scaled = model_scal.transform(X_test)
    return (X_train_scaled, X_test_scaled, model_scal)
