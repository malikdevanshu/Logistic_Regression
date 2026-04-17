import sys
import pandas as pd
import pathlib


def load_data(path: str = None):
    if not pathlib.Path(path).exists():
        print(f"Error file not found in : '{path}'")
        sys.exit(1)
    try:
        df = pd.read_csv(path)
        print(
            f"[OK] loaded {path} -> {df.shape[0]} rows, {df.shape[1]} columns"
        )
        return df
    except Exception as e:
        print(f"[Error] could not read file:{e}")
        sys.exit(1)


def split_target_features(df: pd.DataFrame, target_col: str = "target"):
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"[Ok] Feature matrix X : {X.shape}")
    print(f"[OK] Target vector y  : {y.shape} Classes {sorted(y.unique())} ")

    return X, y
