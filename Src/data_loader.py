import sys
import pandas as pd
import pathlib


def load_data(path: str) -> pd.DataFrame:
    if not pathlib.Path(path).exists():
        print(f"Error file not found in : '{path}'")
        sys.exit(1)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("[Error] could not read file")
        sys.exit(1)
    else:
        print(
            f"[OK] loaded {path} -> {df.shape[0]} rows, {df.shape[1]} columns"
        )
        return df


def split_target_features(
    df: pd.DataFrame, target_col: str = "target"
) -> pd.DataFrame:
    if target_col not in df.columns:
        msg = f"Target column '{target_col}' not found. Available: {list(df.columns)}"
        raise ValueError(msg)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"[Ok] Feature matrix X : {X.shape}")
    print(f"[OK] Target vector y  : {y.shape} Classes {sorted(y.unique())} ")

    return X, y
