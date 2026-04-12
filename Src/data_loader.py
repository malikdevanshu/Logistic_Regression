"""
data_loader.py
responsible for loading data and perform
feature-target split
"""
import pandas as pd
import os
import sys

def load_data(path):
    if not os.path.exists(path):
        print(f"Error file not found: '{path}'")
        sys.exit(1)
    try:
        df = pd.read_csv(path)
        print(f"[OK] loaded {path} -> {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"[Error] could not read file:{e}")
        sys.exit(1)   


def split_target_features(df:pd.DataFrame, target_col:str="target"):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"[Ok] Feature matrix X : {X.shape}")
    print(f"[OK] Target vector y  : {y.shape} Classes {sorted(y.unique())} ")

    return X, y
