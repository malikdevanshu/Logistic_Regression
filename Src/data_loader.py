"""
data_loader.py
responsible for loading data and perform
initial validation checks.
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



def validate_data(df):
     df1 = df.drop_duplicates()
     print(f"Number of rows before: {df.shape[0]}\nAfter removing duplicates rows are: {df1.shape[0]}")
     
     null_counts = df1.isnull().sum()
     total_nulls = null_counts.sum()
     if total_nulls == 0:
        print(f"  Missing values   : None ✓")
     else:
        print(f"  Missing values   : {total_nulls} total")
        print(null_counts[null_counts > 0].to_string())


     print("\n  Column Dtypes:")
     print(df1.dtypes.to_string())

     if "target" in df1.columns:
        counts = df1["target"].value_counts().sort_index()
        print(f"\n  Target distribution:")
        print(f"    No Disease  (0) : {counts.get(0, 0)} samples")
        print(f"    Heart Disease(1): {counts.get(1, 0)} samples")
     
     return df1    


def split_target_features(df:pd.DataFrame, target_col:str="target"):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"[Ok] Feature matrix X : {X.shape}")
    print(f"[OK] Target vector y  : {y.shape} Classes {sorted(y.unique())} ")

    return X, y
