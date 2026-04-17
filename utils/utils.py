import os
import random

from datetime import datetime

import numpy as np
import pandas as pd
import pathlib


def print_section(title: str):
    width = 52
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def save_results(
    df: pd.DataFrame,
    filename: str = "results",
    output_dir: str = "C:/Users/anshu/Jupiter_Learning_phase/lgrs/data",
):
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(output_dir, f"{filename}_{timestamp}.csv")
    df.to_csv(full_path)
    return full_path


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    print(f"[OK] Random seed set to {seed}")
