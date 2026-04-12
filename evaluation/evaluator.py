import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray = None):
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1_score":  f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_proba) if y_proba is not None else None,
    }
    return metrics

def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Disease", "Heart Disease"]
    df_cm  = pd.DataFrame(cm, index=labels, columns=labels)
 
    print("\n  Confusion Matrix:")
    print("  " + "─" * 38)
    print(f"  {'':18s} {'Predicted':>18s}")
    print(f"  {'Actual':<18s} {'No Disease':>10s} {'Heart Disease':>10s}")
    print("  " + "─" * 38)
    for row_label, row in zip(labels, cm):
        print(f"  {row_label:<18s} {row[0]:>10d} {row[1]:>10d}")
    print("  " + "─" * 38)
    tn, fp, fn, tp = cm.ravel()
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    report = classification_report(
        y_true, y_pred,
        target_names=["No Disease", "Heart Disease"]
    )
    print("\n  Classification Report:")
    for line in report.splitlines():
        print("  " + line)


def compare_models(results: dict):
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update({k: (round(v, 4) if v is not None else "N/A")
                    for k, v in metrics.items()})
        rows.append(row)
 
    df = pd.DataFrame(rows).set_index("model")
 
    # Sort by f1_score descending where possible
    df = df.sort_values("f1_score", ascending=False)
    return df
        
def print_metrics(name, metrics, params=None):
    print(f"\n{'─'*52}")
    print(f"  MODEL : {name}")
    if params:
        print(f"  Params: {params}")
    print(f"{'─'*52}")
    for k in ["accuracy","precision","recall","f1_score","roc_auc"]:
        print(f"  {k.capitalize():<12}: {metrics[k]:.4f}")