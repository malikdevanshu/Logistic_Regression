import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn_train.model import Logistic_regression_scratch, get_model

def train_scratch_model(X_train:np.ndarray, y_train:np.ndarray,
                        learning_rate:float = 0.01, n_iterations:int = 1000):
    print(f"  Training scratch model  (lr={learning_rate}, iters={n_iterations}) ...")
    model = Logistic_regression_scratch(learning_rate=learning_rate,
                                        n_iterations=n_iterations)
    model.fit(X_train, y_train)
    return model

def train_all_penalties(X_train: np.ndarray, y_train: np.ndarray):
    penalties = {
        "L1":         get_model("l1"),
        "L2":         get_model("l2"),
        "ElasticNet": get_model("elasticnet", l1_ratio=0.5),
        "None":       get_model("none"),
    }
    
    fitted_models = {}
    for name, model in penalties.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        print(f"  [OK] {name:12s} model trained")

    return fitted_models

def run_cross_validation(model: Logistic_regression_scratch,
                         X: np.ndarray, y: np.ndarray,
                         cv: int = 5,
                         scoring: str = "accuracy"):
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    return {
        "scores": scores,
        "mean":   scores.mean(),
        "std":    scores.std(),
    }

def run_grid_search(X_train: np.ndarray, y_train: np.ndarray,
                    cv: int = 5):
    param_grid = [
        # L1
        {"penalty": ["l1"], "C": [0.01, 0.1, 1, 10, 100],
         "solver": ["liblinear"]},
        # L2
        {"penalty": ["l2"], "C": [0.01, 0.1, 1, 10, 100],
         "solver": ["lbfgs"]},
        # ElasticNet
        {"penalty": ["elasticnet"], "C": [0.01, 0.1, 1, 10],
         "solver": ["saga"], "l1_ratio": [0.2, 0.5, 0.8]},
    ]
    base_estimator = LogisticRegression(max_iter=1000, random_state=42)
 
    grid = GridSearchCV(
        estimator  = base_estimator,
        param_grid = param_grid,
        cv         = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring    = "accuracy",
        n_jobs     = -1,       # use all CPU cores
        verbose    = 0,
    )

    print("  Running GridSearchCV — this may take a moment ...")
    grid.fit(X_train, y_train)
    return grid

