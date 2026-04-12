import time
import numpy as np
 
from utils.utils        import print_section, save_results, set_seed
from src.data_loader  import load_data, split_target_features
from src.preprocessing import splitting, scale_features, validate_data
from model_training.trainer      import (train_scratch_model, train_all_penalties,
                              run_cross_validation, run_grid_search)
from evaluation.evaluator    import (compute_metrics, print_confusion_matrix,
                              print_classification_report, compare_models, print_metrics
                              )

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH      = 'C:/Users/anshu/Jupiter_Learning_phase/lgrs/data/heart.csv'
TARGET_COL     = "target"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
CV_FOLDS       = 5
LR_SCRATCH     = 0.01
ITERS_SCRATCH  = 1000

def main():
    pipeline_start = time.time()
    set_seed(RANDOM_STATE)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────
    print_section("1. LOADING DATA")
    df = load_data(DATA_PATH)
    validate_data(df)

# ── 2. PREPROCESS ─────────────────────────────────────────────────────────
    print_section("2. PREPROCESSING")

    # Separate features and target
    X, y = split_target_features(df, target_col=TARGET_COL)

    # Stratified train/test split — preserves class ratio in both sets
    X_train, X_test, y_train, y_test = splitting(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
 
    # Scale: fit on train ONLY to prevent data leakage into test
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
 
    # Convert targets to numpy arrays (required by scratch model)
    y_train_np = y_train.to_numpy()
    y_test_np  = y_test.to_numpy()

# ── 3. FROM-SCRATCH MODEL ─────────────────────────────────────────────────
    print_section("3. FROM-SCRATCH LOGISTIC REGRESSION  (NumPy only)")
    t0 = time.time()
    scratch_model = train_scratch_model(
        X_train_sc, y_train_np,
        learning_rate=LR_SCRATCH,
        n_iterations=ITERS_SCRATCH,
    )
    print(f"  Training time : {time.time() - t0:.2f}s")
    
    # Evaluate on test set
    scratch_pred    = scratch_model.predict(X_test_sc)
    scratch_proba = scratch_model.predict_proba(X_test_sc)
    if scratch_proba.ndim == 2:
        scratch_proba = scratch_proba[:, 1]
    scratch_metrics = compute_metrics(y_test_np, scratch_pred, scratch_proba)
 
    print_confusion_matrix(y_test_np, scratch_pred)
    print_classification_report(y_test_np, scratch_pred)
    print_metrics("Scratch Logistic Regression", scratch_metrics)

# ── 4. SKLEARN PENALTY VARIANTS ───────────────────────────────────────────
    print_section("4. SKLEARN MODELS  —  L1 / L2 / ElasticNet / None")
 
    t0 = time.time()
    sklearn_models = train_all_penalties(X_train_sc, y_train_np)
    print(f"\n  All variants trained in {time.time() - t0:.2f}s")

# ── 5. CROSS-VALIDATION ───────────────────────────────────────────────────
    print_section("5. CROSS-VALIDATION  (Stratified 5-Fold on Train Set)")
 
    print(f"\n  {'Model':<14} {'Mean Acc':>10} {'Std':>8}  Fold Scores")
    print("  " + "─" * 62)

    cv_results = {}
    for name, model in sklearn_models.items():
        cv = run_cross_validation(
            model, X_train_sc, y_train_np,
            cv=CV_FOLDS, scoring="accuracy"
        )
        cv_results[name] = cv
        fold_scores = "  ".join(f"{s:.3f}" for s in cv["scores"])
        print(f"  {name:<14} {cv['mean']:>10.4f} {cv['std']:>8.4f}  [{fold_scores}]")

    # Identify the strongest cross-val model
    best_cv_name = max(cv_results, key=lambda k: cv_results[k]["mean"])
    print(f"\n  Best cross-val model → {best_cv_name}  "
          f"(mean acc = {cv_results[best_cv_name]['mean']:.4f})")   
# ── 6. GRID SEARCH CV ─────────────────────────────────────────────────────
    print_section("6. GRID SEARCH CV  (exhaustive hyperparameter tuning)")
 
    t0   = time.time()
    grid = run_grid_search(X_train_sc, y_train_np, cv=CV_FOLDS)
    print(f"  Completed in   : {time.time() - t0:.2f}s")
    print(f"  Best params    : {grid.best_params_}")
    print(f"  Best CV score  : {grid.best_score_:.4f}")
    print(f"  Total combos   : {len(grid.cv_results_['mean_test_score'])}")
 
    best_model = grid.best_estimator_

# ── 7. TEST-SET EVALUATION ────────────────────────────────────────────────
    print_section("7. TEST-SET EVALUATION  (unseen data)")
 
    # Collect results for all models
    all_results = {"Scratch (NumPy)": scratch_metrics}

    # Evaluate each sklearn penalty variant
    for name, model in sklearn_models.items():
        pred    = model.predict(X_test_sc)
        proba   = model.predict_proba(X_test_sc)[:, 1]
        metrics = compute_metrics(y_test_np, pred, proba)
        all_results[name] = metrics
        print_confusion_matrix(y_test_np, pred)
        print_classification_report(y_test_np, pred)
        print_metrics(name, metrics, model.get_params())
 
    # Evaluate GridSearch best model
    best_pred    = best_model.predict(X_test_sc)
    best_proba   = best_model.predict_proba(X_test_sc)[:, 1]
    best_metrics = compute_metrics(y_test_np, best_pred, best_proba)
    all_results["GridSearch Best"] = best_metrics
    print_confusion_matrix(y_test_np, pred)
    print_classification_report(y_test_np, pred)
    print_metrics("Best GridSearch Model", best_metrics, grid.best_params_)

 
    

# ── 8. COMPARE & SAVE ─────────────────────────────────────────────────────
    print_section("8. FINAL MODEL COMPARISON  (sorted by F1-Score ↓)")
 
    summary_df = compare_models(all_results)
    print("\n" + summary_df.to_string())

    # Announce the overall winner
    winner     = summary_df.index[0]
    winner_f1  = summary_df.loc[winner, "f1_score"]
    winner_acc = summary_df.loc[winner, "accuracy"]
    print(f"\n  Best model  : {winner}")
    print(f"  F1-Score    : {winner_f1}")
    print(f"  Accuracy    : {winner_acc}")

    # Save comparison table to CSV with timestamp
    csv_path = save_results(summary_df, filename="model_comparison")
    print(f"\n  Results saved → {csv_path}")

    # ── DONE ──────────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    print_section(f"PIPELINE COMPLETE  —  Total time: {total:.1f}s")
 


if __name__ == "__main__":
    main()
