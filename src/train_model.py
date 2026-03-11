"""
train_model.py — Model training, evaluation, and serialization
===============================================================
Part of the ML lifecycle for Used Car Price Prediction.

Steps:
  1. Load & preprocess data
  2. Train/test split (80/20)
  3. Train multiple models (RF, XGBoost, GradientBoosting)
  4. Compare and pick the best model
  5. Evaluate with RMSE, MAE, R² metrics
  6. Save trained model + feature names to .pkl
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import load_data, preprocess


def train_and_evaluate():
    """Full training pipeline with model comparison."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "archive", "car data.csv")
    df = load_data(data_path)

    X, y, feature_names = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[Train] Train set: {X_train.shape[0]} samples")
    print(f"[Train] Test set:  {X_test.shape[0]} samples")

    models = {}

    print("\n[Train] Starting RandomizedSearchCV hyperparameter tuning (this may take a minute)...")

    rf_grid = {
        "n_estimators": list(range(500, 1000, 100)),
        "max_depth": list(range(4, 9, 4)),
        "min_samples_split": list(range(4, 9, 2)),
        "min_samples_leaf": [1, 2, 5, 7],
        "max_features": ["sqrt", "log2"]  # "auto" is deprecated in modern scikit-learn
    }
    rf = RandomForestRegressor(random_state=42)
    rf_rs = RandomizedSearchCV(estimator=rf, param_distributions=rf_grid, n_iter=10, cv=5, random_state=42, n_jobs=-1)
    rf_rs.fit(X_train, y_train)
    models["RandomForest"] = rf_rs.best_estimator_
    print(f"  [Tuned] RandomForest best params: {rf_rs.best_params_}")

    gb_grid = {
        "learning_rate": [0.001, 0.01, 0.1, 0.2],
        "n_estimators": list(range(500, 1000, 100)),
        "max_depth": list(range(4, 9, 4)),
        "min_samples_split": list(range(4, 9, 2)),
        "min_samples_leaf": [1, 2, 5, 7],
        "max_features": ["sqrt", "log2"]
    }
    gb = GradientBoostingRegressor(random_state=42)
    gb_rs = RandomizedSearchCV(estimator=gb, param_distributions=gb_grid, n_iter=10, cv=5, random_state=42, n_jobs=-1)
    gb_rs.fit(X_train, y_train)
    models["GradientBoosting"] = gb_rs.best_estimator_
    print(f"  [Tuned] GradientBoosting best params: {gb_rs.best_params_}")

    try:
        from xgboost import XGBRegressor
        xgb_grid = {
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "n_estimators": [300, 500, 700],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9]
        }
        xgb = XGBRegressor(random_state=42, verbosity=0)
        xgb_rs = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_grid, n_iter=10, cv=5, random_state=42, n_jobs=-1)
        xgb_rs.fit(X_train, y_train)
        models["XGBoost"] = xgb_rs.best_estimator_
        print(f"  [Tuned] XGBoost best params: {xgb_rs.best_params_}")
    except ImportError:
        print("  [Train] XGBoost not available, skipping.")

    best_model_name = None
    best_model = None
    best_r2 = -float("inf")
    results = {}

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON (5-fold Cross-Validation)")
    print("=" * 60)

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"rmse": rmse, "mae": mae, "r2": r2, "cv_r2": cv_mean}

        print(f"\n  {name}:")
        print(f"    CV R²  : {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"    Test R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name

    print(f"\n{'=' * 60}")
    print(f"  ★ BEST MODEL: {best_model_name} (R² = {best_r2:.4f})")
    print(f"{'=' * 60}")

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\n  FINAL METRICS ({best_model_name}):")
    print(f"  Train RMSE : {train_rmse:.4f} lakh")
    print(f"  Test  RMSE : {test_rmse:.4f} lakh")
    print(f"  Test  MAE  : {test_mae:.4f} lakh")
    print(f"  Test  R²   : {test_r2:.4f}")

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feat_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        print(f"\n[Train] Feature Importances:\n{feat_imp.to_string(index=False)}")
    else:
        feat_imp = None

    plots_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, y_pred_test, alpha=0.6, edgecolors="k", linewidth=0.5, c="#667eea")
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[0].set_xlabel("Actual Price (lakh)")
    axes[0].set_ylabel("Predicted Price (lakh)")
    axes[0].set_title(f"Actual vs Predicted — {best_model_name} (R² = {test_r2:.3f})")

    if feat_imp is not None:
        axes[1].barh(feat_imp["Feature"], feat_imp["Importance"], color="#764ba2")
        axes[1].set_xlabel("Importance")
        axes[1].set_title("Feature Importances")
        axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "evaluation_plots.png"), dpi=150)
    plt.close()
    print(f"\n[Train] Saved evaluation plots to models/evaluation_plots.png")

    model_path = os.path.join(plots_dir, "car_price_model.pkl")
    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": feature_names,
        "metrics": {
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2),
        },
        "all_results": results,
    }
    joblib.dump(artifact, model_path)
    print(f"[Train] Model saved to {model_path}")

    return artifact


if __name__ == "__main__":
    train_and_evaluate()
