"""
01_train_all_models.py
======================
Train, tune, and evaluate all five ML and statistical models for
district-level maize yield prediction in Nepal.

Models:
  1. Random Forest (RF)
  2. Gradient Boosting (GB)
  3. Support Vector Regression (SVR - RBF kernel)
  4. Lasso Regression
  5. OLS Multiple Regression

Evaluation:
  - Temporal hold-out split: train 1990–2015, test 2016–2022
  - Metrics: R², RMSE, MAE, NSE, PBIAS
  - 5-fold cross-validation for hyperparameter tuning

Output:
  outputs/model_performance_metrics.csv
  outputs/predictions_test_set.csv
  outputs/best_model_rf.pkl

Authors: Lokendra Paudel et al. (2025)
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
PROC_DIR   = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
TRAIN_YEARS  = range(1990, 2016)   # 80% training set
TEST_YEARS   = range(2016, 2023)   # 20% temporal hold-out
TARGET_COL   = "yield_t_ha"
RANDOM_STATE = 42

# Feature sets
FEATURES_TREE = [                  # Tree models: all features
    "tmax_mean", "tmin_mean", "rain_annual", "spi3_mean",
    "sm_mean", "ndvi_gs_mean", "evi_gs_mean",
    "log_area_ha", "heat_stress_idx",
]
FEATURES_LINEAR = [                # Linear models: curated (no VIF collinearity)
    "tmax_mean", "rain_annual", "spi3_mean",
    "sm_mean", "ndvi_gs_mean",
    "log_area_ha", "heat_stress_idx",
]


# ── Metric functions ──────────────────────────────────────────────────────────
def nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency."""
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)


def pbias(y_true, y_pred):
    """Percent Bias (%)."""
    return 100 * np.sum(y_true - y_pred) / np.sum(y_true)


def compute_metrics(y_true, y_pred, split_name: str, model_name: str) -> dict:
    return {
        "Model":   model_name,
        "Dataset": split_name,
        "R2":      round(r2_score(y_true, y_pred), 4),
        "RMSE":    round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE":     round(mean_absolute_error(y_true, y_pred), 4),
        "NSE":     round(nse(y_true, y_pred), 4),
        "PBIAS":   round(pbias(y_true, y_pred), 2),
    }


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    # Try real data first, fall back to synthetic demo
    for fname in ["nepal_maize_panel_1990_2022.csv",
                  "nepal_maize_panel_SYNTHETIC_demo.csv"]:
        fpath = PROC_DIR / fname
        if fpath.exists():
            print(f"  Loading: {fname}")
            return pd.read_csv(fpath)
    raise FileNotFoundError(
        "No processed data found. Run code/01_data_prep/01_merge_datasets.py first."
    )


def prepare_splits(df, feature_cols):
    """Temporal train/test split and feature extraction."""
    df_train = df[df["year"].isin(TRAIN_YEARS)].dropna(subset=feature_cols + [TARGET_COL])
    df_test  = df[df["year"].isin(TEST_YEARS )].dropna(subset=feature_cols + [TARGET_COL])

    X_train = df_train[feature_cols].values
    y_train = df_train[TARGET_COL].values
    X_test  = df_test[feature_cols].values
    y_test  = df_test[TARGET_COL].values

    print(f"  Train: {X_train.shape[0]} obs | Test: {X_test.shape[0]} obs")
    return X_train, y_train, X_test, y_test, df_test


# ── Model definitions and hyperparameter grids ────────────────────────────────
def get_models():
    return [
        {
            "name": "Random Forest",
            "features": "tree",
            "model": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": {
                "n_estimators": [200, 400],
                "max_depth":    [8, 12, None],
                "max_features": ["sqrt", 0.6],
                "min_samples_split": [2, 5],
            },
        },
        {
            "name": "Gradient Boosting",
            "features": "tree",
            "model": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators":  [200, 300],
                "learning_rate": [0.05, 0.1],
                "max_depth":     [3, 5],
                "subsample":     [0.8, 1.0],
            },
        },
        {
            "name": "Support Vector Regression",
            "features": "tree",   # SVR handles its own scaling internally via Pipeline
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("svr",    SVR(kernel="rbf")),
            ]),
            "param_grid": {
                "svr__C":     [1, 10, 100],
                "svr__gamma": ["scale", 0.01, 0.1],
                "svr__epsilon":[0.05, 0.1],
            },
        },
        {
            "name": "Lasso Regression",
            "features": "linear",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("lasso",  Lasso(max_iter=5000, random_state=RANDOM_STATE)),
            ]),
            "param_grid": {
                "lasso__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            },
        },
        {
            "name": "OLS Multiple Regression",
            "features": "linear",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("ols",    LinearRegression()),
            ]),
            "param_grid": {},   # No hyperparameters
        },
    ]


# ── Main training loop ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Nepal Maize Yield — ML Model Training & Evaluation")
    print("=" * 60)

    df = load_data()

    # Drop province dummy cols for simpler feature alignment
    prov_cols = [c for c in df.columns if c.startswith("prov_")]

    # Ensure derived features exist
    if "log_area_ha" not in df.columns:
        df["log_area_ha"] = np.log1p(df.get("area_ha", 8000))
    if "heat_stress_idx" not in df.columns:
        df["heat_stress_idx"] = np.maximum(0, df.get("tmax_mean", 28) - 35.0)
    if "evi_gs_mean" not in df.columns:
        df["evi_gs_mean"] = df.get("ndvi_gs_mean", np.nan) * 0.85

    # Build feature sets (only use columns that exist in df)
    feat_tree   = [f for f in FEATURES_TREE   if f in df.columns]
    feat_linear = [f for f in FEATURES_LINEAR if f in df.columns]

    X_tr_t, y_tr, X_te_t, y_te, df_test_t = prepare_splits(df, feat_tree)
    X_tr_l, _,    X_te_l, _,   _           = prepare_splits(df, feat_linear)

    all_metrics    = []
    all_preds      = pd.DataFrame({"year": df_test_t["year"].values,
                                   "district": df_test_t.get("district",
                                               pd.Series(["unknown"] * len(y_te))).values,
                                   "y_true": y_te})
    best_rf_model  = None

    for cfg in get_models():
        mname = cfg["name"]
        X_tr  = X_tr_t if cfg["features"] == "tree" else X_tr_l
        X_te  = X_te_t if cfg["features"] == "tree" else X_te_l
        y_true_te = y_te

        print(f"\n  ── {mname} ──")

        if cfg["param_grid"]:
            gs = GridSearchCV(
                cfg["model"], cfg["param_grid"],
                cv=5, scoring="r2", n_jobs=-1, verbose=0
            )
            gs.fit(X_tr, y_tr)
            best_model = gs.best_estimator_
            print(f"    Best params: {gs.best_params_}")
        else:
            best_model = cfg["model"]
            best_model.fit(X_tr, y_tr)

        # Predictions
        y_pred_tr = best_model.predict(X_tr)
        y_pred_te = best_model.predict(X_te if len(X_te) == len(y_true_te) else X_te_t)

        # Metrics
        all_metrics.append(compute_metrics(y_tr,      y_pred_tr, "Train", mname))
        all_metrics.append(compute_metrics(y_true_te, y_pred_te, "Test",  mname))

        # Store test predictions
        all_preds[f"y_pred_{mname.replace(' ', '_')}"] = y_pred_te

        print(f"    Train R² = {all_metrics[-2]['R2']:.4f}  |  "
              f"Test R² = {all_metrics[-1]['R2']:.4f}  |  "
              f"Test RMSE = {all_metrics[-1]['RMSE']:.4f}")

        # Keep best RF for SHAP
        if mname == "Random Forest":
            best_rf_model = best_model

    # ── Save results ───────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = OUTPUT_DIR / "model_performance_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✅ Saved metrics → {metrics_path}")

    preds_path = OUTPUT_DIR / "predictions_test_set.csv"
    all_preds.to_csv(preds_path, index=False)
    print(f"✅ Saved predictions → {preds_path}")

    if best_rf_model is not None:
        rf_path = OUTPUT_DIR / "best_model_rf.pkl"
        with open(rf_path, "wb") as f:
            pickle.dump({"model": best_rf_model, "features": feat_tree}, f)
        print(f"✅ Saved RF model → {rf_path}")

    print("\n" + "=" * 60)
    print("  Final Model Comparison (Test Set)")
    print("=" * 60)
    test_metrics = metrics_df[metrics_df["Dataset"] == "Test"].sort_values("R2", ascending=False)
    print(test_metrics[["Model", "R2", "RMSE", "NSE", "PBIAS"]].to_string(index=False))


if __name__ == "__main__":
    main()
