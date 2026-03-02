"""
Hedonic pricing model — train, evaluate, save, and load.

Compares three regressors:
  1. Ridge Regression    (interpretable baseline)
  2. Random Forest       (handles non-linearity, robust to outliers)
  3. XGBoost             (typically highest accuracy)

Training strategy:
  1. Reserve a 15% holdout — never used during model selection.
  2. Select the best architecture via 5-fold cross-validation on the remaining 85%.
  3. Refit the winner on all training data, applying exponential recency weights
     (half-life = RECENCY_HALFLIFE_DAYS) when feat_days_since_sale is present.
  4. Evaluate final reported metrics on the reserved holdout and save it to disk
     so model_diagnostics.py shows out-of-sample figures.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.features import get_model_feature_cols, prepare_dataset

logger = logging.getLogger(__name__)

# Exponential decay half-life for recency weighting.
# A comp from RECENCY_HALFLIFE_DAYS ago gets weight 0.5; one from today gets 1.0.
RECENCY_HALFLIFE_DAYS = 180


def _build_candidates() -> dict:
    """Return candidate model pipelines keyed by name."""
    candidates = {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }
    if XGBOOST_AVAILABLE:
        candidates["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )),
        ])
    else:
        logger.warning("xgboost not installed — skipping XGBoost candidate.")

    return candidates


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute regression metrics on a held-out test set."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_pct": mape}


def train(
    df: pd.DataFrame,
    model_path: str = "models/hedonic_model.joblib",
    feature_path: str = "models/feature_cols.joblib",
    holdout_path: str = "models/holdout_data.joblib",
    test_size: float = 0.15,
    n_cv_folds: int = 5,
    random_state: int = 42,
    max_price: Optional[float] = None,
) -> tuple[object, list[str], dict]:
    """
    Train candidate models via k-fold CV, select the best, refit, evaluate on
    a reserved holdout, and save everything to disk.

    Strategy:
      1. Reserve ``test_size`` fraction as a final holdout — never used for
         model selection or fitting.
      2. Run ``n_cv_folds``-fold CV on the remaining data to compare architectures.
      3. Refit the winner on all training data, applying exponential recency
         weights from ``feat_days_since_sale`` when that column is present.
      4. Report final metrics on the holdout; save holdout data so
         model_diagnostics.py shows out-of-sample figures.

    Returns:
        best_model   - fitted sklearn Pipeline
        feature_cols - list of feature column names used
        results      - {model_name: metrics}
                       All candidates include CV_RMSE_mean and CV_RMSE_std.
                       The winner also includes RMSE, MAE, R2, MAPE_pct (holdout).
    """
    if max_price is not None:
        before_cap = len(df)
        df = df[df["price"] <= max_price].copy()
        dropped = before_cap - len(df)
        logger.info(
            f"Applying training price cap at ${max_price:,.0f} — "
            f"using {len(df):,} listings at or below the cap "
            f"(dropped {dropped:,} above-cap listings)."
        )
        if len(df) < 100:
            logger.warning(
                "Very few rows remain after applying the training price cap; "
                "model performance may be unstable."
            )

    # Drop rows with missing or implausible sqft.
    # Median-imputing missing sqft causes the model to treat sqft as uninformative
    # (the partial dependence curve goes flat), which is the single biggest driver
    # of poor model accuracy on individual homes.
    if "sqft" in df.columns:
        before_sqft = len(df)
        df = df[df["sqft"].notna() & (df["sqft"] > 200)].copy()
        dropped_sqft = before_sqft - len(df)
        if dropped_sqft:
            logger.info(
                f"Dropped {dropped_sqft:,} training rows with missing/implausible sqft "
                f"(remaining: {len(df):,})."
            )

    feature_cols = get_model_feature_cols(df)
    if not feature_cols:
        raise ValueError("No feature columns found. Run prepare_dataset() first.")

    X = df[feature_cols].values
    y = df["price"].values
    cities_all = df["city"].values if "city" in df.columns else np.full(len(df), "unknown")

    # ── Temporal recency weights ───────────────────────────────────────────────
    # Newer comps count more; weight decays exponentially with age.
    sample_weight = None
    if "feat_days_since_sale" in df.columns:
        days = df["feat_days_since_sale"].fillna(730).values.astype(float)
        lam = np.log(2) / RECENCY_HALFLIFE_DAYS
        raw_w = np.exp(-lam * days)
        sample_weight = raw_w / raw_w.mean()  # normalize so mean weight = 1.0
        logger.info(
            f"Temporal recency weights applied (half-life={RECENCY_HALFLIFE_DAYS}d): "
            f"weight range [{sample_weight.min():.3f}, {sample_weight.max():.3f}]"
        )

    # ── Reserve a final holdout (never used for model selection or fitting) ────
    # Stratify by price quintile so every price tier is proportionally represented
    # in both the training and holdout sets. A pure random split on small datasets
    # can leave the top price range under-represented in the holdout, making
    # metrics look better than they are for expensive homes.
    indices = np.arange(len(X))
    price_quintile = pd.qcut(y, q=5, labels=False, duplicates="drop")
    idx_train, idx_holdout = train_test_split(
        indices, test_size=test_size, random_state=random_state,
        stratify=price_quintile,
    )
    X_train, X_holdout = X[idx_train], X[idx_holdout]
    y_train, y_holdout = y[idx_train], y[idx_holdout]
    cities_holdout = cities_all[idx_holdout]
    w_train = sample_weight[idx_train] if sample_weight is not None else None

    candidates = _build_candidates()
    results: dict = {}

    # ── k-fold CV for model architecture selection ────────────────────────────
    logger.info(
        f"Comparing models via {n_cv_folds}-fold CV "
        f"({len(X_train):,} training rows)…"
    )
    for name, pipeline in candidates.items():
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=n_cv_folds,
            scoring="neg_root_mean_squared_error",
        )
        cv_rmse_mean = float(-cv_scores.mean())
        cv_rmse_std  = float(cv_scores.std())
        results[name] = {"CV_RMSE_mean": cv_rmse_mean, "CV_RMSE_std": cv_rmse_std}
        logger.info(
            f"  {name}: CV RMSE = ${cv_rmse_mean:,.0f} ± ${cv_rmse_std:,.0f}"
        )

    # ── Pick best by mean CV RMSE ──────────────────────────────────────────────
    best_name = min(results, key=lambda n: results[n]["CV_RMSE_mean"])
    best_model = candidates[best_name]
    logger.info(
        f"\nBest model: {best_name} "
        f"(CV RMSE=${results[best_name]['CV_RMSE_mean']:,.0f})"
    )

    # ── Refit winner on full training set (with recency weights if available) ─
    logger.info(f"Refitting {best_name} on {len(X_train):,} training rows…")
    if w_train is not None:
        best_model.fit(X_train, y_train, model__sample_weight=w_train)
    else:
        best_model.fit(X_train, y_train)

    # ── Evaluate on the reserved holdout ─────────────────────────────────────
    holdout_metrics = evaluate_model(best_model, X_holdout, y_holdout)
    results[best_name].update(holdout_metrics)
    logger.info(
        f"Holdout test ({len(X_holdout):,} rows): "
        f"RMSE=${holdout_metrics['RMSE']:,.0f}  MAE=${holdout_metrics['MAE']:,.0f}  "
        f"R²={holdout_metrics['R2']:.3f}  MAPE={holdout_metrics['MAPE_pct']:.1f}%"
    )

    # ── Save model, feature list, and holdout data ────────────────────────────
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    joblib.dump(feature_cols, feature_path)
    joblib.dump(
        {"X_test": X_holdout, "y_test": y_holdout, "cities": cities_holdout},
        holdout_path,
    )
    logger.info(f"Model saved → {model_path}")
    logger.info(f"Holdout data saved → {holdout_path}")

    return best_model, feature_cols, results


def load_model(
    model_path: str = "models/hedonic_model.joblib",
    feature_path: str = "models/feature_cols.joblib",
) -> tuple[object, list[str]]:
    """Load a previously saved model and its feature column list."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Run train() first."
        )
    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)
    return model, feature_cols


def predict(model, feature_cols: list[str], df: pd.DataFrame) -> np.ndarray:
    """
    Run inference on a prepared DataFrame.
    Missing feature columns are filled with 0 (e.g. unseen zip codes).
    """
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].values
    return model.predict(X)
