"""
Hedonic pricing model — train, evaluate, save, and load.

Compares three regressors:
  1. Linear Regression   (interpretable baseline)
  2. Random Forest       (handles non-linearity, robust to outliers)
  3. XGBoost             (typically highest accuracy)

The winning model (lowest RMSE on test set) is saved via joblib.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LinearRegression, Ridge
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


def _build_candidates() -> dict:
    """Return candidate model pipelines keyed by name."""
    candidates = {
        "LinearRegression": Pipeline([
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
    test_size: float = 0.2,
    random_state: int = 42,
    max_price: Optional[float] = None,
) -> tuple[object, list[str], dict]:
    """
    Train candidate models, select the best by RMSE, save it.

    Returns:
        best_model  - fitted sklearn Pipeline
        feature_cols - list of feature column names used
        results      - dict of {model_name: metrics}
    Args:
        df: preprocessed listings DataFrame (typically sold-only, with engineered features)
        model_path: where to save the trained model pipeline
        feature_path: where to save the list of feature column names
        test_size: fraction of data reserved for the test set
        random_state: RNG seed for train/test split
        max_price: optional upper bound on price for training data
            (e.g. 1_000_000 to focus the model on sub-$1M homes)

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

    feature_cols = get_model_feature_cols(df)
    if not feature_cols:
        raise ValueError("No feature columns found. Run prepare_dataset() first.")

    X = df[feature_cols].values
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    candidates = _build_candidates()
    results = {}

    for name, pipeline in candidates.items():
        logger.info(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics
        logger.info(
            f"  {name}: RMSE=${metrics['RMSE']:,.0f}  MAE=${metrics['MAE']:,.0f}  "
            f"R²={metrics['R2']:.3f}  MAPE={metrics['MAPE_pct']:.1f}%"
        )

    # Pick best by RMSE
    best_name = min(results, key=lambda n: results[n]["RMSE"])
    best_model = candidates[best_name]
    logger.info(f"\nBest model: {best_name} (RMSE=${results[best_name]['RMSE']:,.0f})")

    # Save model and feature column list
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    joblib.dump(feature_cols, feature_path)
    logger.info(f"Model saved to {model_path}")

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
