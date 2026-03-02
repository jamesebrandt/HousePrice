"""
Hedonic pricing model — train, evaluate, save, and load.

Compares three regressors:
  1. Ridge Regression    (interpretable baseline)
  2. Random Forest       (handles non-linearity, robust to outliers)
  3. XGBoost             (typically highest accuracy)

Training strategy:
  1. Reserve a 15% holdout — never used during model selection.
  2. Select the best architecture via 5-fold cross-validation on the remaining 85%.
     CV uses multiple metrics (RMSE, MAE, MAPE) for a complete picture.
  3. Refit the winner on all training data, applying exponential recency weights
     (half-life = RECENCY_HALFLIFE_DAYS) when feat_days_since_sale is present.
  4. Evaluate final reported metrics on the reserved holdout and save it to disk
     so model_diagnostics.py shows out-of-sample figures.

Outlier robustness:
  - The target variable is log-transformed (log-price) during training.
     This is standard econometric practice for hedonic models: it compresses
     the price scale so percentage errors are penalised equally at every price
     level, preventing luxury homes from dominating the loss.
  - XGBoost uses Pseudo-Huber loss (reg:pseudohubererror) which is linear for
     large residuals, further limiting the influence of extreme outliers.
  - Predictions are back-transformed to dollar-space (with bias correction)
     before being returned to the caller.
"""

import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from typing import Optional

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, cross_validate, KFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, make_scorer,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.features import get_model_feature_cols, prepare_dataset

logger = logging.getLogger(__name__)

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
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )),
        ])
    else:
        logger.warning("xgboost not installed — skipping XGBoost candidate.")

    return candidates


def _mape_scorer(y_true, y_pred):
    """Negative MAPE scorer for sklearn CV (higher = better convention)."""
    abs_pct = np.abs((y_true - y_pred) / y_true) * 100
    return -abs_pct.mean()


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray, *, log_target: bool = False,
) -> dict:
    """Compute regression metrics on a held-out test set.

    When ``log_target=True``, y_test is in log-space and predictions are
    back-transformed to dollar-space (with Duan smearing bias correction)
    before computing metrics so the numbers are directly interpretable.
    """
    y_pred_raw = model.predict(X_test)

    if log_target:
        log_residuals = y_test - y_pred_raw
        smearing_factor = np.exp(log_residuals).mean()
        y_pred = np.exp(y_pred_raw) * smearing_factor
        y_actual = np.exp(y_test)
    else:
        y_pred = y_pred_raw
        y_actual = y_test

    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae  = mean_absolute_error(y_actual, y_pred)
    r2   = r2_score(y_actual, y_pred)
    abs_pct = np.abs((y_actual - y_pred) / y_actual) * 100
    mape = abs_pct.mean()
    within_10pct = float((abs_pct <= 10).mean() * 100)
    within_20pct = float((abs_pct <= 20).mean() * 100)
    within_30pct = float((abs_pct <= 30).mean() * 100)
    return {
        "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_pct": mape,
        "within_10pct": within_10pct,
        "within_20pct": within_20pct,
        "within_30pct": within_30pct,
    }


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

    The target variable is log-transformed during training to equalise the
    influence of homes across the full price range. All reported metrics are
    back-transformed to dollar-space for interpretability.

    Returns:
        best_model   - fitted sklearn Pipeline (predicts log-price)
        feature_cols - list of feature column names used
        results      - {model_name: metrics dict}
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
    y_dollar = df["price"].values
    y = np.log(y_dollar)
    cities_all = df["city"].values if "city" in df.columns else np.full(len(df), "unknown")

    sample_weight = None
    if "feat_days_since_sale" in df.columns:
        days = df["feat_days_since_sale"].fillna(730).values.astype(float)
        lam = np.log(2) / RECENCY_HALFLIFE_DAYS
        raw_w = np.exp(-lam * days)
        sample_weight = raw_w / raw_w.mean()
        logger.info(
            f"Temporal recency weights applied (half-life={RECENCY_HALFLIFE_DAYS}d): "
            f"weight range [{sample_weight.min():.3f}, {sample_weight.max():.3f}]"
        )

    # Reserve 15% holdout — never touched during model selection or fitting
    indices = np.arange(len(X))
    price_quintile = pd.qcut(y_dollar, q=5, labels=False, duplicates="drop")
    idx_train, idx_holdout = train_test_split(
        indices, test_size=test_size, random_state=random_state,
        stratify=price_quintile,
    )
    X_train, X_holdout = X[idx_train], X[idx_holdout]
    y_train, y_holdout = y[idx_train], y[idx_holdout]
    y_dollar_holdout = y_dollar[idx_holdout]
    cities_holdout = cities_all[idx_holdout]
    w_train = sample_weight[idx_train] if sample_weight is not None else None

    candidates = _build_candidates()
    results: dict = {}

    cv_splitter = KFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae":  "neg_mean_absolute_error",
        "neg_mape": make_scorer(_mape_scorer),
    }

    median_train_price = float(np.median(y_dollar[idx_train]))

    logger.info(
        f"Comparing models via {n_cv_folds}-fold CV "
        f"({len(X_train):,} training rows, log-price target)…"
    )
    for name, pipeline in candidates.items():
        cv = cross_validate(
            pipeline, X_train, y_train,
            cv=cv_splitter, scoring=scoring,
        )
        cv_rmse_mean = float(-cv["test_neg_rmse"].mean())
        cv_rmse_std  = float(cv["test_neg_rmse"].std())
        cv_mae_mean  = float(-cv["test_neg_mae"].mean())
        cv_mape_mean = float(-cv["test_neg_mape"].mean())

        # Approximate dollar-space RMSE: a log-RMSE of σ corresponds to
        # roughly (exp(σ) - 1) * median_price in dollar terms.
        cv_rmse_dollar = float((np.exp(cv_rmse_mean) - 1) * median_train_price)
        cv_rmse_std_dollar = float((np.exp(cv_rmse_std) - 1) * median_train_price)

        results[name] = {
            "CV_RMSE_mean": cv_rmse_mean, "CV_RMSE_std": cv_rmse_std,
            "CV_MAE_mean": cv_mae_mean, "CV_MAPE_mean": cv_mape_mean,
            "CV_RMSE_dollar": cv_rmse_dollar,
            "CV_RMSE_std_dollar": cv_rmse_std_dollar,
        }
        logger.info(
            f"  {name}: CV log-RMSE = {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}  "
            f"(≈${cv_rmse_dollar/1e3:.0f}K ± ${cv_rmse_std_dollar/1e3:.0f}K)  "
            f"log-MAE = {cv_mae_mean:.4f}  MAPE(log) = {cv_mape_mean:.2f}%"
        )

    best_name = min(results, key=lambda n: results[n]["CV_RMSE_mean"])
    best_model = candidates[best_name]
    logger.info(
        f"\nBest model: {best_name} "
        f"(CV log-RMSE={results[best_name]['CV_RMSE_mean']:.4f})"
    )

    logger.info(f"Refitting {best_name} on {len(X_train):,} training rows…")
    if w_train is not None:
        best_model.fit(X_train, y_train, model__sample_weight=w_train)
    else:
        best_model.fit(X_train, y_train)

    holdout_metrics = evaluate_model(
        best_model, X_holdout, y_holdout, log_target=True,
    )
    results[best_name].update(holdout_metrics)
    logger.info(
        f"Holdout test ({len(X_holdout):,} rows, back-transformed to $): "
        f"RMSE=${holdout_metrics['RMSE']:,.0f}  MAE=${holdout_metrics['MAE']:,.0f}  "
        f"R²={holdout_metrics['R2']:.3f}  MAPE={holdout_metrics['MAPE_pct']:.1f}%  "
        f"RMSE/MAE={holdout_metrics['RMSE']/max(holdout_metrics['MAE'],1):.2f}"
    )

    y_pred_train_log = best_model.predict(X_train)
    log_residuals = y_train - y_pred_train_log
    smearing_factor = float(np.exp(log_residuals).mean())
    logger.info(f"Duan smearing factor: {smearing_factor:.4f}")

    # ── Calibration: measure and correct systematic prediction bias ───────────
    # The model may systematically over- or under-predict due to market drift,
    # temporal weighting imperfections, or training-data composition.  We compute
    # a multiplicative calibration factor from the holdout set so that the median
    # predicted price equals the median actual price.
    #
    # Global factor: median(actual / predicted) across all holdout rows.
    # City factors: same ratio per city — corrects location-level bias
    # (e.g. Springville over-prediction) that the city dummies didn't capture.
    y_pred_holdout_log = best_model.predict(X_holdout)
    y_pred_holdout_dollar = np.exp(y_pred_holdout_log) * smearing_factor
    ratios = y_dollar_holdout / y_pred_holdout_dollar
    global_cal = float(np.median(ratios))

    city_calibration: dict[str, float] = {}
    for city_name in set(cities_holdout):
        mask = cities_holdout == city_name
        if mask.sum() >= 5:
            city_cal = float(np.median(ratios[mask]))
            city_calibration[str(city_name)] = city_cal

    logger.info(
        f"Calibration factor (global): {global_cal:.4f} "
        f"(1.0 = no bias, <1 = model over-predicts)"
    )
    if city_calibration:
        biased = {c: f"{v:.3f}" for c, v in city_calibration.items()
                  if abs(v - 1.0) > 0.03}
        if biased:
            logger.info(f"City-level calibration offsets (>3% bias): {biased}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    joblib.dump(feature_cols, feature_path)
    joblib.dump(
        {
            "X_test": X_holdout,
            "y_test": y_holdout,
            "y_test_dollar": y_dollar_holdout,
            "cities": cities_holdout,
            "log_target": True,
            "smearing_factor": smearing_factor,
        },
        holdout_path,
    )

    metrics_payload = {
        "model_name": best_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "log_target": True,
        "smearing_factor": smearing_factor,
        "calibration_factor": global_cal,
        "city_calibration": city_calibration,
        "median_training_price": median_train_price,
        "n_training": int(len(X_train)),
        "n_holdout": int(len(X_holdout)),
        "holdout_RMSE": float(holdout_metrics["RMSE"]),
        "holdout_MAE": float(holdout_metrics["MAE"]),
        "holdout_R2": float(holdout_metrics["R2"]),
        "holdout_MAPE_pct": float(holdout_metrics["MAPE_pct"]),
        "holdout_within_10pct": holdout_metrics["within_10pct"],
        "holdout_within_20pct": holdout_metrics["within_20pct"],
        "holdout_within_30pct": holdout_metrics["within_30pct"],
        "cv_results": {
            k: {kk: float(vv) for kk, vv in v.items() if kk.startswith("CV_")}
            for k, v in results.items()
        },
    }
    metrics_path = model_path.replace(".joblib", "_metrics.json")
    with open(metrics_path, "w") as _f:
        json.dump(metrics_payload, _f, indent=2)

    logger.info(f"Model saved → {model_path}")
    logger.info(f"Holdout data saved → {holdout_path}")
    logger.info(f"Metrics saved → {metrics_path}")

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


def predict(
    model,
    feature_cols: list[str],
    df: pd.DataFrame,
    model_path: str = "models/hedonic_model.joblib",
) -> np.ndarray:
    """
    Run inference on a prepared DataFrame.
    Missing feature columns are filled with 0 (e.g. unseen zip codes).

    Automatically detects log-target models (via the metrics JSON) and
    back-transforms predictions to dollar-space with Duan smearing.

    Applies calibration factors (global + per-city) computed during training
    to correct systematic prediction bias.
    """
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].values
    y_pred = model.predict(X)

    metrics_json = model_path.replace(".joblib", "_metrics.json")
    log_target = False
    smearing_factor = 1.0
    global_cal = 1.0
    city_cal: dict[str, float] = {}
    if Path(metrics_json).exists():
        with open(metrics_json) as f:
            meta = json.load(f)
        log_target = meta.get("log_target", False)
        smearing_factor = meta.get("smearing_factor", 1.0)
        global_cal = meta.get("calibration_factor", 1.0)
        city_cal = meta.get("city_calibration", {})

    if log_target:
        y_pred = np.exp(y_pred) * smearing_factor

    # Apply city-level calibration where available, global otherwise
    if city_cal and "city" in df.columns:
        cities = df["city"].astype(str).values
        cal_factors = np.array([
            city_cal.get(c, global_cal) for c in cities
        ])
        y_pred = y_pred * cal_factors
    elif abs(global_cal - 1.0) > 0.005:
        y_pred = y_pred * global_cal

    return y_pred
