"""
Value scoring module.

Given a DataFrame of active listings with predicted prices,
compute a value score and rank homes.

Value Score = (predicted_price - actual_price) / predicted_price

  > 0  → home is cheaper than the model expects (potential deal)
  < 0  → home is more expensive than the model expects (overpriced)
  = 0  → fairly priced

Additional composite score factors in features-per-dollar:
  - Sqft per dollar
  - Beds + baths relative to price

When ADU data is available, an affordability bonus is blended in:
  - Homes with likely ADU income get a boost proportional to the
    fraction of the mortgage that ADU rent offsets.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _flag_low_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Flag predictions where the model is likely extrapolating.

    A prediction is low-confidence when key physical features fall far
    outside the training-data distribution seen among active listings.
    This catches edge cases (e.g., a 10-acre horse property with 1 bed)
    where the model's prediction shouldn't be trusted blindly.
    """
    flags = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index)

    for col, label in [("sqft", "sqft"), ("price", "price"), ("lot_sqft", "lot size")]:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if vals.empty:
            continue
        p5, p95 = vals.quantile(0.05), vals.quantile(0.95)
        iqr = p95 - p5
        lo, hi = p5 - 1.5 * iqr, p95 + 1.5 * iqr
        outlier = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        flags |= outlier
        reasons = reasons.where(~outlier, reasons + f"{label} out-of-range; ")

    df["low_confidence"] = flags
    df["confidence_reason"] = reasons.str.rstrip("; ")
    return df


def compute_value_scores(df: pd.DataFrame, predicted_prices: np.ndarray) -> pd.DataFrame:
    """
    Attach predicted prices and compute value scores to the listings DataFrame.

    Args:
        df: Prepared listings DataFrame (output of prepare_dataset).
        predicted_prices: Array of price predictions from model.predict().

    Returns:
        DataFrame with added columns:
          - predicted_price
          - value_score        (0–1 range after clipping)
          - pct_below_market   (how many % cheaper than predicted)
          - sqft_per_dollar
          - composite_score    (weighted blend of value + features)
          - low_confidence     (bool — True when model may be extrapolating)
          - confidence_reason  (human-readable explanation)
    """
    df = df.copy()
    df["predicted_price"] = predicted_prices
    df = _flag_low_confidence(df)

    # Core value score: fraction below predicted
    df["pct_below_market"] = (df["predicted_price"] - df["price"]) / df["predicted_price"] * 100
    df["value_score"] = df["pct_below_market"] / 100  # as a ratio

    # Features-per-dollar scores (normalized 0–1 across the dataset)
    if "sqft" in df.columns:
        df["sqft_per_dollar"] = df["sqft"] / df["price"]
        df["sqft_per_dollar_norm"] = _normalize(df["sqft_per_dollar"])
    else:
        df["sqft_per_dollar_norm"] = 0.0

    # Beds + baths per $100k
    if "beds" in df.columns and "baths" in df.columns:
        df["rooms_per_100k"] = (df["beds"] + df["baths"]) / (df["price"] / 100_000)
        df["rooms_per_100k_norm"] = _normalize(df["rooms_per_100k"])
    else:
        df["rooms_per_100k_norm"] = 0.0

    # ADU affordability bonus: fraction of mortgage offset by ADU rent
    has_adu_data = "estimated_adu_rent" in df.columns and "estimated_mortgage" in df.columns
    if has_adu_data:
        safe_mortgage = df["estimated_mortgage"].replace(0, np.nan)
        df["adu_offset_pct"] = (df["estimated_adu_rent"] / safe_mortgage).fillna(0).clip(0, 1)
        df["adu_offset_norm"] = _normalize(df["adu_offset_pct"])
    else:
        df["adu_offset_pct"] = 0.0
        df["adu_offset_norm"] = 0.0

    # Composite score: value + sqft/dollar + rooms/dollar + ADU bonus
    # With ADU data: 50% value, 20% sqft, 12% rooms, 18% ADU affordability
    # Without:       60% value, 25% sqft, 15% rooms (original weights)
    value_norm = _normalize(df["value_score"].clip(-0.5, 0.5))
    if has_adu_data and (df["adu_offset_pct"] > 0).any():
        df["composite_score"] = (
            0.50 * value_norm +
            0.20 * df["sqft_per_dollar_norm"] +
            0.12 * df["rooms_per_100k_norm"] +
            0.18 * df["adu_offset_norm"]
        )
    else:
        df["composite_score"] = (
            0.60 * value_norm +
            0.25 * df["sqft_per_dollar_norm"] +
            0.15 * df["rooms_per_100k_norm"]
        )

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def top_deals(df: pd.DataFrame, min_value_score: float = 0.05, top_n: int = 5) -> pd.DataFrame:
    """
    Filter to homes that are at least min_value_score below predicted price,
    then return the top_n by composite score.
    """
    deals = df[df["value_score"] >= min_value_score].copy()
    if deals.empty:
        logger.warning(
            f"No homes found with value_score >= {min_value_score:.0%}. "
            "Try lowering min_value_score in config.yaml."
        )
        # Fall back to top N by composite score regardless of cutoff
        return df.head(top_n)

    logger.info(f"{len(deals)} listings meet the min_value_score threshold.")
    return deals.head(top_n)


def score_summary(df: pd.DataFrame) -> str:
    """Return a human-readable summary of the scoring distribution."""
    lines = [
        f"Total listings scored: {len(df)}",
        f"Median predicted price:  ${df['predicted_price'].median():,.0f}",
        f"Median actual price:     ${df['price'].median():,.0f}",
        f"Median pct below market: {df['pct_below_market'].median():.1f}%",
        f"Listings below market:   {(df['value_score'] > 0).sum()} "
        f"({(df['value_score'] > 0).mean():.0%})",
        f"Listings >5% below:      {(df['value_score'] >= 0.05).sum()}",
        f"Listings >10% below:     {(df['value_score'] >= 0.10).sum()}",
    ]
    return "\n".join(lines)
