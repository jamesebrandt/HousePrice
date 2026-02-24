"""
Criteria filter module.

Reads your preferences from config.yaml and filters a scored
listings DataFrame down to only the homes that meet your criteria.
"""

import logging
from datetime import datetime
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_criteria(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Filter listings to those matching all criteria in config['criteria'].

    Logs how many listings each filter removes so you can tune the criteria.
    """
    criteria = config.get("criteria", {})
    original = len(df)

    def log_filter(label: str, mask: pd.Series) -> pd.DataFrame:
        nonlocal df
        before = len(df)
        df = df[mask]
        removed = before - len(df)
        logger.info(f"  [{label}] removed {removed}, remaining {len(df)}")
        return df

    # Price
    if criteria.get("max_price"):
        log_filter(f"max_price ≤ ${criteria['max_price']:,}", df["price"] <= criteria["max_price"])

    # Beds
    if criteria.get("min_beds"):
        if "beds" in df.columns:
            log_filter(f"min_beds ≥ {criteria['min_beds']}", df["beds"] >= criteria["min_beds"])

    # Baths
    if criteria.get("min_baths"):
        if "baths" in df.columns:
            log_filter(f"min_baths ≥ {criteria['min_baths']}", df["baths"] >= criteria["min_baths"])

    # Square footage
    if criteria.get("min_sqft"):
        if "sqft" in df.columns:
            log_filter(f"min_sqft ≥ {criteria['min_sqft']:,}", df["sqft"] >= criteria["min_sqft"])

    # Lot size
    if criteria.get("min_lot_sqft"):
        if "lot_sqft" in df.columns:
            mask = df["lot_sqft"].isna() | (df["lot_sqft"] >= criteria["min_lot_sqft"])
            log_filter(f"min_lot_sqft ≥ {criteria['min_lot_sqft']:,}", mask)

    # Year built / home age
    current_year = datetime.now().year
    if criteria.get("max_year_built"):
        if "year_built" in df.columns:
            mask = df["year_built"].isna() | (df["year_built"] >= criteria["max_year_built"])
            log_filter(f"year_built ≥ {criteria['max_year_built']}", mask)
    elif criteria.get("max_home_age_years"):
        min_year = current_year - criteria["max_home_age_years"]
        if "year_built" in df.columns:
            mask = df["year_built"].isna() | (df["year_built"] >= min_year)
            log_filter(f"year_built ≥ {min_year} (age ≤ {criteria['max_home_age_years']} yrs)", mask)

    # Days on market
    if criteria.get("max_days_on_market"):
        if "days_on_market" in df.columns:
            mask = df["days_on_market"].isna() | (df["days_on_market"] <= criteria["max_days_on_market"])
            log_filter(f"days_on_market ≤ {criteria['max_days_on_market']}", mask)

    # City filter — only show target cities (not training-only cities)
    if "city" in df.columns:
        target_cities = config.get("search", {}).get("cities", [])
        if target_cities:
            log_filter(
                f"city in {target_cities}",
                df["city"].isin(target_cities)
            )

    # Only active listings (not sold comps used for training)
    if "sold" in df.columns:
        log_filter("active listings only", df["sold"] == False)  # noqa: E712

    logger.info(f"Criteria filter: {original} → {len(df)} listings")
    return df.reset_index(drop=True)


def filter_summary(df_before: pd.DataFrame, df_after: pd.DataFrame, config: dict) -> str:
    """Return a human-readable string describing what the filter kept."""
    criteria = config.get("criteria", {})
    lines = [
        "--- Criteria Applied ---",
        f"  Max price:       ${criteria.get('max_price', 'any'):,}" if criteria.get('max_price') else "  Max price:       any",
        f"  Min beds:        {criteria.get('min_beds', 'any')}",
        f"  Min baths:       {criteria.get('min_baths', 'any')}",
        f"  Min sqft:        {criteria.get('min_sqft', 'any'):,}" if criteria.get('min_sqft') else "  Min sqft:        any",
        f"  Max home age:    {criteria.get('max_home_age_years', 'any')} years",
        f"  Min lot:         {criteria.get('min_lot_sqft', 'any'):,} sqft" if criteria.get('min_lot_sqft') else "  Min lot:         any",
        f"Passed filter: {len(df_after)} / {len(df_before)} listings",
    ]
    return "\n".join(lines)
