"""
Feature engineering for the hedonic pricing model.

Redfin column names are messy (all caps, spaces, special chars).
This module standardizes them and derives useful model features.
"""

import re
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

# Redfin sometimes prepends a 6-char hash to city names (e.g. "8hno7i Lake Pt").
_GARBLED_CITY_RE = re.compile(r"^[A-Za-z0-9]{6}\s+(.+)$")

# Common abbreviations in Redfin data → canonical names.
_CITY_ALIAS = {
    "Lake Pt": "Lake Point",
}

# Map raw Redfin column names → clean names.
# Covers both the current format (starts with SALE TYPE) and the older MLS#-first format.
COLUMN_MAP = {
    # New format columns
    "SALE TYPE":        "sale_type",
    "SOLD DATE":        "sold_date",
    "PROPERTY TYPE":    "property_type",
    "ADDRESS":          "address",         # new format: street address
    "CITY":             "city_raw",        # new format: city in CSV (merged into "city" later)
    "STATE OR PROVINCE":"state",
    # Shared columns (both formats)
    "MLS#":             "mls_id",
    "PRICE":            "price",
    "BEDS":             "beds",
    "BATHS":            "baths",
    "LOCATION":         "neighborhood",   # in new format this is a neighborhood name
    "ZIP OR POSTAL CODE":"zip_code",
    "SQUARE FEET":      "sqft",
    "LOT SIZE":         "lot_sqft",
    "YEAR BUILT":       "year_built",
    "DAYS ON MARKET":   "days_on_market",
    "$/SQUARE FEET":    "price_per_sqft",
    "HOA/MONTH":        "hoa_monthly",
    "STATUS":           "status",
    "URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)": "url",
    "URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)": "url",
    "LATITUDE":         "latitude",
    "LONGITUDE":        "longitude",
    "city":             "city",
    "sold":             "sold",
    "description":      "description",
}

NUMERIC_COLS = ["price", "beds", "baths", "sqft", "lot_sqft", "year_built",
                "days_on_market", "price_per_sqft", "hoa_monthly", "latitude", "longitude"]

CATEGORICAL_COLS = ["city", "zip_code"]


def _parse_price(val) -> float:
    """Convert '$450,000' or '450000' to float."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_sqft(val) -> float:
    """Convert '2,400 Sq. Ft.' or '2400' to float."""
    if pd.isna(val):
        return np.nan
    s = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(s)
    except ValueError:
        return np.nan


def _sanitize_city(name: str) -> str:
    """Strip garbled Redfin hash prefixes and normalise abbreviations."""
    if pd.isna(name):
        return name
    s = str(name).strip()
    m = _GARBLED_CITY_RE.match(s)
    if m:
        s = m.group(1).strip()
    return _CITY_ALIAS.get(s, s)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Redfin columns to clean names. Tolerates missing columns."""
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Merge city_raw → city when city isn't already set by the scraper
    if "city_raw" in df.columns:
        if "city" not in df.columns:
            df["city"] = df["city_raw"]
        else:
            df["city"] = df["city"].fillna(df["city_raw"])
        df = df.drop(columns=["city_raw"])

    # Sanitize city names: strip hash prefixes, normalise abbreviations
    if "city" in df.columns:
        before = df["city"].nunique()
        df["city"] = df["city"].apply(_sanitize_city)
        after = df["city"].nunique()
        if before != after:
            logger.info(
                f"City name sanitization: {before} → {after} unique cities "
                f"(merged {before - after} garbled/abbreviated variants)"
            )

    # Also handle any columns already renamed
    for col in df.columns:
        clean = col.strip().lower().replace(" ", "_")
        if clean not in df.columns and clean != col:
            df = df.rename(columns={col: clean})

    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Parse price, sqft, and other numerics from string formats."""
    if "price" in df.columns:
        df["price"] = df["price"].apply(_parse_price)
    if "sqft" in df.columns:
        df["sqft"] = df["sqft"].apply(_parse_sqft)
    if "lot_sqft" in df.columns:
        df["lot_sqft"] = df["lot_sqft"].apply(_parse_sqft)

    for col in ["beds", "baths", "year_built", "days_on_market", "hoa_monthly"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def engineer_features(
    df: pd.DataFrame,
    permit_features: dict | None = None,
    zhvi_features: dict | None = None,
    income_map: dict | None = None,
) -> pd.DataFrame:
    """
    Derive model features from cleaned data.
    All new columns have a `feat_` prefix for clarity.

    Args:
        df: cleaned listings DataFrame
        permit_features: optional dict from bps_loader.compute_permit_features()
            county_fips (str) → {permits_last12m, permits_yoy_growth, permits_3yr_cagr}
        zhvi_features: optional dict from zhvi_loader.compute_zhvi_features()
            city (str) → {zhvi_current, zhvi_yoy_pct, zhvi_3yr_cagr, zhvi_5yr_cagr, zhvi_momentum}
    """
    from src.bps_loader import get_county_fips

    current_year = datetime.now().year

    if "year_built" in df.columns:
        df["feat_home_age"] = current_year - df["year_built"]

    # ── Days since sale (recency — used for temporal weighting in training) ────
    if "sold_date" in df.columns:
        today = pd.Timestamp.today().normalize()
        sold_dt = pd.to_datetime(df["sold_date"], errors="coerce")
        df["feat_days_since_sale"] = (today - sold_dt).dt.days.clip(lower=0)
        # Active/unsold listings have no sold_date — fill with 0 (not used in training)
        df["feat_days_since_sale"] = df["feat_days_since_sale"].fillna(0).astype(int)

    if "sqft" in df.columns and "beds" in df.columns:
        df["feat_sqft_per_bed"] = df["sqft"] / df["beds"].replace(0, np.nan)

    if "sqft" in df.columns and "lot_sqft" in df.columns:
        df["feat_lot_to_sqft_ratio"] = df["lot_sqft"] / df["sqft"].replace(0, np.nan)

    if "hoa_monthly" in df.columns:
        df["feat_has_hoa"] = (~df["hoa_monthly"].isna() & (df["hoa_monthly"] > 0)).astype(int)
        df["hoa_monthly"] = df["hoa_monthly"].fillna(0)

    if "days_on_market" in df.columns:
        df["days_on_market"] = df["days_on_market"].fillna(0)
        df["feat_long_on_market"] = (df["days_on_market"] > 60).astype(int)

    # ── City-level Zillow ZHVI appreciation features ───────────────────────
    # Direct appreciation signal: YoY, 3yr CAGR, 5yr CAGR, and short-term momentum.
    if zhvi_features and "city" in df.columns:
        zhvi_keys = ["zhvi_current", "zhvi_yoy_pct", "zhvi_3yr_cagr", "zhvi_5yr_cagr", "zhvi_momentum"]
        for key in zhvi_keys:
            df[f"feat_{key}"] = df["city"].apply(
                lambda c: zhvi_features.get(str(c), {}).get(key, 0.0)
            )
        mapped = df["city"].apply(lambda c: str(c) in zhvi_features).sum()
        logger.info(f"ZHVI features joined for {mapped:,} / {len(df):,} listings.")

    # ── County-level building permit features ──────────────────────────────
    # These signal neighborhood demand growth and are strong appreciation proxies.
    if permit_features and "city" in df.columns:
        def _permits(city, key, default):
            fips = get_county_fips(str(city))
            if fips and fips in permit_features:
                return permit_features[fips].get(key, default)
            return default

        df["feat_permits_last12m"]    = df["city"].apply(lambda c: _permits(c, "permits_last12m",    0))
        df["feat_permits_yoy_growth"] = df["city"].apply(lambda c: _permits(c, "permits_yoy_growth", 0.0))
        df["feat_permits_3yr_cagr"]   = df["city"].apply(lambda c: _permits(c, "permits_3yr_cagr",   0.0))
        df["feat_permits_10yr_cagr"]  = df["city"].apply(lambda c: _permits(c, "permits_10yr_cagr",  0.0))
        df["feat_permits_long_trend"] = df["city"].apply(lambda c: _permits(c, "permits_long_trend", 0.0))

        mapped = df["city"].apply(lambda c: get_county_fips(str(c)) is not None).sum()
        logger.info(f"Permit features joined for {mapped:,} / {len(df):,} listings.")

    # ── ZIP-level Census median household income ───────────────────────────
    # Strongest sub-city price signal: wealthier ZIP → higher prices.
    if income_map and "zip_code" in df.columns:
        zip5 = df["zip_code"].astype(str).str[:5]
        df["feat_median_income"] = zip5.map(income_map).fillna(0.0)
        matched = (df["feat_median_income"] > 0).sum()
        logger.info(f"ACS income joined for {matched:,} / {len(df):,} listings.")

    # ── One-hot encode city and zip ─────────────────────────────────────────
    if "city" in df.columns:
        city_dummies = pd.get_dummies(df["city"], prefix="city", drop_first=False)
        df = pd.concat([df, city_dummies], axis=1)

    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype(str).str[:5]
        zip_dummies = pd.get_dummies(df["zip_code"], prefix="zip", drop_first=False)
        df = pd.concat([df, zip_dummies], axis=1)

    return df


def get_model_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of columns to use as model inputs (X)."""
    base = ["beds", "baths", "sqft", "lot_sqft", "feat_home_age",
            "feat_sqft_per_bed", "feat_lot_to_sqft_ratio",
            "feat_has_hoa", "hoa_monthly", "days_on_market", "feat_long_on_market",
            "latitude", "longitude",
            # Recency signal: lets the model capture market drift over the 2-year comp window
            "feat_days_since_sale",
            # ZIP-level Census median household income
            "feat_median_income",
            # County-level building permit features (demand signal — recent + long-term)
            "feat_permits_last12m", "feat_permits_yoy_growth", "feat_permits_3yr_cagr",
            "feat_permits_10yr_cagr", "feat_permits_long_trend",
            # City-level Zillow ZHVI appreciation features
            "feat_zhvi_current", "feat_zhvi_yoy_pct", "feat_zhvi_3yr_cagr",
            "feat_zhvi_5yr_cagr", "feat_zhvi_momentum",
            # ADU potential (structural heuristic — keyword score isn't available for sold comps)
            "adu_structural_score"]
    city_cols = [c for c in df.columns if c.startswith("city_")]
    # Match zip dummy columns like zip_84003 — exclude the raw "zip_code" string column
    zip_cols  = [c for c in df.columns if c.startswith("zip_") and c != "zip_code"]
    return [c for c in base + city_cols + zip_cols if c in df.columns]


def prepare_dataset(
    raw_df: pd.DataFrame,
    drop_outliers: bool = True,
    permit_features: dict | None = None,
    zhvi_features: dict | None = None,
    income_map: dict | None = None,
    exclude_cities: list[str] | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: standardize → clean → engineer → drop rows with missing target.
    Returns a DataFrame ready for model training or scoring.

    Args:
        raw_df: raw listings DataFrame (from scraper or CSV)
        drop_outliers: if True, remove extreme price outliers
        permit_features: optional county permit features from bps_loader
        zhvi_features: optional city-level ZHVI features from zhvi_loader
        income_map: optional ZIP→median_income dict from acs_loader
        exclude_cities: optional list of city names to drop (e.g. outliers)
    """
    df = raw_df.copy()
    df = standardize_columns(df)
    df = clean_numeric(df)

    # ── Exclude specific cities (outliers or places you don't care about) ─────
    if exclude_cities and "city" in df.columns:
        exclude_set = {str(c).strip() for c in exclude_cities}
        before_excl = len(df)
        df = df[~df["city"].astype(str).str.strip().isin(exclude_set)].copy()
        removed = before_excl - len(df)
        if removed:
            logger.info(f"Excluded {removed:,} listings in {exclude_set}. Remaining: {len(df):,}")

    # ── Deduplicate ────────────────────────────────────────────────────────────
    # Multiple downloaded files can overlap geographically, producing the same
    # listing in more than one CSV. Deduplicate before feature engineering so
    # duplicates don't inflate training counts or appear twice in results.
    before_dedup = len(df)
    if "mls_id" in df.columns:
        # Keep the row with the most complete data (fewest NaNs) for each MLS#
        df = (df
              .assign(_null_count=df.isnull().sum(axis=1))
              .sort_values("_null_count")
              .drop_duplicates(subset=["mls_id"], keep="first")
              .drop(columns=["_null_count"])
              .reset_index(drop=True))
    else:
        # Fallback: deduplicate on address + price + beds
        addr_col = next((c for c in ["address", "location", "neighborhood"] if c in df.columns), None)
        if addr_col:
            df = df.drop_duplicates(subset=[addr_col, "price", "beds"], keep="first").reset_index(drop=True)
    removed_dedup = before_dedup - len(df)
    if removed_dedup:
        logger.info(f"Removed {removed_dedup:,} duplicate listings. Remaining: {len(df):,}")

    df = engineer_features(df, permit_features=permit_features, zhvi_features=zhvi_features, income_map=income_map)

    # Normalize the sold column — CSV round-trips turn booleans into strings
    if "sold" in df.columns:
        df["sold"] = df["sold"].map(
            {True: True, False: False, "True": True, "False": False,
             "true": True, "false": False, 1: True, 0: False}
        ).fillna(False)

    # Drop rows without a price (the target variable)
    before = len(df)
    df = df.dropna(subset=["price"])
    logger.info(f"Dropped {before - len(df)} rows with missing price. Remaining: {len(df)}")

    # Drop rows with obviously wrong prices
    df = df[df["price"] > 50_000]
    df = df[df["price"] < 5_000_000]

    if drop_outliers:
        # IQR-based outlier removal on price
        q1, q3 = df["price"].quantile(0.01), df["price"].quantile(0.99)
        df = df[(df["price"] >= q1) & (df["price"] <= q3)]
        logger.info(f"After outlier removal: {len(df)} rows")

    return df.reset_index(drop=True)
