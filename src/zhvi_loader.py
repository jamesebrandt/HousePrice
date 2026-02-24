"""
Zillow Home Value Index (ZHVI) loader.

Reads the city-level ZHVI CSV (wide format, one row per city, date columns)
and computes per-city appreciation features that are joined onto listings as
signals of which neighborhoods are trending up vs. down.

Expected file: City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
  - Columns: RegionID, SizeRank, RegionName, RegionType, StateName, State,
             Metro, CountyName, <YYYY-MM-DD>...
  - Each date column is the smoothed, seasonally-adjusted median ZHVI for that month.
"""

import glob
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Date columns are YYYY-MM-DD strings; we identify them by this pattern.
_DATE_PREFIX = "20"  # all dates in the dataset start with "20xx"


def _find_zhvi_file(zhvi_dir: str) -> str | None:
    """Return the path to the ZHVI city-level CSV, or None if not found."""
    patterns = [
        os.path.join(zhvi_dir, "City_zhvi_*.csv"),
        os.path.join(zhvi_dir, "*zhvi*city*.csv"),
        os.path.join(zhvi_dir, "*zhvi*.csv"),
    ]
    for pat in patterns:
        files = glob.glob(pat)
        if files:
            return sorted(files)[-1]  # most recently modified
    return None


def load_zhvi_data(zhvi_dir: str, state: str = "UT") -> pd.DataFrame:
    """
    Load the ZHVI city-level CSV, filter to the given state, and return a
    tidy DataFrame with columns:
        city (str), date (datetime), zhvi (float)

    Only rows with at least one non-null ZHVI value are kept.
    """
    path = _find_zhvi_file(zhvi_dir)
    if path is None:
        logger.warning(f"No ZHVI city CSV found in {zhvi_dir}")
        return pd.DataFrame()

    logger.info(f"Loading ZHVI from {os.path.basename(path)} ...")
    df = pd.read_csv(path, low_memory=False)

    # Identify date columns (all start with "20")
    meta_cols = [c for c in df.columns if not c.startswith(_DATE_PREFIX)]
    date_cols = [c for c in df.columns if c.startswith(_DATE_PREFIX)]

    if not date_cols:
        logger.warning("No date columns found in ZHVI file.")
        return pd.DataFrame()

    # Filter to state
    state_col = "State" if "State" in df.columns else "StateName"
    state_val = state if "State" in df.columns else _state_abbr_to_name(state)
    df = df[df[state_col] == state_val].copy()

    if df.empty:
        logger.warning(f"No rows for state '{state}' in ZHVI file.")
        return pd.DataFrame()

    # Melt to long format: city, date, zhvi
    id_vars = [c for c in ["RegionName", "State", "Metro", "CountyName"] if c in df.columns]
    melted = df[id_vars + date_cols].melt(
        id_vars=id_vars, var_name="date_str", value_name="zhvi"
    )
    melted["date"] = pd.to_datetime(melted["date_str"], errors="coerce")
    melted = melted.dropna(subset=["date", "zhvi"])
    melted = melted.rename(columns={"RegionName": "city"})
    melted["city"] = melted["city"].str.strip()
    melted = melted.sort_values(["city", "date"]).reset_index(drop=True)

    logger.info(
        f"ZHVI: loaded {melted['city'].nunique()} Utah cities, "
        f"{melted['date'].min().strftime('%Y-%m')} – {melted['date'].max().strftime('%Y-%m')}"
    )
    return melted[["city", "date", "zhvi"]]


def compute_zhvi_features(zhvi_df: pd.DataFrame) -> dict[str, dict]:
    """
    Compute per-city appreciation features from the tidy ZHVI DataFrame.

    Returns a dict:  city (str, title-case) → {
        'zhvi_current':      float  — most recent ZHVI value ($)
        'zhvi_yoy_pct':      float  — YoY % appreciation
        'zhvi_3yr_cagr':     float  — 3-year compound annual growth rate
        'zhvi_5yr_cagr':     float  — 5-year compound annual growth rate
        'zhvi_momentum':     float  — recent 3-month avg vs prior 3-month avg
    }
    """
    if zhvi_df.empty:
        return {}

    features: dict[str, dict] = {}
    max_date = zhvi_df["date"].max()

    def months_ago(n: int) -> pd.Timestamp:
        # Approximate: subtract n*30 days, then snap to nearest data point
        return max_date - pd.DateOffset(months=n)

    for city, grp in zhvi_df.groupby("city"):
        grp = grp.sort_values("date").set_index("date")["zhvi"]

        def nearest(target: pd.Timestamp) -> float | None:
            """Return the ZHVI value closest to target date, or None."""
            if grp.empty:
                return None
            idx = grp.index.get_indexer([target], method="nearest")[0]
            val = grp.iloc[idx]
            return float(val) if not np.isnan(val) else None

        current  = nearest(max_date)
        ago_12m  = nearest(months_ago(12))
        ago_36m  = nearest(months_ago(36))
        ago_60m  = nearest(months_ago(60))
        ago_3m   = nearest(months_ago(3))
        ago_6m   = nearest(months_ago(6))

        yoy        = (current / ago_12m - 1) if current and ago_12m else 0.0
        cagr_3yr   = (current / ago_36m) ** (1 / 3) - 1 if current and ago_36m else 0.0
        cagr_5yr   = (current / ago_60m) ** (1 / 5) - 1 if current and ago_60m else 0.0
        # Momentum: how much faster/slower is recent 3m vs prior 3m
        momentum   = (current / ago_3m - 1) - (ago_3m / ago_6m - 1) if current and ago_3m and ago_6m else 0.0

        features[city] = {
            "zhvi_current":   round(float(current or 0), 0),
            "zhvi_yoy_pct":   round(float(yoy), 4),
            "zhvi_3yr_cagr":  round(float(cagr_3yr), 4),
            "zhvi_5yr_cagr":  round(float(cagr_5yr), 4),
            "zhvi_momentum":  round(float(momentum), 4),
        }

    logger.info(f"ZHVI: computed appreciation features for {len(features)} cities.")
    return features


def zhvi_feature_summary(zhvi_features: dict[str, dict], cities: list[str] | None = None) -> str:
    """Return a human-readable summary, optionally filtered to a city list."""
    if not zhvi_features:
        return "No ZHVI features available."

    show = cities if cities else sorted(zhvi_features.keys())
    lines = ["City ZHVI appreciation features:"]
    for city in show:
        if city not in zhvi_features:
            continue
        f = zhvi_features[city]
        lines.append(
            f"  {city:<20s}  current=${f['zhvi_current']:,.0f}"
            f"  YoY={f['zhvi_yoy_pct']:+.1%}"
            f"  3yr={f['zhvi_3yr_cagr']:+.1%}/yr"
            f"  5yr={f['zhvi_5yr_cagr']:+.1%}/yr"
            f"  momentum={f['zhvi_momentum']:+.2%}"
        )
    return "\n".join(lines)


def _state_abbr_to_name(abbr: str) -> str:
    mapping = {
        "UT": "Utah", "CA": "California", "TX": "Texas", "FL": "Florida",
        "NY": "New York", "AZ": "Arizona", "CO": "Colorado", "NV": "Nevada",
        "ID": "Idaho", "WA": "Washington", "OR": "Oregon",
    }
    return mapping.get(abbr.upper(), abbr)
