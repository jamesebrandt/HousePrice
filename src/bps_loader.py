"""
Census Bureau Building Permits Survey (BPS) loader.

Reads the county-level monthly TXT files (co{YY}{MM}c.txt) and computes
per-Utah-county permit growth features that are joined onto home listings
as leading indicators of neighborhood demand and appreciation.

File naming convention:
  co{YY}{MM}c.txt  — current month estimates  (we use these)
  co{YY}{MM}y.txt  — year-to-date cumulative   (skipped to avoid double-counting)
"""

import glob
import logging
import os
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Utah state FIPS = 49.  County FIPS (3-digit) → readable name.
UTAH_COUNTY_NAMES = {
    "003": "Box Elder",
    "005": "Cache",
    "007": "Carbon",
    "009": "Daggett",
    "011": "Davis",
    "013": "Duchesne",
    "015": "Emery",
    "017": "Garfield",
    "019": "Grand",
    "021": "Iron",
    "023": "Juab",
    "025": "Kane",
    "027": "Millard",
    "029": "Morgan",
    "031": "Piute",
    "033": "Rich",
    "035": "Salt Lake",
    "037": "San Juan",
    "039": "Sanpete",
    "041": "Sevier",
    "043": "Summit",
    "045": "Tooele",
    "047": "Uintah",
    "049": "Utah",
    "051": "Wasatch",
    "053": "Washington",
    "055": "Wayne",
    "057": "Weber",
}

# City name (lowercase) → 3-digit county FIPS within Utah
CITY_TO_COUNTY_FIPS: dict[str, str] = {
    # Utah County (049)
    "saratoga springs": "049",
    "eagle mountain":   "049",
    "lehi":             "049",
    "american fork":    "049",
    "highland":         "049",
    "cedar hills":      "049",
    "pleasant grove":   "049",
    "lindon":           "049",
    "vineyard":         "049",
    "spanish fork":     "049",
    "mapleton":         "049",
    "springville":      "049",
    "payson":           "049",
    "elk ridge":        "049",
    "santaquin":        "049",
    "orem":             "049",
    "provo":            "049",
    # Salt Lake County (035)
    "salt lake city":   "035",
    "south salt lake":  "035",
    "west valley city": "035",
    "west jordan":      "035",
    "south jordan":     "035",
    "sandy":            "035",
    "draper":           "035",
    "murray":           "035",
    "midvale":          "035",
    "taylorsville":     "035",
    "bluffdale":        "035",
    "herriman":         "035",
    "riverton":         "035",
    "cottonwood heights": "035",
    # Davis County (011)
    "syracuse":         "011",
    "clearfield":       "011",
    "layton":           "011",
    "bountiful":        "011",
    "north salt lake":  "011",
    "centerville":      "011",
    "farmington":       "011",
    "kaysville":        "011",
    "clinton":          "011",
    "sunset":           "011",
    # Weber County (057)
    "ogden":            "057",
    "west haven":       "057",
    "roy":              "057",
    "riverdale":        "057",
    "washington terrace": "057",
    "south ogden":      "057",
    "north ogden":      "057",
    "pleasant view":    "057",
    # Tooele County (045)
    "tooele":           "045",
    "lake point":       "045",
    "stansbury park":   "045",
    "grantsville":      "045",
    "burmester":        "045",
    "erda":             "045",
    # Wasatch County (051)
    "heber city":       "051",
    "daniel":           "051",
    "midway":           "051",
    "charleston":       "051",
    # Juab County (023)
    "nephi":            "023",
    # Sanpete County (039)
    "fountain green":   "039",
    "manti":            "039",
    "ephraim":          "039",
    # Summit County (043)
    "park city":        "043",
    # Washington County (053)
    "st. george":       "053",
    "st george":        "053",
    "hurricane":        "053",
}


def _load_annual_county_file(bps_dir: str) -> dict[str, dict]:
    """
    Load permits_by_county_annual_1980_2022.csv and return a dict:
        county_fips (3-digit str) → {
            permits_10yr_cagr: float  — 2012→2022 single-family permit CAGR
            permits_long_trend: float — 2000→2022 single-family permit CAGR (pre-covid baseline)
        }
    Only Utah counties are included.
    """
    annual_path = os.path.join(bps_dir, "permits_by_county_annual_1980_2022.csv")
    if not os.path.exists(annual_path):
        return {}

    try:
        df = pd.read_csv(annual_path, low_memory=False)
        df = df[df["STUSAB"] == "UT"].copy()
        df["county_fips"] = df["COUNTY"].astype(str).str.zfill(3)

        result: dict[str, dict] = {}
        for _, row in df.iterrows():
            fips = row["county_fips"]

            def _permits(year: int) -> float:
                col = f"SINGLE_FAMILY_PERMITS_{year}"
                val = pd.to_numeric(row.get(col, 0), errors="coerce")
                return max(float(val) if not pd.isna(val) else 0.0, 1.0)

            # 10-year CAGR: 2012 → 2022 (covers full post-2008 recovery cycle)
            p2012 = _permits(2012)
            p2022 = _permits(2022)
            cagr10 = (p2022 / p2012) ** (1 / 10) - 1

            # Long-term CAGR: 2000 → 2022 (pre-housing-bubble baseline)
            p2000 = _permits(2000)
            cagr22 = (p2022 / p2000) ** (1 / 22) - 1

            result[fips] = {
                "permits_10yr_cagr":   round(cagr10, 4),
                "permits_long_trend":  round(cagr22, 4),
            }

        logger.info(f"BPS annual: loaded long-term permit CAGRs for {len(result)} Utah counties.")
        return result

    except Exception as e:
        logger.warning(f"Could not load annual county permit file: {e}")
        return {}


def load_bps_data(bps_dir: str) -> pd.DataFrame:
    """
    Read all monthly county BPS files (*c.txt) from bps_dir.

    Returns a DataFrame with columns:
        date (int YYYYMM), county_fips (str 3-digit), county_name (str),
        permits_1unit (int)
    Filtered to Utah only.
    """
    pattern = os.path.join(bps_dir, "co*c.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning(f"No BPS monthly files (co*c.txt) found in {bps_dir}")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            # Rows 0-1 are a two-line header; row 2 is blank — skip all three.
            df = pd.read_csv(
                f,
                header=None,
                skiprows=3,
                on_bad_lines="skip",
                low_memory=False,
            )
            if df.shape[1] < 9:
                continue

            # Keep only the columns we need: date, state, county, name, 1-unit bldgs
            sub = df.iloc[:, [0, 1, 2, 5, 6]].copy()
            sub.columns = ["date", "state_fips", "county_fips", "county_name", "permits_1unit"]

            # Filter to Utah (state FIPS 49)
            sub["state_fips"] = sub["state_fips"].astype(str).str.strip().str.zfill(2)
            sub = sub[sub["state_fips"] == "49"].copy()
            if sub.empty:
                continue

            sub["county_fips"] = sub["county_fips"].astype(str).str.strip().str.zfill(3)
            sub["date"] = pd.to_numeric(sub["date"], errors="coerce")
            sub["permits_1unit"] = pd.to_numeric(sub["permits_1unit"], errors="coerce").fillna(0).astype(int)
            sub["county_name"] = sub["county_name"].astype(str).str.strip()
            frames.append(sub[["date", "county_fips", "county_name", "permits_1unit"]])

        except Exception as e:
            logger.warning(f"Error reading BPS file {os.path.basename(f)}: {e}")

    if not frames:
        logger.warning("No Utah data found in any BPS files.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["date"]).copy()
    combined["date"] = combined["date"].astype(int)
    combined = combined.sort_values("date").reset_index(drop=True)
    logger.info(
        f"BPS: loaded {len(combined):,} Utah county-month rows "
        f"({combined['date'].min()} – {combined['date'].max()})"
    )
    return combined


def compute_permit_features(
    bps_df: pd.DataFrame,
    bps_dir: str | None = None,
) -> dict[str, dict]:
    """
    Compute per-county permit growth features from the BPS time series.

    Returns a dict:  county_fips (str) → {
        'permits_last12m':    int   — total 1-unit permits in most recent 12 months
        'permits_yoy_growth': float — YoY % change vs prior 12 months
        'permits_3yr_cagr':   float — 3-year CAGR vs 3 years prior
        'permits_10yr_cagr':  float — 10-year CAGR 2012→2022 (post-crash recovery)
        'permits_long_trend': float — 22-year CAGR 2000→2022 (secular trend)
    }

    Pass bps_dir to also load the long-term annual file. If None, the
    10yr/long_trend fields default to 0.
    """
    if bps_df.empty:
        return {}

    max_date = int(bps_df["date"].max())
    max_year  = max_date // 100
    max_month = max_date % 100

    def months_back(n: int) -> int:
        y, m = max_year, max_month
        for _ in range(n):
            m -= 1
            if m == 0:
                m, y = 12, y - 1
        return y * 100 + m

    cut12 = months_back(12)
    cut24 = months_back(24)
    cut36 = months_back(36)
    cut48 = months_back(48)

    features: dict[str, dict] = {}
    for county_fips, grp in bps_df.groupby("county_fips"):
        grp = grp.sort_values("date")
        last12     = grp[grp["date"] >  cut12]["permits_1unit"].sum()
        prior12    = grp[(grp["date"] > cut24) & (grp["date"] <= cut12)]["permits_1unit"].sum()
        last36     = grp[grp["date"] >  cut36]["permits_1unit"].sum()
        prior12_36 = grp[(grp["date"] > cut48) & (grp["date"] <= cut36)]["permits_1unit"].sum()

        yoy   = float(last12 - prior12) / max(int(prior12), 1)
        cagr3 = float((last36 / 36) / max(prior12_36 / 12, 1)) ** (1 / 3) - 1

        features[county_fips] = {
            "permits_last12m":    int(last12),
            "permits_yoy_growth": round(yoy, 4),
            "permits_3yr_cagr":   round(cagr3, 4),
            "permits_10yr_cagr":  0.0,
            "permits_long_trend": 0.0,
        }

    # Merge in long-term annual features if the annual file is available
    if bps_dir:
        annual = _load_annual_county_file(bps_dir)
        for fips, long_feats in annual.items():
            if fips in features:
                features[fips].update(long_feats)
            else:
                # County in annual file but no monthly data — include with defaults
                features[fips] = {
                    "permits_last12m": 0, "permits_yoy_growth": 0.0,
                    "permits_3yr_cagr": 0.0, **long_feats,
                }

    logger.info(f"BPS: computed permit features for {len(features)} Utah counties.")
    return features


def get_county_fips(city: str) -> str | None:
    """Return 3-digit county FIPS for a Utah city name, or None if unknown."""
    return CITY_TO_COUNTY_FIPS.get(city.strip().lower())


def permit_feature_summary(permit_features: dict[str, dict]) -> str:
    """Return a human-readable summary of permit features by county."""
    if not permit_features:
        return "No permit features available."
    lines = ["County permit features (Utah):"]
    for fips, feats in sorted(permit_features.items()):
        name = UTAH_COUNTY_NAMES.get(fips, f"FIPS {fips}")
        lines.append(
            f"  {name:15s} ({fips})  "
            f"last12m={feats['permits_last12m']:4d}  "
            f"YoY={feats['permits_yoy_growth']:+.1%}  "
            f"3yr_CAGR={feats['permits_3yr_cagr']:+.1%}"
        )
    return "\n".join(lines)
