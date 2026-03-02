"""
ADU (Accessory Dwelling Unit) detection, rent estimation, and mortgage offset scoring.

Detects whether a listing likely has (or could support) an ADU / mother-in-law
apartment using two complementary signals:

  1. **Keyword matching** on listing descriptions (strongest signal)
     Looks for phrases like "mother-in-law", "ADU", "basement apartment",
     "separate entrance", "kitchenette", "rental income", etc.

  2. **Structural heuristics** from numeric fields (weaker but always available)
     High bed+bath totals, excess bathrooms beyond a typical single-family home,
     large sqft relative to bed count, and other patterns that suggest a
     separate living unit is present.

The combined confidence (0.0–1.0) gates detection (is there an ADU?).
Rent is estimated independently based on the *inferred size* of the ADU
(bed/bath count, sqft), not scaled by confidence.
"""

import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ADU_KEYWORDS = [
    r"mother[\s\-]?in[\s\-]?law",
    r"\badu\b",
    r"accessory\s+dwelling",
    r"in[\s\-]?law\s+(suite|apt|apartment|quarters|unit)",
    r"basement\s+(apt|apartment|rental|suite|unit|living)",
    r"separate\s+(entrance|entry|living|unit|apartment|suite|quarters)",
    r"rental\s+(income|unit|suite|potential|opportunity)",
    r"\bkitchenette\b",
    r"guest\s+(house|suite|quarters|apartment|unit)",
    r"casita",
    r"granny\s+(flat|suite|unit)",
    r"secondary\s+(suite|unit|dwelling|kitchen)",
    r"private\s+(entrance|entry|suite|apartment|living)",
    r"two\s+kitchen",
    r"2nd\s+kitchen",
    r"second\s+kitchen",
    r"dual\s+(living|kitchen)",
    r"income\s+(property|potential|producing|opportunity)",
    r"house\s*hack",
    r"multi[\s\-]?gen(erational)?",
    r"lower[\s\-]?level\s+(apartment|suite|unit|living|rental)",
    r"walkout\s+(basement|apartment|suite|rental)",
    r"finished\s+basement\s+with\s+(kitchen|bath|bedroom|separate)",
    r"full\s+(kitchen|apartment)\s+in\s+(the\s+)?basement",
    r"basement\s+has\s+(its\s+own|a\s+separate|full)",
    r"(upstairs|downstairs|lower|upper)\s+unit",
    r"tenant",
    r"(live[\s\-]?in|rented|currently\s+renting)",
    r"separate\s+(laundry|washer|dryer|w/d)",
    r"(rambler|home)\s+with\s+(rental|income|apartment)",
]
_ADU_PATTERN = re.compile("|".join(_ADU_KEYWORDS), re.IGNORECASE)

# Strong signals get extra weight — these almost always mean a real ADU
_STRONG_KEYWORDS = [
    r"\badu\b",
    r"accessory\s+dwelling",
    r"mother[\s\-]?in[\s\-]?law",
    r"rental\s+(income|unit)",
    r"income\s+(property|producing)",
    r"house\s*hack",
    r"separate\s+(entrance|entry)",
    r"\bkitchenette\b",
    r"second\s+kitchen",
    r"2nd\s+kitchen",
    r"two\s+kitchen",
    r"full\s+(kitchen|apartment)\s+in\s+(the\s+)?basement",
    r"tenant",
    r"currently\s+renting",
]
_STRONG_PATTERN = re.compile("|".join(_STRONG_KEYWORDS), re.IGNORECASE)


def _keyword_score(description: str) -> float:
    """Score 0.0–1.0 based on ADU-related keywords found in listing description."""
    if not description or not isinstance(description, str):
        return 0.0

    matches = _ADU_PATTERN.findall(description)
    strong = _STRONG_PATTERN.findall(description)

    if not matches:
        return 0.0

    if len(strong) >= 3:
        return 1.0
    if len(strong) >= 2:
        return min(0.7 + 0.1 * len(matches), 1.0)
    if strong:
        return min(0.55 + 0.1 * len(matches), 0.95)

    return min(0.25 + 0.1 * len(matches), 0.65)


def _structural_score(row: pd.Series) -> float:
    """
    Score 0.0–1.0 based on structural indicators that suggest an ADU.

    Key heuristics:
      - Excess rooms:  A 9bd/6ba home almost certainly has a separate unit.
        Normal single-family homes rarely exceed 5bd/3.5ba.
      - Bath density:  >=1 bath per bed is unusual for single-family.
      - Large footprint with many rooms: 4000+ sqft with 7+ beds.
      - Excess baths:  More baths than expected for the bed count suggests
        a second full bathroom set (i.e. a separate unit).
    """
    score = 0.0
    beds = row.get("beds")
    baths = row.get("baths")
    sqft = row.get("sqft")

    if not (pd.notna(beds) and pd.notna(baths)):
        return 0.0

    total_rooms = beds + baths

    if total_rooms >= 14:
        score += 0.50
    elif total_rooms >= 11:
        score += 0.35
    elif total_rooms >= 9:
        score += 0.20

    # Normal SFH has ~0.5–0.6 baths/bed; excess suggests a second bathroom set
    if beds > 0:
        expected_baths = 1.0 + (beds - 1) * 0.5  # e.g. 4bd → 2.5ba expected
        excess_baths = baths - expected_baths
        if excess_baths >= 2.0:
            score += 0.20
        elif excess_baths >= 1.0:
            score += 0.10

    if beds > 0 and baths / beds >= 0.9:
        score += 0.10

    if pd.notna(sqft):
        if sqft >= 4000 and beds >= 7:
            score += 0.15
        elif sqft >= 3500 and beds >= 6:
            score += 0.10
        elif sqft >= 3000 and beds >= 5:
            score += 0.05

    return min(score, 1.0)


def detect_adu_potential(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ADU detection columns to a listings DataFrame.

    New columns:
      - adu_keyword_score:    0–1 from description keywords
      - adu_structural_score: 0–1 from bed/bath/sqft heuristics
      - adu_confidence:       combined score (0–1)
      - adu_likely:           True if confidence >= 0.3
    """
    df = df.copy()

    if "description" in df.columns:
        df["adu_keyword_score"] = df["description"].apply(_keyword_score)
    else:
        df["adu_keyword_score"] = 0.0

    df["adu_structural_score"] = df.apply(_structural_score, axis=1)

    # Max of both signals + small bonus when both agree
    kw = df["adu_keyword_score"]
    st = df["adu_structural_score"]
    base = pd.concat([kw, st], axis=1).max(axis=1)
    both_bonus = pd.concat([kw, st], axis=1).min(axis=1) * 0.3
    df["adu_confidence"] = (base + both_bonus).clip(0.0, 1.0)

    df["adu_likely"] = df["adu_confidence"] >= 0.3

    n_likely = df["adu_likely"].sum()
    n_keyword = (df["adu_keyword_score"] > 0).sum()
    n_structural = ((df["adu_keyword_score"] == 0) & (df["adu_structural_score"] >= 0.3)).sum()
    logger.info(
        f"ADU detection: {n_likely} likely ADU listings "
        f"({n_keyword} keyword-matched, {n_structural} structural-only)"
    )
    return df


def _estimate_adu_beds(row: pd.Series) -> int:
    """
    Estimate how many bedrooms are in the ADU portion of the home.

    Heuristic: a "normal" single-family home in Utah has 3–4 bedrooms.
    Beds beyond that likely belong to the ADU.  For very large homes (8+),
    assume roughly half the beds are in the ADU.
    """
    beds = row.get("beds")
    if not pd.notna(beds) or beds < 4:
        return 1  # minimum: assume at least a studio/1-bed ADU

    main_beds = min(4, math.ceil(beds * 0.5))
    adu_beds = max(1, int(beds - main_beds))
    return min(adu_beds, 5)  # cap at 5; beyond that the estimate gets unreliable


# Base monthly rent for a 1-bed ADU by ZIP in Utah County (2025–2026).
_UT_BASE_RENT_1BR = {
    "84003": 1050,   # American Fork
    "84004": 1100,   # Alpine / Highland
    "84005": 1000,   # Eagle Mountain
    "84043": 1050,   # Lehi
    "84045": 1050,   # Saratoga Springs
    "84062": 1000,   # Pleasant Grove
    "84065": 1100,   # Riverton / Bluffdale
    "84096": 1100,   # Herriman
    "84660": 950,    # Spanish Fork
    "84664": 950,    # Mapleton
    "84663": 900,    # Springville
    "84651": 900,    # Payson
    "84032": 1000,   # Heber City
    "84074": 900,    # Tooele
    "84029": 850,    # Grantsville
    "84009": 1100,   # South Jordan
    "84095": 1100,   # South Jordan
}
_DEFAULT_BASE_RENT_1BR = 1000

_BED_MULTIPLIER = {
    1: 1.00,
    2: 1.35,
    3: 1.65,
    4: 1.90,
    5: 2.10,
}


def estimate_adu_rent(
    df: pd.DataFrame,
    adu_rent_map: dict[str, float] | None = None,
    default_rent: float | None = None,
) -> pd.DataFrame:
    """
    Estimate monthly ADU rental income for listings flagged as likely ADUs.

    Rent is based on:
      1. ZIP-level 1BR base rent
      2. Estimated ADU bedroom count (scaled from total home bed count)
      3. Bedroom multiplier

    Confidence does NOT scale rent. Once a home is flagged as adu_likely,
    rent is estimated at face value based on the inferred unit size.
    """
    df = df.copy()
    rent_map = adu_rent_map or _UT_BASE_RENT_1BR
    fallback = default_rent or _DEFAULT_BASE_RENT_1BR

    def _get_rent(row):
        if not row.get("adu_likely", False):
            return 0.0

        zip5 = str(row.get("zip_code", ""))[:5]
        base_1br = rent_map.get(zip5, fallback)

        adu_beds = _estimate_adu_beds(row)
        multiplier = _BED_MULTIPLIER.get(adu_beds, 2.10)

        return round(base_1br * multiplier)

    df["estimated_adu_rent"] = df.apply(_get_rent, axis=1)
    df["estimated_adu_beds"] = df.apply(
        lambda r: _estimate_adu_beds(r) if r.get("adu_likely", False) else 0,
        axis=1,
    )
    return df


def _load_mortgage_rate(fred_path: str | None = None) -> float:
    """Load most recent 30-year fixed rate from FRED CSV, or return a fallback."""
    if fred_path is None:
        fred_path = str(Path(__file__).parent.parent / "data" / "raw" / "FRED Data" / "MORTGAGE30US.csv")

    try:
        rates = pd.read_csv(fred_path)
        latest = rates.iloc[-1]["MORTGAGE30US"]
        return float(latest)
    except Exception:
        logger.warning("Could not load FRED mortgage rate; defaulting to 6.5%.")
        return 6.5


def estimate_mortgage_payment(
    price: float,
    annual_rate_pct: float,
    down_payment_pct: float = 0.05,
    term_years: int = 30,
) -> float:
    """Monthly P&I payment for a fixed-rate mortgage."""
    loan = price * (1 - down_payment_pct)
    monthly_rate = (annual_rate_pct / 100) / 12
    n_payments = term_years * 12
    if monthly_rate == 0:
        return loan / n_payments
    return loan * (monthly_rate * (1 + monthly_rate) ** n_payments) / (
        (1 + monthly_rate) ** n_payments - 1
    )


def compute_adu_affordability(
    df: pd.DataFrame,
    adu_cfg: dict | None = None,
    fred_path: str | None = None,
) -> pd.DataFrame:
    """
    Add mortgage and ADU-adjusted affordability columns.

    New columns:
      - mortgage_rate:       annual rate used
      - estimated_mortgage:  monthly P&I
      - estimated_adu_rent:  monthly ADU income (0 if no ADU)
      - net_monthly_cost:    mortgage - ADU rent
      - monthly_savings:     how much ADU income offsets mortgage
    """
    cfg = adu_cfg or {}
    rate = cfg.get("mortgage_rate_pct") or _load_mortgage_rate(fred_path)
    down_pct = cfg.get("down_payment_pct", 5) / 100
    term = cfg.get("mortgage_term_years", 30)

    df = df.copy()
    df["mortgage_rate"] = rate

    if "price" in df.columns:
        df["estimated_mortgage"] = df["price"].apply(
            lambda p: estimate_mortgage_payment(p, rate, down_pct, term)
            if pd.notna(p) and p > 0 else np.nan
        )
    else:
        df["estimated_mortgage"] = np.nan

    if "estimated_adu_rent" not in df.columns:
        df["estimated_adu_rent"] = 0.0

    df["net_monthly_cost"] = df["estimated_mortgage"] - df["estimated_adu_rent"]
    df["monthly_savings"] = df["estimated_adu_rent"]

    n_offset = (df["estimated_adu_rent"] > 0).sum()
    if n_offset:
        avg_savings = df.loc[df["estimated_adu_rent"] > 0, "estimated_adu_rent"].mean()
        logger.info(
            f"ADU affordability: {n_offset} listings with ADU rent offset "
            f"(avg ${avg_savings:,.0f}/mo savings)"
        )

    return df


def adu_summary(df: pd.DataFrame) -> str:
    """Human-readable ADU detection summary for logging."""
    if "adu_likely" not in df.columns:
        return "ADU detection not run."

    total = len(df)
    likely = df["adu_likely"].sum()
    lines = [f"ADU Summary — {likely} of {total} listings flagged as likely ADU:"]

    if likely and "estimated_adu_rent" in df.columns:
        adu_rows = df[df["adu_likely"]]
        avg_rent = adu_rows["estimated_adu_rent"].mean()
        lines.append(f"  Avg estimated ADU rent: ${avg_rent:,.0f}/mo")

    if likely and "net_monthly_cost" in df.columns:
        adu_rows = df[df["adu_likely"]]
        avg_mortgage = adu_rows["estimated_mortgage"].mean()
        avg_net = adu_rows["net_monthly_cost"].mean()
        lines.append(f"  Avg mortgage (P&I):     ${avg_mortgage:,.0f}/mo")
        lines.append(f"  Avg net after ADU rent: ${avg_net:,.0f}/mo")

    if "adu_confidence" in df.columns:
        high = (df["adu_confidence"] >= 0.7).sum()
        med = ((df["adu_confidence"] >= 0.3) & (df["adu_confidence"] < 0.7)).sum()
        lines.append(f"  High confidence (≥70%): {high}")
        lines.append(f"  Medium confidence:      {med}")

    return "\n".join(lines)
