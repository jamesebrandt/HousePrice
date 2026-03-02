"""
ADU (Accessory Dwelling Unit) detection, rent estimation, and mortgage offset scoring.

Detects whether a listing likely has (or could support) an ADU / mother-in-law
apartment using two complementary signals:

  1. **Keyword matching** on listing descriptions (strongest signal)
     Looks for phrases like "mother-in-law", "ADU", "basement apartment",
     "separate entrance", "kitchenette", "rental income", etc.

  2. **Structural heuristics** from numeric fields (weaker but always available)
     High bath-to-bed ratios, large sqft relative to bed count, and
     extra bathrooms beyond what a typical home would have.

The combined confidence (0.0–1.0) feeds into:
  - `estimated_adu_rent`: ZIP-level or config-based monthly rent estimate
  - `estimated_mortgage`: standard 30-year fixed from FRED rate data
  - `net_monthly_cost`: mortgage minus ADU rent offset
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Keyword patterns for description-based ADU detection ─────────────────────

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
]
_ADU_PATTERN = re.compile("|".join(_ADU_KEYWORDS), re.IGNORECASE)

# Strong signals get extra weight
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
    if strong:
        return min(0.5 + 0.15 * len(strong), 1.0)
    return min(0.25 + 0.1 * len(matches), 0.7)


def _structural_score(row: pd.Series) -> float:
    """
    Score 0.0–0.5 based on structural indicators that suggest ADU potential.
    Capped at 0.5 because structure alone is a weak signal.
    """
    score = 0.0
    beds = row.get("beds")
    baths = row.get("baths")
    sqft = row.get("sqft")

    if pd.notna(beds) and pd.notna(baths) and beds > 0:
        bath_ratio = baths / beds
        if bath_ratio >= 1.0:
            score += 0.15
        elif bath_ratio >= 0.8:
            score += 0.05

    if pd.notna(beds) and pd.notna(baths):
        if baths >= 4 and beds >= 5:
            score += 0.10

    if pd.notna(sqft) and pd.notna(beds) and beds > 0:
        sqft_per_bed = sqft / beds
        if sqft_per_bed >= 700:
            score += 0.10
        elif sqft_per_bed >= 550:
            score += 0.05

    if pd.notna(sqft) and sqft >= 3500:
        score += 0.10

    return min(score, 0.5)


def detect_adu_potential(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ADU detection columns to a listings DataFrame.

    New columns:
      - adu_keyword_score:    0–1 from description keywords
      - adu_structural_score: 0–0.5 from bed/bath/sqft heuristics
      - adu_confidence:       combined score (0–1)
      - adu_likely:           True if confidence >= 0.3
    """
    df = df.copy()

    if "description" in df.columns:
        df["adu_keyword_score"] = df["description"].apply(_keyword_score)
    else:
        df["adu_keyword_score"] = 0.0

    df["adu_structural_score"] = df.apply(_structural_score, axis=1)

    # Combine: keyword signal dominates when available, structural adds on top
    df["adu_confidence"] = (df["adu_keyword_score"] + df["adu_structural_score"]).clip(0.0, 1.0)
    df["adu_likely"] = df["adu_confidence"] >= 0.3

    n_likely = df["adu_likely"].sum()
    n_keyword = (df["adu_keyword_score"] > 0).sum()
    n_structural = ((df["adu_keyword_score"] == 0) & (df["adu_structural_score"] >= 0.3)).sum()
    logger.info(
        f"ADU detection: {n_likely} likely ADU listings "
        f"({n_keyword} keyword-matched, {n_structural} structural-only)"
    )
    return df


# ── Rent estimation ──────────────────────────────────────────────────────────

# Utah County ADU rent ranges by ZIP prefix. These are conservative estimates
# for a 1-bed/1-bath basement apartment based on 2025-2026 local rental data.
_UT_ADU_RENT_BY_ZIP = {
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
_DEFAULT_ADU_RENT = 1000


def estimate_adu_rent(
    df: pd.DataFrame,
    adu_rent_map: dict[str, float] | None = None,
    default_rent: float | None = None,
) -> pd.DataFrame:
    """
    Estimate monthly ADU rental income for listings flagged as likely ADUs.

    Uses ZIP-level lookup, falling back to a config default.
    Only assigns rent to rows where adu_likely is True.
    """
    df = df.copy()
    rent_map = adu_rent_map or _UT_ADU_RENT_BY_ZIP
    fallback = default_rent or _DEFAULT_ADU_RENT

    def _get_rent(row):
        if not row.get("adu_likely", False):
            return 0.0
        zip5 = str(row.get("zip_code", ""))[:5]
        base = rent_map.get(zip5, fallback)
        return base * row.get("adu_confidence", 0.5)

    df["estimated_adu_rent"] = df.apply(_get_rent, axis=1)
    return df


# ── Mortgage estimation ──────────────────────────────────────────────────────

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
