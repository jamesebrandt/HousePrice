"""
Data staleness checker.

Inspects each supplemental data source and returns structured results
indicating how fresh the data is and where to download an update when stale.

Staleness thresholds (configurable via config.yaml under data_freshness):
  bps_days  (default 45)  — Census BPS permit files; monthly release ~1-month lag
  zhvi_days (default 45)  — Zillow ZHVI city CSV; monthly release ~1-month lag
  acs_days  (default 400) — Census ACS income; annual release (~13 months)
  redfin_days (default 1) — Redfin listing CSVs; refreshed daily via --refetch
"""

import glob
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default thresholds in days; can be overridden from config.yaml
DEFAULT_THRESHOLDS = {
    "bps_days":    45,
    "zhvi_days":   45,
    "acs_days":    400,
    "redfin_days": 1,
}

# Download / info URLs shown in alert emails
DOWNLOAD_URLS = {
    "BPS Permits":       "https://www2.census.gov/econ/bps/County/",
    "Zillow ZHVI":       "https://www.zillow.com/research/data/",
    "Census ACS Income": "https://data.census.gov/table/ACSDT1Y2023.B19013",
}


def _thresholds(cfg: dict) -> dict:
    """Merge user config values over defaults."""
    t = DEFAULT_THRESHOLDS.copy()
    fresh_cfg = cfg.get("data_freshness", {}) if cfg else {}
    for k in t:
        if k in fresh_cfg:
            t[k] = int(fresh_cfg[k])
    return t


# ── Individual source checkers ──────────────────────────────────────────────

def check_redfin_staleness(raw_dir: str, threshold_days: int = 1) -> dict:
    """Return staleness info for the Redfin listing CSVs."""
    csv_files = [
        f for f in Path(raw_dir).glob("*.csv")
        if f.name != "all_listings_raw.csv"
    ]

    if not csv_files:
        return {
            "source": "Redfin Listings",
            "latest": None,
            "age_days": None,
            "is_stale": True,
            "message": "No Redfin listing CSVs found — run with --refetch.",
            "download_url": None,
        }

    newest_mtime = max(os.path.getmtime(f) for f in csv_files)
    newest_date  = datetime.fromtimestamp(newest_mtime).date()
    age_days     = (date.today() - newest_date).days
    is_stale     = age_days > threshold_days

    return {
        "source":       "Redfin Listings",
        "latest":       str(newest_date),
        "age_days":     age_days,
        "is_stale":     is_stale,
        "message":      (
            f"Listing CSVs last updated {newest_date} ({age_days} day(s) old). "
            "Stale — run with --refetch to pull fresh data."
            if is_stale
            else f"Listing CSVs are current ({newest_date})."
        ),
        "download_url": None,
    }


def check_bps_staleness(bps_dir: str, threshold_days: int = 45) -> dict:
    """Return staleness info for Census BPS monthly permit files."""
    files = sorted(glob.glob(os.path.join(bps_dir, "co*c.txt")))

    if not files:
        return {
            "source":       "BPS Permits",
            "latest":       None,
            "age_days":     None,
            "is_stale":     True,
            "message":      "No BPS monthly files (co*c.txt) found.",
            "download_url": DOWNLOAD_URLS["BPS Permits"],
        }

    latest_file = files[-1]
    basename    = os.path.basename(latest_file)
    m = re.match(r"co(\d{2})(\d{2})c\.txt", basename)

    if m:
        yy, mm   = int(m.group(1)), int(m.group(2))
        year     = 2000 + yy
        import calendar
        last_day = calendar.monthrange(year, mm)[1]
        data_end = date(year, mm, last_day)
        age_days = (date.today() - data_end).days
        is_stale = age_days > threshold_days

        if is_stale:
            # Compute which file we expect next
            next_mm   = mm + 1 if mm < 12 else 1
            next_yy   = (year + (1 if mm == 12 else 0)) - 2000
            next_file = f"co{next_yy:02d}{next_mm:02d}c.txt"
            message = (
                f"BPS data covers through {year}-{mm:02d} ({age_days} days old). "
                f"Newer file '{next_file}' may be available."
            )
        else:
            message = f"BPS data is current through {year}-{mm:02d} ({age_days} days old)."
    else:
        # Fall back to file modification time
        mtime    = os.path.getmtime(latest_file)
        mdate    = datetime.fromtimestamp(mtime).date()
        age_days = (date.today() - mdate).days
        is_stale = age_days > threshold_days
        message  = f"BPS latest file: {basename} ({age_days} days since modified)."

    return {
        "source":       "BPS Permits",
        "latest":       basename,
        "age_days":     age_days,
        "is_stale":     is_stale,
        "message":      message,
        "download_url": DOWNLOAD_URLS["BPS Permits"] if is_stale else None,
    }


def check_zhvi_staleness(zhvi_dir: str, threshold_days: int = 45) -> dict:
    """Return staleness info for the Zillow ZHVI city CSV."""
    zhvi_files = list(Path(zhvi_dir).glob("City_zhvi*.csv"))
    if not zhvi_files:
        zhvi_files = list(Path(zhvi_dir).glob("*zhvi*.csv"))

    if not zhvi_files:
        return {
            "source":       "Zillow ZHVI",
            "latest":       None,
            "age_days":     None,
            "is_stale":     True,
            "message":      "No Zillow ZHVI CSV found.",
            "download_url": DOWNLOAD_URLS["Zillow ZHVI"],
        }

    zhvi_file = zhvi_files[0]

    # Read only the header row to find the latest date column (fast)
    try:
        df = pd.read_csv(zhvi_file, nrows=0)
        date_cols = sorted(
            c for c in df.columns
            if re.match(r"\d{4}-\d{2}-\d{2}$", str(c))
        )
        if date_cols:
            latest_col = date_cols[-1]
            latest_d   = datetime.strptime(latest_col, "%Y-%m-%d").date()
            age_days   = (date.today() - latest_d).days
            is_stale   = age_days > threshold_days
            return {
                "source":       "Zillow ZHVI",
                "latest":       latest_col,
                "age_days":     age_days,
                "is_stale":     is_stale,
                "message":      (
                    f"ZHVI data current through {latest_col} ({age_days} days old). "
                    "New monthly data likely available."
                    if is_stale
                    else f"ZHVI data is current through {latest_col} ({age_days} days old)."
                ),
                "download_url": DOWNLOAD_URLS["Zillow ZHVI"] if is_stale else None,
            }
    except Exception as e:
        logger.warning(f"Could not inspect ZHVI column headers: {e}")

    # Fall back to file modification time
    mtime    = os.path.getmtime(zhvi_file)
    mdate    = datetime.fromtimestamp(mtime).date()
    age_days = (date.today() - mdate).days
    is_stale = age_days > threshold_days
    return {
        "source":       "Zillow ZHVI",
        "latest":       str(mdate),
        "age_days":     age_days,
        "is_stale":     is_stale,
        "message":      f"ZHVI file last modified {mdate} ({age_days} days ago).",
        "download_url": DOWNLOAD_URLS["Zillow ZHVI"] if is_stale else None,
    }


def check_acs_staleness(census_dir: str, threshold_days: int = 400) -> dict:
    """Return staleness info for the Census ACS median income CSV."""
    acs_files = list(Path(census_dir).glob("acs_median_income*.csv"))
    if not acs_files:
        acs_files = list(Path(census_dir).glob("*.csv"))

    if not acs_files:
        return {
            "source":       "Census ACS Income",
            "latest":       None,
            "age_days":     None,
            "is_stale":     True,
            "message":      "No ACS income CSV found.",
            "download_url": DOWNLOAD_URLS["Census ACS Income"],
        }

    acs_file = acs_files[0]
    mtime    = os.path.getmtime(acs_file)
    mdate    = datetime.fromtimestamp(mtime).date()
    age_days = (date.today() - mdate).days

    # Try to extract the survey year from the filename
    year_m   = re.search(r"(\d{4})", acs_file.stem)
    data_year = year_m.group(1) if year_m else "unknown"

    is_stale = age_days > threshold_days
    return {
        "source":       "Census ACS Income",
        "latest":       data_year,
        "age_days":     age_days,
        "is_stale":     is_stale,
        "message":      (
            f"ACS income data is from survey year {data_year} ({age_days} days old). "
            "A newer annual release may be available."
            if is_stale
            else f"ACS income data ({data_year}) is current enough ({age_days} days old)."
        ),
        "download_url": DOWNLOAD_URLS["Census ACS Income"] if is_stale else None,
    }


# ── Aggregated check ─────────────────────────────────────────────────────────

def check_all_staleness(raw_dir: str = "data/raw", cfg: dict = None) -> list[dict]:
    """
    Run staleness checks for all data sources that exist on disk.

    Args:
        raw_dir: Path to the raw data directory.
        cfg:     Full config dict (reads data_freshness thresholds if present).

    Returns:
        List of result dicts, one per data source.
    """
    t = _thresholds(cfg)
    results: list[dict] = []

    results.append(check_redfin_staleness(raw_dir, threshold_days=t["redfin_days"]))

    bps_dir = str(Path(raw_dir) / "BPS Data")
    if Path(bps_dir).exists():
        results.append(check_bps_staleness(bps_dir, threshold_days=t["bps_days"]))

    zhvi_dir = str(Path(raw_dir) / "Zillow Data")
    if Path(zhvi_dir).exists():
        results.append(check_zhvi_staleness(zhvi_dir, threshold_days=t["zhvi_days"]))

    census_dir = str(Path(raw_dir) / "Census Data")
    if Path(census_dir).exists():
        results.append(check_acs_staleness(census_dir, threshold_days=t["acs_days"]))

    return results


def stale_sources(results: list[dict], exclude_redfin: bool = False) -> list[dict]:
    """Return only the results where is_stale is True."""
    return [
        r for r in results
        if r["is_stale"] and not (exclude_redfin and r["source"] == "Redfin Listings")
    ]


def staleness_summary(results: list[dict]) -> str:
    """Return a human-readable log-friendly summary of all staleness results."""
    lines = ["Data freshness check:"]
    for r in results:
        icon = "STALE  " if r["is_stale"] else "OK     "
        lines.append(f"  [{icon}] {r['source']}: {r['message']}")
    return "\n".join(lines)
