"""
Automatic supplemental data feed updater.

Downloads the latest BPS, ZHVI, and ACS files when they're stale,
so the pipeline doesn't depend on manual downloads for routine updates.

URL patterns:
  BPS:  https://www2.census.gov/econ/bps/County/co{YY}{MM}c.txt
  ZHVI: https://files.zillowstatic.com/research/public_csvs/zhvi/
        City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
  ACS:  data.census.gov API (table B19013, ZIP-level)
"""

import logging
import os
import re
from datetime import date, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
}

BPS_BASE_URL = "https://www2.census.gov/econ/bps/County/"
ZHVI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)

# ── BPS ──────────────────────────────────────────────────────────────────────


def _latest_bps_on_disk(bps_dir: str) -> tuple[int, int] | None:
    """Return (year, month) of the latest co{YY}{MM}c.txt on disk, or None."""
    files = sorted(Path(bps_dir).glob("co*c.txt"))
    if not files:
        return None
    m = re.match(r"co(\d{2})(\d{2})c\.txt", files[-1].name)
    if not m:
        return None
    return 2000 + int(m.group(1)), int(m.group(2))


def _next_months(year: int, month: int, count: int = 3) -> list[tuple[int, int]]:
    """Return the next `count` (year, month) tuples after the given month."""
    result = []
    y, m = year, month
    for _ in range(count):
        m += 1
        if m > 12:
            m, y = 1, y + 1
        result.append((y, m))
    return result


def update_bps(bps_dir: str) -> list[str]:
    """
    Try to download any BPS monthly files newer than what's on disk.

    Census typically publishes with a 1-2 month lag, so we probe the next
    few months after the latest file we have. Returns a list of newly
    downloaded filenames.
    """
    Path(bps_dir).mkdir(parents=True, exist_ok=True)
    latest = _latest_bps_on_disk(bps_dir)

    if latest is None:
        today = date.today()
        start_y, start_m = today.year, today.month - 3
        if start_m <= 0:
            start_m += 12
            start_y -= 1
        candidates = _next_months(start_y, start_m - 1, count=4)
    else:
        candidates = _next_months(latest[0], latest[1], count=3)

    downloaded = []
    for y, m in candidates:
        yy = y % 100
        filename = f"co{yy:02d}{m:02d}c.txt"
        dest = Path(bps_dir) / filename
        if dest.exists():
            continue

        url = f"{BPS_BASE_URL}{filename}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 500:
                dest.write_bytes(resp.content)
                downloaded.append(filename)
                logger.info(f"BPS: downloaded {filename}")
            else:
                logger.debug(f"BPS: {filename} not yet available (HTTP {resp.status_code})")
        except Exception as e:
            logger.debug(f"BPS: failed to fetch {filename}: {e}")

    if not downloaded:
        logger.info("BPS: no new monthly files available from Census.")
    return downloaded


# ── ZHVI ─────────────────────────────────────────────────────────────────────


def update_zhvi(zhvi_dir: str) -> str | None:
    """
    Download the latest ZHVI city-level CSV from Zillow Research.

    Zillow hosts the current file at a stable URL; we download it and
    overwrite the existing file. Returns the filename if updated, else None.
    """
    Path(zhvi_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(zhvi_dir) / "City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"

    try:
        resp = requests.get(ZHVI_URL, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()

        content_len = int(resp.headers.get("content-length", 0))
        if content_len < 10_000:
            logger.warning("ZHVI: response too small, likely not a valid CSV.")
            return None

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        logger.info(f"ZHVI: downloaded {dest.name} ({content_len / 1_000_000:.1f} MB)")
        return dest.name

    except requests.HTTPError as e:
        logger.warning(f"ZHVI: download failed — HTTP {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"ZHVI: download failed — {e}")
        return None


# ── ACS ──────────────────────────────────────────────────────────────────────

# The Census data.census.gov API is complex and requires table-specific
# queries.  The ACS B19013 data is released annually and the download
# format is non-trivial (wide transposed CSV with ZCTA column headers).
# Rather than replicating that fragile pipeline, we log a clear message
# directing the user to the download page — this source only updates once
# per year.

ACS_DOWNLOAD_URL = "https://data.census.gov/table/ACSDT1Y2023.B19013"


def check_acs_update_available(census_dir: str) -> str | None:
    """
    Check if a newer ACS vintage may be available.

    ACS 1-year estimates are released each September for the prior year.
    Returns a message string if an update is likely available, else None.
    """
    today = date.today()
    existing = list(Path(census_dir).glob("acs_median_income*.csv")) if Path(census_dir).exists() else []

    current_year_on_disk = None
    for f in existing:
        m = re.search(r"(\d{4})", f.stem)
        if m:
            current_year_on_disk = int(m.group(1))

    # ACS 1-year for year Y is released ~September of Y+1
    latest_expected = today.year - 1 if today.month >= 10 else today.year - 2

    if current_year_on_disk and current_year_on_disk >= latest_expected:
        return None

    return (
        f"ACS income data on disk is from {current_year_on_disk or 'unknown year'}. "
        f"The {latest_expected} vintage should be available. "
        f"Download from: {ACS_DOWNLOAD_URL}"
    )


# ── Orchestrator ─────────────────────────────────────────────────────────────


def update_all_feeds(raw_dir: str = "data/raw") -> dict[str, list[str] | str | None]:
    """
    Attempt to auto-update all supplemental feeds.

    Returns a summary dict:
        {
            "bps": ["co2601c.txt", ...],  # newly downloaded files
            "zhvi": "filename.csv" | None,
            "acs": "message" | None,       # instructions if update available
        }
    """
    bps_dir = str(Path(raw_dir) / "BPS Data")
    zhvi_dir = str(Path(raw_dir) / "Zillow Data")
    census_dir = str(Path(raw_dir) / "Census Data")

    results: dict[str, list[str] | str | None] = {}

    logger.info("Checking for supplemental feed updates...")
    results["bps"] = update_bps(bps_dir)
    results["zhvi"] = update_zhvi(zhvi_dir)
    results["acs"] = check_acs_update_available(census_dir)

    if results["acs"]:
        logger.info(f"ACS: {results['acs']}")

    return results
