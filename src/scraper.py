"""
Redfin listing scraper — multi-page, multi-source, deduplicating.

Strategies used to maximize data volume:
  1. Pagination — loops through up to MAX_PAGES per query
  2. sold_within_days=730 — 2 years of sold comps for model training
  3. Multiple property types — single-family + townhomes (uipt 1,2)
  4. Deduplication — drops duplicate MLS# rows across all sources

Manual CSV download (fastest first run):
  1. Go to https://www.redfin.com/city/18736/UT/Saratoga-Springs
  2. Switch filter to "Sold" and set time range to "Last 2 years"
  3. Click the download button at the bottom of the map
  4. Save to data/raw/saratoga_springs_sold.csv
  5. Repeat for Eagle Mountain, Lehi, American Fork, etc.
  6. Call load_all_raw_csv() to merge everything.

NOTE: The gis-csv API endpoint (used by fetch_listings) frequently returns
listings without sale prices due to MLS restrictions. For model training,
manually downloaded CSVs are strongly preferred over the live API.
"""

import io
import json
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

REDFIN_BASE = "https://www.redfin.com"
SEARCH_URL  = f"{REDFIN_BASE}/stingray/api/gis-csv"
STINGRAY_BASE = f"{REDFIN_BASE}/stingray/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.redfin.com/",
}

# City → Redfin region ID (region_type 6 = city)
# IDs sourced from redfin.com/city/<id>/UT/<CityName> URLs.
# Cities too small for a Redfin city page (Elk Ridge, Lake Point, Stansbury Park,
# Daniel, Payson UT, Nephi) are covered only by manual CSV download.
CITY_REGIONS = {
    # Utah County — north
    "Saratoga Springs": {"region_id": "18736", "region_type": "6"},
    "Eagle Mountain":   {"region_id": "17872", "region_type": "6"},
    "Lehi":             {"region_id": "18248", "region_type": "6"},
    "American Fork":    {"region_id": "16788", "region_type": "6"},
    "Cedar Hills":      {"region_id": "17262", "region_type": "6"},
    "Highland":         {"region_id": "17971", "region_type": "6"},
    "Bluffdale":        {"region_id": "17014", "region_type": "6"},
    "Herriman":         {"region_id": "17963", "region_type": "6"},
    "Riverton":         {"region_id": "18601", "region_type": "6"},
    # Utah County — south (Spanish Fork corridor)
    "Spanish Fork":     {"region_id": "18207", "region_type": "6"},
    "Mapleton":         {"region_id": "12244", "region_type": "6"},
    "Santaquin":        {"region_id": "17357", "region_type": "6"},
    # Wasatch County
    "Heber City":       {"region_id": "8736",  "region_type": "6"},  # Redfin page: "Heber"
    # Tooele County
    "Tooele":           {"region_id": "19397", "region_type": "6"},
    "Grantsville":      {"region_id": "7960",  "region_type": "6"},
}

# How far back to look for sold listings
SOLD_WITHIN_DAYS = 730   # 2 years
MAX_PAGES        = 5     # pages per query (350 results/page = up to 1,750 per source)
PAGE_SIZE        = 350
SLEEP_BETWEEN_REQUESTS = 1.8  # seconds — be a polite scraper


def _parse_redfin_csv(text: str) -> Optional[pd.DataFrame]:
    """
    Parse Redfin's CSV response.

    Redfin's format has changed over time. Currently:
      Line 0: Header (starts with SALE TYPE or MLS#)
      Line 1: Disclaimer (embedded as a junk row — skipped via on_bad_lines)
      Line 2+: Data rows

    Older format had a disclaimer *before* the MLS# header row.
    Both are handled here.
    """
    lines = text.splitlines()
    # Find the header row — starts with SALE TYPE (new) or MLS# (old)
    start = next(
        (i for i, line in enumerate(lines)
         if line.startswith("SALE TYPE") or
            line.startswith("MLS#") or
            line.startswith('"MLS#')),
        None,
    )
    if start is None:
        logger.debug("Could not find CSV header row in Redfin response.")
        return None

    csv_text = "\n".join(lines[start:])
    try:
        df = pd.read_csv(io.StringIO(csv_text), on_bad_lines="skip")
        df = df.dropna(how="all")
        # Drop the disclaimer row that gets parsed as a junk data row
        if "SALE TYPE" in df.columns:
            df = df[df["SALE TYPE"].notna() & ~df["SALE TYPE"].str.startswith("In accordance", na=False)]
        return df if len(df) > 0 else None
    except Exception as e:
        logger.debug(f"CSV parse error: {e}")
        return None


def _fetch_page(
    region_id: str,
    region_type: str,
    sold: bool,
    page: int,
) -> Optional[pd.DataFrame]:
    """Fetch a single page from the Redfin stingray API."""
    status = "9" if sold else "1"

    params = {
        "al":             "1",
        "market":         "utah",
        "num_homes":      str(PAGE_SIZE),
        "ord":            "redfin-recommended-asc",
        "page_number":    str(page),
        "region_id":      region_id,
        "region_type":    region_type,
        "sf":             "1,2,3,5,6,7",   # listing types to include
        "status":         status,
        "uipt":           "1,2",            # 1=single family, 2=townhouse/condo
        "v":              "8",
    }
    if sold:
        params["sold_within_days"] = str(SOLD_WITHIN_DAYS)

    try:
        resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        df = _parse_redfin_csv(resp.text)
        return df
    except requests.HTTPError as e:
        logger.warning(f"HTTP {e.response.status_code} on page {page} "
                       f"(region {region_id}, sold={sold})")
        return None
    except Exception as e:
        logger.warning(f"Request failed (region {region_id}, page {page}): {e}")
        return None


def _fetch_all_pages(
    label: str,
    region_id: str,
    region_type: str,
    sold: bool,
    city_tag: str,
) -> pd.DataFrame:
    """
    Paginate through all available pages for one region/status combination.
    Stops when a page returns fewer rows than PAGE_SIZE (last page reached).
    """
    frames = []
    for page in range(1, MAX_PAGES + 1):
        df = _fetch_page(region_id, region_type, sold, page)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        if df is None or len(df) == 0:
            break

        # Redfin region IDs are unstable and can silently resolve to out-of-state
        # locations — validate and drop non-Utah rows.
        if "STATE OR PROVINCE" in df.columns:
            ut_rows = df[df["STATE OR PROVINCE"].str.upper() == "UT"]
            if len(ut_rows) == 0:
                states = df["STATE OR PROVINCE"].dropna().unique().tolist()
                logger.warning(
                    f"  [{label}] page {page}: {len(df)} rows from {states} — "
                    "not Utah, skipping. Use manual CSV download for reliable data."
                )
                break
            df = ut_rows

        if "CITY" in df.columns and df["CITY"].notna().any():
            df["city"] = df["CITY"].fillna(city_tag)
        else:
            df["city"] = city_tag
        df["sold"] = sold
        frames.append(df)

        logger.info(
            f"  [{label}] page {page}: {len(df)} rows  "
            f"(running total: {sum(len(f) for f in frames)})"
        )

        # If we got a full page there might be more; otherwise we're done
        if len(df) < PAGE_SIZE:
            break

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate listings by MLS#. Keeps the row with most non-null fields."""
    mls_col = next((c for c in df.columns if "MLS" in c.upper() and "#" in c), None)
    if mls_col is None:
        return df

    before = len(df)
    # Fill blanks so we can count non-nulls per row
    df["_nonnull"] = df.notna().sum(axis=1)
    df = (
        df.sort_values("_nonnull", ascending=False)
          .drop_duplicates(subset=[mls_col], keep="first")
          .drop(columns=["_nonnull"])
    )
    dropped = before - len(df)
    if dropped:
        logger.info(f"Deduplication removed {dropped} duplicate MLS# rows.")
    return df.reset_index(drop=True)


def fetch_listings(
    cities: list[str],
    include_sold: bool = True,
    save_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Fetch active + sold listings for target cities via the Redfin gis-csv API.

    Args:
        cities:       City names matching keys in CITY_REGIONS.
        include_sold: Whether to also fetch recently-sold comps (needed for training).
        save_dir:     Directory to write per-source CSV files.

    Returns:
        Combined, deduplicated DataFrame of all listings.

    Note:
        Sold listings returned by the API frequently have a blank PRICE column due
        to MLS data restrictions. For training data, manually downloaded CSVs from
        redfin.com are more reliable — see the module docstring for instructions.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    frames = []

    for city in cities:
        if city not in CITY_REGIONS:
            logger.warning(f"No region ID for '{city}' — skipping city search.")
            continue

        r = CITY_REGIONS[city]
        slug = city.lower().replace(" ", "_")

        logger.info(f"Fetching active listings — {city}")
        df_active = _fetch_all_pages(city, r["region_id"], r["region_type"],
                                     sold=False, city_tag=city)
        if not df_active.empty:
            df_active.to_csv(f"{save_dir}/{slug}_active.csv", index=False)
            frames.append(df_active)
            logger.info(f"  {city} active total: {len(df_active)}")

        if include_sold:
            logger.info(f"Fetching sold listings ({SOLD_WITHIN_DAYS}d) — {city}")
            df_sold = _fetch_all_pages(city, r["region_id"], r["region_type"],
                                       sold=True, city_tag=city)
            if not df_sold.empty:
                df_sold.to_csv(f"{save_dir}/{slug}_sold.csv", index=False)
                frames.append(df_sold)
                logger.info(f"  {city} sold total: {len(df_sold)}")

    if not frames:
        logger.error(
            "No data fetched from any source.\n"
            "Try the manual CSV download method described at the top of this file."
        )
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Raw combined rows (before dedup): {len(combined):,}")
    combined = _deduplicate(combined)
    logger.info(f"After deduplication: {len(combined):,}")

    combined.to_csv(f"{save_dir}/all_listings_raw.csv", index=False)
    return combined


# ── Description enrichment via Redfin Stingray API ───────────────────────────

DESCRIPTION_SLEEP         = 1.2   # seconds between detail requests per worker
DESCRIPTION_SLEEP_FAILURE = 0.3   # shorter sleep when request fails (not worth being polite)
DESCRIPTION_WORKERS       = 8     # concurrent fetch threads
DESCRIPTION_CACHE_FILE    = "data/processed/description_cache.json"
# If this fraction of the first N probes fail, assume we're blocked and abort early
EARLY_ABORT_PROBE         = 20
EARLY_ABORT_THRESHOLD     = 0.90


def _stingray_request(endpoint: str, params: dict) -> dict | None:
    """Make a request to the Redfin Stingray API and parse the JSON response."""
    url = STINGRAY_BASE + endpoint
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        # Stingray responses are prefixed with "{}&&" to prevent JSON hijacking
        text = resp.text
        if text.startswith("{}&&"):
            text = text[4:]
        return json.loads(text)
    except (requests.HTTPError, json.JSONDecodeError, Exception) as e:
        logger.debug(f"Stingray request failed ({endpoint}): {e}")
        return None


def _get_property_id(url_path: str) -> tuple[int | None, int | None]:
    """
    Given a Redfin listing URL path (e.g. /UT/Lehi/123-Main-St/.../home/12345),
    call initialInfo to get propertyId and listingId.
    """
    data = _stingray_request("api/home/details/initialInfo", {"path": url_path})
    if not data or "payload" not in data:
        return None, None
    payload = data["payload"]
    return payload.get("propertyId"), payload.get("listingId")


def _get_listing_description(property_id: int) -> str | None:
    """
    Fetch the listing description/remarks via the belowTheFold endpoint.
    Returns the full text description, or None if unavailable.
    """
    data = _stingray_request("api/home/details/belowTheFold", {
        "propertyId": property_id,
        "accessLevel": 1,
        "pageType": 3,
    })
    if not data or "payload" not in data:
        return None

    payload = data["payload"]

    # Try multiple known locations for the description text
    for key_path in [
        ("listingRemarks",),
        ("publicRecordsInfo", "remarks"),
    ]:
        obj = payload
        for k in key_path:
            if isinstance(obj, dict):
                obj = obj.get(k)
            else:
                obj = None
                break
        if obj and isinstance(obj, str) and len(obj) > 10:
            return obj

    # Some responses nest it differently
    if isinstance(payload.get("listingRemarks"), dict):
        return payload["listingRemarks"].get("remarks") or payload["listingRemarks"].get("value")

    return None


def _load_description_cache() -> dict:
    """Load the on-disk description cache (URL → description text)."""
    cache_path = Path(DESCRIPTION_CACHE_FILE)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_description_cache(cache: dict) -> None:
    """Persist the description cache to disk."""
    cache_path = Path(DESCRIPTION_CACHE_FILE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_one_description(raw_url: str) -> str | None:
    """
    Fetch a single listing description given its full Redfin URL.
    Returns the description text, or None on failure.
    Sleeps after the request to rate-limit per worker.
    """
    url_path = raw_url
    if "redfin.com" in url_path:
        url_path = url_path.split("redfin.com", 1)[1]

    property_id, _ = _get_property_id(url_path)
    if property_id is None:
        time.sleep(DESCRIPTION_SLEEP_FAILURE)
        return None

    desc = _get_listing_description(property_id)
    time.sleep(DESCRIPTION_SLEEP)
    return desc


def fetch_listing_descriptions(
    df: pd.DataFrame,
    max_listings: int = 200,
    url_col: str = "url",
) -> pd.DataFrame:
    """
    Enrich a DataFrame with listing descriptions by hitting Redfin's detail API.

    Only fetches descriptions for rows that have a URL and don't already have one.
    Uses a persistent on-disk cache to avoid re-fetching across runs, concurrent
    workers to speed up fetching, and early abort when the API appears to be
    blocking requests.

    Args:
        df:            DataFrame with a URL column pointing to Redfin listing pages.
        max_listings:  Cap on how many listings to fetch (avoids hammering the API).
        url_col:       Name of the column containing Redfin URLs.

    Returns:
        DataFrame with 'description' column added.
    """
    df = df.copy()
    if "description" not in df.columns:
        df["description"] = None

    if url_col not in df.columns:
        logger.info("No URL column found — skipping description enrichment.")
        return df

    cache = _load_description_cache()
    cache_hits = 0
    for idx, row in df.iterrows():
        raw_url = str(row.get(url_col, "") or "")
        if raw_url in cache and pd.isna(df.at[idx, "description"]):
            df.at[idx, "description"] = cache[raw_url]
            cache_hits += 1

    if cache_hits:
        logger.info(f"Description cache: {cache_hits} hit(s) loaded from disk.")

    needs_desc = df[
        df["description"].isna() &
        df[url_col].notna() &
        (df[url_col].astype(str).str.len() > 10)
    ]

    if needs_desc.empty:
        logger.info("All listings already have descriptions, or no URLs available.")
        return df

    to_fetch = needs_desc.head(max_listings)
    total = len(to_fetch)
    logger.info(
        f"Fetching descriptions for {total} listings "
        f"(of {len(needs_desc)} needing enrichment, capped at {max_listings}), "
        f"{DESCRIPTION_WORKERS} workers..."
    )

    fetched = 0
    failed = 0
    aborted = False

    work_items = [(idx, str(row[url_col])) for idx, row in to_fetch.iterrows()]

    with ThreadPoolExecutor(max_workers=DESCRIPTION_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_one_description, url): (idx, url)
            for idx, url in work_items
        }

        completed = 0
        for future in as_completed(futures):
            idx, raw_url = futures[future]
            completed += 1

            try:
                desc = future.result()
            except Exception as e:
                logger.debug(f"Description future raised: {e}")
                desc = None

            if desc:
                df.at[idx, "description"] = desc
                cache[raw_url] = desc
                fetched += 1
            else:
                failed += 1

            if completed == 1 or completed % 25 == 0 or completed == total:
                logger.info(
                    f"  Descriptions: {completed}/{total} "
                    f"({fetched} ok, {failed} failed)..."
                )

            # Early-abort: if first EARLY_ABORT_PROBE results are overwhelmingly
            # failing, Redfin is almost certainly blocking us — stop wasting time.
            if completed == EARLY_ABORT_PROBE and fetched == 0:
                fail_rate = failed / completed
                if fail_rate >= EARLY_ABORT_THRESHOLD:
                    logger.warning(
                        f"  Early abort: {fail_rate:.0%} failure rate after "
                        f"{EARLY_ABORT_PROBE} probes — Redfin appears to be blocking "
                        "description requests. Cancelling remaining fetches."
                    )
                    for f in futures:
                        f.cancel()
                    aborted = True
                    break

    if fetched:
        _save_description_cache(cache)
        logger.info(f"  Saved {fetched} new description(s) to cache.")

    skipped_cap = len(needs_desc) - len(to_fetch)
    skipped_abort = (total - completed) if aborted else 0
    logger.info(
        f"Description enrichment complete: {fetched} fetched, {failed} failed, "
        f"{skipped_cap} skipped (over cap)"
        + (f", {skipped_abort} cancelled (early abort)" if aborted else "") + "."
    )
    return df


def _read_listing_csv(path: Path) -> pd.DataFrame:
    """
    Read a single listing CSV, handling both:
      - New Redfin format: header row 0, disclaimer row 1, data from row 2
      - Old / sample format: header row 0, data from row 1
    """
    # Peek at the first two lines to detect format
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
        second = fh.readline()

    is_new_redfin = (
        "SALE TYPE" in first or "SOLD DATE" in first
    ) and "In accordance" in second

    if is_new_redfin:
        return pd.read_csv(path, header=0, skiprows=[1],
                           on_bad_lines="skip", low_memory=False)
    return pd.read_csv(path, low_memory=False, on_bad_lines="skip")


def load_all_raw_csv(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and concatenate all listing CSV files in raw_dir (top-level only).
    Skips subdirectories (BPS Data, discarded, Downloaded Data, etc.) and
    the old combined all_listings_raw.csv to avoid double-counting.
    """
    skip_names = {"all_listings_raw.csv"}
    csv_files = [
        f for f in Path(raw_dir).glob("*.csv")
        if f.name not in skip_names
    ]

    if not csv_files:
        combined_path = Path(raw_dir) / "all_listings_raw.csv"
        if combined_path.exists():
            csv_files = [combined_path]
        else:
            raise FileNotFoundError(
                f"No CSV files found in {raw_dir}.\n"
                "Download listing CSVs from redfin.com — see data/DOWNLOAD_GUIDE.md."
            )

    frames = []
    for f in sorted(csv_files):
        try:
            df = _read_listing_csv(f)

            if "city" not in df.columns and "CITY" not in df.columns:
                name = (f.stem
                        .replace("_active", "").replace("_sold", "")
                        .replace("_and_", " & ").replace("_", " ")
                        .title())
                df["city"] = name

            if "sold" not in df.columns:
                df["sold"] = "_sold" in f.stem

            listing_signals = {"PRICE", "BEDS", "SALE TYPE", "price", "beds"}
            if not listing_signals.intersection(df.columns):
                logger.info(f"Skipping {f.name} — not a listing CSV (no PRICE/BEDS column).")
                continue

            frames.append(df)
            logger.info(f"Loaded {f.name}: {len(df):,} rows")
        except Exception as e:
            logger.warning(f"Could not load {f.name}: {e}")

    if not frames:
        raise FileNotFoundError(f"No readable CSV files in {raw_dir}.")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total rows loaded: {len(combined):,}")
    return combined
