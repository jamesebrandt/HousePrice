"""
Census ACS Median Household Income by ZIP Code loader.

Reads the wide-format Census Bureau ACS export (table B19013):
  data/raw/Census Data/acs_median_income_by_zip_2023.csv

The Census download format is transposed — ZIPs are columns, the single
data row is the estimate. This loader melts it into a tidy
  zip_code (str, 5-digit) → median_household_income (float)
lookup that feature engineering can join onto listings by zip_code.
"""

import logging
import os
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_acs_income(census_dir: str) -> dict[str, float]:
    """
    Parse the ACS B19013 median household income CSV and return:
        { zip_code: median_income, ... }   (zip_code is a 5-digit str)

    Returns an empty dict if the file is not found.
    """
    path = os.path.join(census_dir, "acs_median_income_by_zip_2023.csv")
    if not os.path.exists(path):
        logger.warning(f"ACS income file not found at {path}. Skipping income feature.")
        return {}

    try:
        # BOM-safe read; wide format — each column is a ZCTA5 XXXXX!!Estimate/MOE pair
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

        income_map: dict[str, float] = {}

        for col in df.columns:
            # Only process "Estimate" columns; skip "Margin of Error"
            m = re.match(r"ZCTA5\s+(\d{5})!!Estimate", col.strip())
            if not m:
                continue
            zip_code = m.group(1)

            raw_val = str(df[col].iloc[0]).strip()
            # Census uses "-" for suppressed/no data; strip commas from numbers
            if raw_val in {"-", "N", "(X)", "**", ""}:
                continue
            cleaned = raw_val.replace(",", "").replace("+", "")
            try:
                income_map[zip_code] = float(cleaned)
            except ValueError:
                continue

        logger.info(
            f"ACS: loaded median household income for {len(income_map):,} ZIP codes."
        )
        return income_map

    except Exception as e:
        logger.warning(f"Could not load ACS income file: {e}")
        return {}


def acs_income_summary(income_map: dict[str, float], zip_codes: list[str] | None = None) -> str:
    """Return a readable summary, optionally filtered to a list of ZIPs."""
    if not income_map:
        return "No ACS income data loaded."

    if zip_codes:
        subset = {z: income_map[z] for z in zip_codes if z in income_map}
    else:
        subset = income_map

    if not subset:
        return "ACS income data loaded but none of the target ZIPs matched."

    lines = [f"ACS median income — {len(subset)} ZIPs:"]
    for z, inc in sorted(subset.items(), key=lambda x: -x[1])[:15]:
        lines.append(f"  ZIP {z}: ${inc:,.0f}")
    if len(subset) > 15:
        lines.append(f"  ... and {len(subset) - 15} more")
    return "\n".join(lines)
