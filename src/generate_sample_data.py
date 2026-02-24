"""
Sample data generator for testing the pipeline without live Redfin data.

Generates realistic Utah County home listings (active + sold) based on
real market characteristics for Saratoga Springs, Eagle Mountain, and
surrounding cities.

Run: python -m src.generate_sample_data
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

# Real geographic bounding boxes for each city
CITY_CONFIGS = {
    "Saratoga Springs": {
        "lat_range": (40.31, 40.37),
        "lon_range": (-111.93, -111.85),
        "base_price": 450_000,
        "price_std": 80_000,
        "zip_codes": ["84045"],
        "n_active": 70,
        "n_sold": 140,
    },
    "Eagle Mountain": {
        "lat_range": (40.27, 40.35),
        "lon_range": (-112.03, -111.93),
        "base_price": 420_000,
        "price_std": 75_000,
        "zip_codes": ["84005"],
        "n_active": 80,
        "n_sold": 160,
    },
    "Lehi": {
        "lat_range": (40.37, 40.44),
        "lon_range": (-111.90, -111.82),
        "base_price": 510_000,
        "price_std": 100_000,
        "zip_codes": ["84043"],
        "n_active": 50,
        "n_sold": 120,
    },
    "American Fork": {
        "lat_range": (40.37, 40.40),
        "lon_range": (-111.82, -111.77),
        "base_price": 480_000,
        "price_std": 90_000,
        "zip_codes": ["84003"],
        "n_active": 40,
        "n_sold": 100,
    },
    "Cedar Hills": {
        "lat_range": (40.40, 40.44),
        "lon_range": (-111.77, -111.74),
        "base_price": 520_000,
        "price_std": 95_000,
        "zip_codes": ["84062"],
        "n_active": 25,
        "n_sold": 60,
    },
    "Highland": {
        "lat_range": (40.41, 40.45),
        "lon_range": (-111.82, -111.77),
        "base_price": 590_000,
        "price_std": 120_000,
        "zip_codes": ["84003"],
        "n_active": 30,
        "n_sold": 70,
    },
}


def _hedonic_price(sqft, beds, baths, year_built, lot_sqft, hoa, lat, lon, base_price, city):
    """Simulate a realistic hedonic pricing function with some noise."""
    price = base_price
    price += (sqft - 2000) * 120          # $120/extra sqft above 2000
    price += (beds - 3) * 8_000           # $8k per extra bed
    price += (baths - 2) * 12_000         # $12k per extra bath
    price += (year_built - 2000) * 2_500  # newer = more expensive
    price += (lot_sqft - 6000) * 3        # $3/extra sqft of lot
    price -= hoa * 80                     # HOA drags price slightly
    # Add geographic gradient (closer to I-15 / north = more expensive)
    price += (lat - 40.30) * 80_000
    # Add noise — some homes are overpriced, some are deals
    price *= RNG.normal(1.0, 0.07)
    return max(150_000, price)


def _generate_city(city: str, cfg: dict, sold: bool) -> pd.DataFrame:
    n = cfg["n_sold"] if sold else cfg["n_active"]
    lat_range = cfg["lat_range"]
    lon_range = cfg["lon_range"]
    base = cfg["base_price"]
    zip_codes = cfg["zip_codes"]

    # Generate home characteristics
    sqft      = RNG.integers(1400, 3800, n).astype(float)
    beds      = RNG.choice([3, 4, 4, 5, 5, 6], n)
    baths     = RNG.choice([2.0, 2.5, 3.0, 3.5, 4.0], n)
    year_built = RNG.integers(1999, 2024, n).astype(float)
    lot_sqft  = RNG.integers(4000, 12000, n).astype(float)
    hoa       = RNG.choice([0, 0, 0, 50, 75, 100, 150], n).astype(float)
    lat       = RNG.uniform(*lat_range, n)
    lon       = RNG.uniform(*lon_range, n)
    dom       = RNG.integers(0, 90, n).astype(float) if not sold else RNG.integers(0, 60, n).astype(float)
    zip_code  = RNG.choice(zip_codes, n)

    # Simulate true hedonic prices (what the model should learn)
    true_prices = np.array([
        _hedonic_price(sqft[i], beds[i], baths[i], year_built[i],
                       lot_sqft[i], hoa[i], lat[i], lon[i], base, city)
        for i in range(n)
    ])

    # For sold homes, listing price ≈ sold price (ground truth for training)
    # For active homes, inject some underpriced and overpriced listings
    if sold:
        prices = true_prices * RNG.normal(1.0, 0.02, n)
    else:
        # ~20% of listings are genuinely underpriced (deals), ~15% overpriced
        multipliers = RNG.normal(1.0, 0.05, n)
        deals_mask = RNG.random(n) < 0.20
        overpriced_mask = (~deals_mask) & (RNG.random(n) < 0.15)
        multipliers[deals_mask] *= RNG.uniform(0.85, 0.95, deals_mask.sum())
        multipliers[overpriced_mask] *= RNG.uniform(1.06, 1.15, overpriced_mask.sum())
        prices = true_prices * multipliers

    # Build street addresses
    streets = [
        "Maple Dr", "Oak Ave", "Cedar St", "Birch Ln", "Aspen Way",
        "Juniper Ct", "Willow Rd", "Pine St", "Elm Blvd", "Spruce Way",
        "Valley View Dr", "Mountain Rd", "Sunset Blvd", "Lakeside Dr",
    ]
    house_nums = RNG.integers(100, 9999, n)
    addresses = [f"{house_nums[i]} {RNG.choice(streets)}" for i in range(n)]

    mls_ids = [f"UT{RNG.integers(100000, 999999)}" for _ in range(n)]
    redfin_ids = [f"{RNG.integers(10000000, 99999999)}" for _ in range(n)]

    df = pd.DataFrame({
        "MLS#": mls_ids,
        "PRICE": prices.round(0),
        "BEDS": beds,
        "BATHS": baths,
        "LOCATION": addresses,
        "ZIP OR POSTAL CODE": zip_code,
        "SQUARE FEET": sqft,
        "LOT SIZE": lot_sqft,
        "YEAR BUILT": year_built,
        "DAYS ON MARKET": dom,
        "$/SQUARE FEET": (prices / sqft).round(0),
        "HOA/MONTH": hoa,
        "STATUS": "Sold" if sold else "Active",
        "URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)":
            [f"https://www.redfin.com/UT/{city.replace(' ', '-')}/home/{redfin_ids[i]}"
             for i in range(n)],
        "LATITUDE": lat.round(6),
        "LONGITUDE": lon.round(6),
        "city": city,
        "sold": sold,
    })
    return df


def generate(save_dir: str = "data/raw", verbose: bool = True) -> pd.DataFrame:
    """
    Generate sample listings for all configured cities and save to CSV.
    Returns the combined DataFrame.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    frames = []

    for city, cfg in CITY_CONFIGS.items():
        active = _generate_city(city, cfg, sold=False)
        sold   = _generate_city(city, cfg, sold=True)

        slug = city.lower().replace(" ", "_")
        active.to_csv(f"{save_dir}/{slug}_active.csv", index=False)
        sold.to_csv(f"{save_dir}/{slug}_sold.csv", index=False)

        if verbose:
            print(f"  {city}: {len(active)} active, {len(sold)} sold")
        frames.append(active)
        frames.append(sold)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(f"{save_dir}/all_listings_raw.csv", index=False)
    if verbose:
        print(f"\nTotal listings: {len(combined):,}")
        print(f"Saved to {save_dir}/")
    return combined


if __name__ == "__main__":
    print("Generating sample Utah County home listings...")
    generate()
    print("Done. Run: python main.py --run-now --no-email")
