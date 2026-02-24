# Home Finder

A Python project that automatically finds high-value homes in Saratoga Springs and Eagle Mountain, UT using a statistical hedonic pricing model.

## How It Works

1. **Fetch listings** from Redfin (active + recently sold)
2. **Clean & engineer features** (sqft, beds, baths, age, location, etc.)
3. **Train a hedonic pricing model** — learns what homes *should* cost based on comparable sales
4. **Score active listings** — computes how far below predicted market value each home is
5. **Filter** to homes matching your criteria (price, size, age, etc.)
6. **Email you** the top deals daily

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure secrets
cp .env.example .env
# Edit .env with your Gmail App Password

# 4. Edit config.yaml with your preferences

# 5. Run once (fetches data, trains model, prints top deals)
python main.py --run-now --no-email

# 6. Start daily alerts at 7am
python main.py --schedule --time 07:00
```

## Manual Data Download (Faster First Run)

If the automatic fetcher is slow or rate-limited:

1. Go to https://www.redfin.com/city/18736/UT/Saratoga-Springs
2. Click the **Download** button at the bottom of the map → save to `data/raw/saratoga_springs_active.csv`
3. Switch to "Sold" filter → download → save to `data/raw/saratoga_springs_sold.csv`
4. Repeat for Eagle Mountain: https://www.redfin.com/city/17872/UT/Eagle-Mountain
5. Run `python main.py --run-now --no-email`

## Notebooks

Work through these in order:

| Notebook | Purpose |
|---|---|
| `notebooks/01_eda.ipynb` | Explore price distributions, compare cities |
| `notebooks/02_model.ipynb` | Train & evaluate hedonic pricing models |
| `notebooks/03_scoring.ipynb` | Score active listings, find deals, preview email |

```bash
cd notebooks
jupyter notebook
```

## Configuration

Edit `config.yaml` to change your criteria:

```yaml
criteria:
  max_price: 500000      # Your maximum budget
  min_beds: 3
  min_baths: 2.0
  min_sqft: 1800
  max_home_age_years: 25 # No homes older than 25 years
  min_lot_sqft: 5000

scoring:
  min_value_score: 0.05  # Alert on homes 5%+ below predicted price
  top_n_alerts: 5        # How many homes per email
```

## Email Setup

1. Enable 2-factor authentication on your Google account
2. Create an App Password: https://myaccount.google.com/apppasswords
3. Add to `.env`:
   ```
   GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
   ```
4. Set `email_to` and `email_from` in `config.yaml`

## Project Structure

```
home-finder/
├── config.yaml          # Your preferences & criteria
├── requirements.txt
├── main.py              # Entry point (run-once or scheduler)
├── src/
│   ├── scraper.py       # Fetch Redfin listings
│   ├── features.py      # Clean data & engineer features
│   ├── model.py         # Train & load hedonic pricing model
│   ├── scorer.py        # Compute value scores
│   ├── filter.py        # Apply criteria filter
│   └── notifier.py      # Send HTML email alerts
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model.ipynb
│   └── 03_scoring.ipynb
├── data/
│   ├── raw/             # Redfin CSVs (gitignored)
│   └── processed/       # Cleaned data (gitignored)
└── models/              # Saved model files (gitignored)
```

## Value Score Explained

**Value Score** = `(predicted_price - actual_price) / predicted_price`

- `+0.10` means the home is listed 10% **below** what comparable homes sell for → potential deal
- `-0.05` means the home is listed 5% **above** comparable homes → overpriced
- `0.00` means fairly priced

The **Composite Score** blends value score (60%), sqft-per-dollar (25%), and rooms-per-dollar (15%) into a single ranking number.

## Retraining the Model

The model is trained once and saved. To retrain (recommended when you have new sold data):

```bash
python main.py --run-now --retrain
```
