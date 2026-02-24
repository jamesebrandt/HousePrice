# Getting Real Redfin Data — Manual Download Guide

The automated Redfin scraper works when Redfin's internal API is accessible,
but their region IDs and rate limits change. **Manual CSV download is always
100% reliable** and takes about 5 minutes.

## Step-by-Step Instructions

### For Each City Below:

**Do this twice per city — once for Active listings, once for Sold listings.**

---

### 1. Saratoga Springs, UT

- **Active:** https://www.redfin.com/city/18736/UT/Saratoga-Springs
  - At the bottom of the map, click **"Download All (CSV)"**
  - Save as `data/raw/saratoga_springs_active.csv`

- **Sold (last 2 years):** https://www.redfin.com/city/18736/UT/Saratoga-Springs/filter/include=sold-2yr
  - Download → Save as `data/raw/saratoga_springs_sold.csv`

---

### 2. Eagle Mountain, UT

- **Active:** https://www.redfin.com/city/17872/UT/Eagle-Mountain
  - Download → Save as `data/raw/eagle_mountain_active.csv`

- **Sold:** https://www.redfin.com/city/17872/UT/Eagle-Mountain/filter/include=sold-2yr
  - Download → Save as `data/raw/eagle_mountain_sold.csv`

---

### 3. Lehi, UT (training data — more data = better model)

- **Active:** https://www.redfin.com/city/18248/UT/Lehi
  - Save as `data/raw/lehi_active.csv`

- **Sold:** https://www.redfin.com/city/18248/UT/Lehi/filter/include=sold-2yr
  - Save as `data/raw/lehi_sold.csv`

---

### 4. American Fork, UT

- **Sold:** https://www.redfin.com/city/16788/UT/American-Fork/filter/include=sold-2yr
  - Save as `data/raw/american_fork_sold.csv`

---

### 5. Highland, UT

- **Sold:** https://www.redfin.com/city/17971/UT/Highland/filter/include=sold-2yr
  - Save as `data/raw/highland_sold.csv`

---

### 6. Riverton, UT

- **Sold:** https://www.redfin.com/city/18601/UT/Riverton/filter/include=sold-2yr
  - Save as `data/raw/riverton_sold.csv`

---

## After Downloading

Run the full pipeline with your real data:

```bash
source venv/bin/activate
python main.py --run-now --no-email --retrain
```

This will:
1. Load all CSVs from `data/raw/`
2. Retrain the model on the real sold comps
3. Score all active listings
4. Print the top deals to your terminal

## Tips

- **More sold data = better model.** The sold CSVs are the training data for the
  hedonic pricing model. Aim for 500+ sold listings total across all cities.

- **Redfin limits downloads to ~350 per search.** If a city has more, try splitting
  by price range (e.g. under $500k and over $500k) and combining the CSVs.

- **The "Sold 2yr" filter** gets you the last 2 years of closed sales,
  which gives the model recent market context.

- After downloading, delete `data/raw/all_listings_raw.csv` (the combined cache)
  so the pipeline regenerates it from your new files.
