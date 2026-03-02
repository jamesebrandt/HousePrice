"""
Model Diagnostics — saves a multi-panel PNG to diagnostics/model_diagnostics.png

Layout (3 rows × 3 columns):

  Row 1 — Core Fit
    [A] Predicted vs Actual, colored by city  — does it track reality?
    [B] % Prediction Error distribution       — practical accuracy bands
    [C] Feature Importances (top 15)          — what the model actually learned

  Row 2 — Residual Analysis
    [D] Residuals vs Fitted                   — checks heteroscedasticity & price-level bias
    [E] Accuracy by Price Tier                — MAPE broken down by quintile (price-range bias)
    [F] Spatial Residuals (lat/lon)           — geographic clustering of errors

  Row 3 — Location & Normality
    [G] City medians: Actual vs Predicted     — does it nail each market?
    [H] Residuals by City                     — systematic over/under-prediction
    [I] Normal Q-Q Plot                       — are residuals bell-shaped?

Run:  python -m src.model_diagnostics
"""

import sys
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper import load_all_raw_csv
from src.features import prepare_dataset, get_model_feature_cols
from src.model import load_model
from src.bps_loader import load_bps_data, compute_permit_features
from src.zhvi_loader import load_zhvi_data, compute_zhvi_features
from src.acs_loader import load_acs_income

logging.basicConfig(level=logging.WARNING)

OUTPUT_DIR  = Path("diagnostics")
OUTPUT_FILE = OUTPUT_DIR / "model_diagnostics.png"

# ── Colour palette ─────────────────────────────────────────────────────────────
P = {
    "blue":   "#2980b9", "green":  "#27ae60", "red":    "#e74c3c",
    "orange": "#e67e22", "purple": "#8e44ad", "gray":   "#95a5a6",
    "dark":   "#2c3e50", "teal":   "#16a085", "yellow": "#f39c12",
    "pink":   "#c0392b",
}

# City palette — up to 22 cities
CITY_COLORS = [
    "#2980b9","#27ae60","#e74c3c","#e67e22","#8e44ad",
    "#16a085","#f39c12","#c0392b","#1abc9c","#d35400",
    "#2ecc71","#9b59b6","#3498db","#e67e22","#1abc9c",
    "#e74c3c","#f1c40f","#2c3e50","#7f8c8d","#bdc3c7",
    "#8e44ad","#27ae60",
]

# Human-readable feature labels (raw name → display label)
FEAT_LABELS: dict[str, str] = {
    "sqft":                   "Square Feet",
    "beds":                   "Bedrooms",
    "baths":                  "Bathrooms",
    "lot_sqft":               "Lot Size (sqft)",
    "feat_home_age":          "Home Age (yrs)",
    "feat_sqft_per_bed":      "Sqft / Bedroom",
    "feat_lot_to_sqft_ratio": "Lot-to-Home Ratio",
    "feat_has_hoa":           "Has HOA",
    "hoa_monthly":            "HOA Monthly ($)",
    "days_on_market":         "Days on Market",
    "feat_long_on_market":    "Listed >60 Days",
    "latitude":               "Latitude",
    "longitude":              "Longitude",
    "feat_median_income":     "ZIP Median Income",
    "feat_permits_last12m":   "Permits Last 12mo",
    "feat_permits_yoy_growth":"Permits YoY Growth",
    "feat_permits_3yr_cagr":  "Permits 3yr CAGR",
    "feat_permits_10yr_cagr": "Permits 10yr CAGR",
    "feat_permits_long_trend":"Permits Long Trend",
    "feat_zhvi_current":      "ZHVI (Current)",
    "feat_zhvi_yoy_pct":      "ZHVI YoY %",
    "feat_zhvi_3yr_cagr":     "ZHVI 3yr CAGR",
    "feat_zhvi_5yr_cagr":     "ZHVI 5yr CAGR",
    "feat_zhvi_momentum":     "ZHVI Momentum",
}
# Feature categories for colour-coding in the importance chart
def _feat_category(name: str) -> str:
    if any(name.startswith(p) for p in ("city_", "zip_")):  return "location"
    if "zhvi"    in name:                                    return "market"
    if "permit"  in name:                                    return "demand"
    if "income"  in name:                                    return "wealth"
    if name in ("latitude", "longitude"):                    return "location"
    return "property"

CAT_COLORS = {
    "property": P["blue"],
    "location": P["teal"],
    "market":   P["green"],
    "demand":   P["orange"],
    "wealth":   P["purple"],
}
CAT_LABELS = {
    "property": "Property (beds/baths/sqft…)",
    "market":   "Market (ZHVI appreciation)",
    "demand":   "Demand (building permits)",
    "wealth":   "Wealth (ZIP income)",
    "location": "Location (city/zip dummies)",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _label(name: str) -> str:
    """Return display label for a feature name."""
    if name.startswith("city_"): return "City: " + name[5:].replace("_", " ")
    if name.startswith("zip_"):  return "ZIP " + name[4:]
    return FEAT_LABELS.get(name, name)


def _qq_data(residuals: np.ndarray):
    n = len(residuals)
    q  = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    return stats.norm.ppf(q), np.sort(residuals / residuals.std())


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    raw_dir: str  = "data/raw",
    model_path:   str = "models/hedonic_model.joblib",
    feature_path: str = "models/hedonic_model_features.joblib",
    holdout_path: str = "models/hedonic_model_holdout.joblib",
    config_path:  str = "config.yaml",
):
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load exclude_cities from config if present
    exclude_cities = None
    if Path(config_path).exists():
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        exclude_cities = cfg.get("search", {}).get("exclude_cities")

    print("Loading model…")
    model, feature_cols = load_model(model_path, feature_path)

    # ── Prefer the saved holdout for honest out-of-sample metrics ─────────────
    # If no holdout exists (e.g. model was loaded from a pre-improvement run),
    # fall back to the full prepared dataset with a clear in-sample warning.
    data_label: str
    log_target = False
    smearing_factor = 1.0
    if Path(holdout_path).exists():
        print(f"Loading holdout test data from {holdout_path}…")
        holdout = joblib.load(holdout_path)
        X    = holdout["X_test"].astype(float)
        log_target = holdout.get("log_target", False)
        smearing_factor = holdout.get("smearing_factor", 1.0)
        if log_target and "y_test_dollar" in holdout:
            y = holdout["y_test_dollar"].astype(float)
        else:
            y = holdout["y_test"].astype(float)
        cities = holdout["cities"]
        data_label = "Out-of-Sample Test Set"
        print(f"  {len(y):,} holdout rows (out-of-sample, log_target={log_target})")
    else:
        print(
            "⚠  No holdout data found — falling back to full prepared dataset.\n"
            "   Metrics are IN-SAMPLE and will be optimistic.\n"
            "   Re-run with --retrain to generate a proper test set."
        )
        prepared_path = Path("data/processed/prepared_listings.csv")
        if prepared_path.exists():
            df = pd.read_csv(prepared_path)
            print(f"  {len(df):,} rows from prepared_listings.csv")
        else:
            print("  Re-processing from raw CSVs…")
            raw = load_all_raw_csv(raw_dir)
            permit_features = compute_permit_features(
                load_bps_data(f"{raw_dir}/BPS Data"),
                bps_dir=f"{raw_dir}/BPS Data",
            )
            zhvi_features = compute_zhvi_features(
                load_zhvi_data(f"{raw_dir}/Zillow Data", state="UT")
            )
            income_map = load_acs_income(f"{raw_dir}/Census Data")
            df = prepare_dataset(raw, permit_features=permit_features,
                                 zhvi_features=zhvi_features, income_map=income_map,
                                 exclude_cities=exclude_cities)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        X = df[feature_cols].astype(float).values
        y = df["price"].values
        city_col = "city" if "city" in df.columns else None
        cities   = df[city_col].values if city_col else np.full(len(df), "unknown")
        data_label = "Full Dataset (IN-SAMPLE — metrics are optimistic)"

    # Load calibration factors from metrics JSON if available
    metrics_json = Path(model_path.replace(".joblib", "_metrics.json"))
    global_cal = 1.0
    city_cal: dict[str, float] = {}
    if metrics_json.exists():
        import json as _json
        with open(metrics_json) as _f:
            _meta = _json.load(_f)
        global_cal = _meta.get("calibration_factor", 1.0)
        city_cal = _meta.get("city_calibration", {})
        if abs(global_cal - 1.0) > 0.005 or city_cal:
            print(f"  Calibration factor (global): {global_cal:.4f}")

    y_pred_raw = model.predict(X)
    if log_target:
        y_pred = np.exp(y_pred_raw) * smearing_factor
    else:
        y_pred = y_pred_raw

    # Apply city-level calibration where available, global otherwise
    if city_cal:
        cal_factors = np.array([
            city_cal.get(str(c), global_cal) for c in cities
        ])
        y_pred = y_pred * cal_factors
    elif abs(global_cal - 1.0) > 0.005:
        y_pred = y_pred * global_cal

    residuals = y_pred - y

    # Metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae  = mean_absolute_error(y, y_pred)
    r2   = r2_score(y, y_pred)
    pct_err = (y_pred - y) / y * 100          # signed % error
    abs_pct = np.abs(pct_err)
    mape = abs_pct.mean()
    within10 = (abs_pct <= 10).mean() * 100
    within20 = (abs_pct <= 20).mean() * 100
    within30 = (abs_pct <= 30).mean() * 100

    sw_stat, sw_p = stats.shapiro(residuals[:min(len(residuals), 5000)])

    print(f"R²={r2:.3f}  RMSE=${rmse:,.0f}  MAPE={mape:.1f}%  [{data_label}]")
    print(f"Within ±10%: {within10:.0f}%  ±20%: {within20:.0f}%  ±30%: {within30:.0f}%")
    unique_cities = sorted(set(cities))
    city_color_map = {c: CITY_COLORS[i % len(CITY_COLORS)]
                      for i, c in enumerate(unique_cities)}

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22), facecolor="white")
    fig.suptitle(
        f"Home Price Model — Diagnostic Report  [{data_label}]\n"
        f"Ridge / RandomForest / XGBoost  ·  {len(y):,} listings  ·  "
        f"R² = {r2:.3f}  ·  RMSE = ${rmse/1e3:.0f}K  ·  MAPE = {mape:.1f}%",
        fontsize=15, fontweight="bold", color=P["dark"], y=0.995,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.48, wspace=0.35,
                           top=0.96, bottom=0.04,
                           left=0.07, right=0.97)

    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]
    (ax_A, ax_B, ax_C,
     ax_D, ax_E, ax_F,
     ax_G, ax_H, ax_I) = axes

    row_labels = [
        ("Row 1 — Core Fit",          0),
        ("Row 2 — Residual Analysis", 1),
        ("Row 3 — Location & Bias",   2),
    ]
    for label, row in row_labels:
        fig.text(0.005, 0.965 - row * 0.323, label,
                 fontsize=11, fontweight="bold", color=P["dark"],
                 va="top", rotation=90)

    # ── [A] Predicted vs Actual, coloured by city ──────────────────────────────
    lim_lo = min(y.min(), y_pred.min()) * 0.95
    lim_hi = max(y.max(), y_pred.max()) * 1.05

    for city in unique_cities:
        mask = cities == city
        ax_A.scatter(y[mask] / 1e3, y_pred[mask] / 1e3,
                     alpha=0.5, s=22, color=city_color_map[city],
                     edgecolors="white", linewidths=0.3, zorder=3)

    xs = np.array([lim_lo, lim_hi]) / 1e3
    ax_A.plot(xs, xs, "k--", linewidth=1.5, label="Perfect fit", zorder=4)
    ax_A.fill_between(xs, xs * 0.80, xs * 1.20, alpha=0.10, color=P["green"], label="±20% band")

    # Compact city legend (max 8 entries)
    handles = [Line2D([0],[0], marker="o", color="w",
                      markerfacecolor=city_color_map[c], markersize=7, label=c)
               for c in unique_cities[:8]]
    if len(unique_cities) > 8:
        handles.append(Line2D([0],[0], marker="", color="w",
                               label=f"…+{len(unique_cities)-8} more"))
    ax_A.legend(handles=handles, fontsize=7, ncol=2,
                loc="upper left", framealpha=0.85)

    ax_A.set_xlabel("Actual Price ($K)")
    ax_A.set_ylabel("Predicted Price ($K)")
    ax_A.set_title("[A] Predicted vs Actual  (by city)", fontweight="bold")
    ax_A.text(0.97, 0.05,
              f"R² = {r2:.3f}\nRMSE = ${rmse/1e3:.0f}K\nMAPE = {mape:.1f}%",
              transform=ax_A.transAxes, ha="right", va="bottom", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", alpha=0.9))

    # ── [B] % Error distribution with accuracy tiers ───────────────────────────
    clip = np.clip(pct_err, -80, 80)
    ax_B.hist(clip, bins=60, color=P["blue"], edgecolor="white",
              alpha=0.85, density=True)

    # Shade accuracy tiers
    for lo, hi, col, lbl in [
        (-10, 10, "#27ae60", "±10 %"),
        (-20, -10, "#f39c12", ""),
        (10, 20, "#f39c12", "±20 %"),
        (-30, -20, "#e74c3c", ""),
        (20, 30, "#e74c3c", "±30 %"),
    ]:
        ax_B.axvspan(lo, hi, alpha=0.15, color=col)
    ax_B.axvline(0, color="black", linewidth=1.2, linestyle="--")

    ax_B.set_xlabel("Prediction Error  (%)  [Predicted − Actual]")
    ax_B.set_ylabel("Density")
    ax_B.set_title("[B] Prediction Error Distribution", fontweight="bold")

    # Accuracy tier annotations
    tier_text = (
        f"Within ±10 %: {within10:.0f} %\n"
        f"Within ±20 %: {within20:.0f} %\n"
        f"Within ±30 %: {within30:.0f} %"
    )
    ax_B.text(0.97, 0.97, tier_text,
              transform=ax_B.transAxes, ha="right", va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", alpha=0.9))

    tier_legend = [
        Patch(facecolor="#27ae60", alpha=0.4, label="±10 %"),
        Patch(facecolor="#f39c12", alpha=0.4, label="10–20 %"),
        Patch(facecolor="#e74c3c", alpha=0.4, label="20–30 %"),
    ]
    ax_B.legend(handles=tier_legend, fontsize=8, loc="upper left")

    # ── [C] Feature Importances (top 15, colour-coded by category) ────────────
    last_step = model[-1] if hasattr(model, "__getitem__") else model
    imp_attr = "feature_importances_" if hasattr(last_step, "feature_importances_") else None

    if imp_attr:
        imp_df = pd.DataFrame({
            "feature":    feature_cols,
            "importance": last_step.feature_importances_,
        }).sort_values("importance").tail(15)

        imp_df["label"]    = imp_df["feature"].map(_label)
        imp_df["category"] = imp_df["feature"].map(_feat_category)
        bar_colors = imp_df["category"].map(CAT_COLORS)

        ax_C.barh(imp_df["label"], imp_df["importance"],
                  color=bar_colors, edgecolor="white", height=0.7)

        # Value annotations
        for _, row in imp_df.iterrows():
            ax_C.text(row["importance"] + imp_df["importance"].max() * 0.01,
                      row.name if isinstance(row.name, (int, float)) else
                      list(imp_df.index).index(row.name),
                      f"{row['importance']:.3f}", va="center", fontsize=7.5,
                      color=P["dark"])

        ax_C.set_xlabel("Importance Score")
        ax_C.set_title("[C] Feature Importances  (top 15)", fontweight="bold")

        cat_legend = [Patch(facecolor=col, alpha=0.85, label=CAT_LABELS[cat])
                      for cat, col in CAT_COLORS.items()
                      if cat in imp_df["category"].values]
        ax_C.legend(handles=cat_legend, fontsize=7.5, loc="lower right")
    else:
        ax_C.text(0.5, 0.5, "Feature importances\nnot available",
                  ha="center", va="center", transform=ax_C.transAxes, fontsize=12)
        ax_C.set_title("[C] Feature Importances", fontweight="bold")

    # ── [D] Residuals vs Fitted ─────────────────────────────────────────────────
    # Standard regression diagnostic: checks for heteroscedasticity (variance
    # that grows with price) and any price-level bias in predictions.
    # A good model shows a flat cloud of points centered on the zero line.
    for city in unique_cities:
        mask = cities == city
        ax_D.scatter(y_pred[mask] / 1e3, residuals[mask] / 1e3,
                     alpha=0.35, s=16, color=city_color_map[city],
                     edgecolors="none", zorder=3)

    ax_D.axhline(0, color="red", linestyle="--", linewidth=1.5, label="No error", zorder=4)
    ax_D.axhspan(-50,  50, alpha=0.06, color=P["green"])

    # Binned mean trend — reveals any systematic curve or funnel
    _y_sorted_idx = np.argsort(y_pred)
    _yp_s = y_pred[_y_sorted_idx]
    _res_s = residuals[_y_sorted_idx]
    n_bins_D = 16
    bin_edges_D = np.percentile(_yp_s, np.linspace(5, 95, n_bins_D + 1))
    bin_edges_D = np.unique(bin_edges_D)
    bin_x_D, bin_y_D = [], []
    for _i in range(len(bin_edges_D) - 1):
        _bm = (_yp_s >= bin_edges_D[_i]) & (_yp_s <= bin_edges_D[_i + 1])
        if _bm.sum() >= 5:
            bin_x_D.append(_yp_s[_bm].mean() / 1e3)
            bin_y_D.append(_res_s[_bm].mean() / 1e3)
    if bin_x_D:
        ax_D.plot(bin_x_D, bin_y_D, "k-", linewidth=2.5,
                  label="Binned mean trend", zorder=5, alpha=0.85)

    ax_D.set_xlabel("Fitted (Predicted) Price ($K)")
    ax_D.set_ylabel("Residual ($K)  [Predicted − Actual]")
    ax_D.set_title("[D] Residuals vs Fitted\n(Flat trend = no price-level bias; even spread = homoscedastic)",
                   fontweight="bold")
    ax_D.legend(fontsize=8, loc="upper left")
    ax_D.text(0.97, 0.97,
              "✓ Good: flat cloud near 0\n✗ Bad: funnel = heteroscedastic\n"
              "   curve = price-level bias",
              transform=ax_D.transAxes, ha="right", va="top", fontsize=8.5,
              color=P["dark"],
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.85))

    # ── [E] Accuracy by Price Tier ──────────────────────────────────────────────
    # Breaks MAPE down by price quintile to reveal any price-range bias —
    # a common failure mode in hedonic models where cheap/expensive homes
    # are harder to predict. Bars should be roughly even across tiers.
    n_tiers = 5
    tier_bounds = np.percentile(y, np.linspace(0, 100, n_tiers + 1))
    tier_idx = np.searchsorted(tier_bounds[1:-1], y)   # 0 … n_tiers-1

    tier_mapes, tier_w10, tier_w20, tier_ns = [], [], [], []
    for t in range(n_tiers):
        t_mask = tier_idx == t
        n_t = int(t_mask.sum())
        if n_t > 0:
            t_mape = abs_pct[t_mask].mean()
            t_w10  = float((abs_pct[t_mask] <= 10).mean() * 100)
            t_w20  = float((abs_pct[t_mask] <= 20).mean() * 100)
        else:
            t_mape = t_w10 = t_w20 = 0.0
        tier_mapes.append(t_mape)
        tier_w10.append(t_w10)
        tier_w20.append(t_w20)
        tier_ns.append(n_t)

    tier_labels = [
        f"Q{t+1}\n${tier_bounds[t]/1e3:.0f}K–\n${tier_bounds[t+1]/1e3:.0f}K"
        for t in range(n_tiers)
    ]
    bar_cols_E = [
        P["green"] if m <= 10 else P["orange"] if m <= 20 else P["red"]
        for m in tier_mapes
    ]
    ax_E.bar(range(n_tiers), tier_mapes, color=bar_cols_E,
             edgecolor="white", width=0.65, alpha=0.85)
    ax_E.axhline(10, color=P["gray"], linestyle="--", linewidth=1,
                 alpha=0.8, label="10 % threshold", zorder=4)

    for t, (tm, tw10, tw20, n) in enumerate(zip(tier_mapes, tier_w10, tier_w20, tier_ns)):
        ax_E.text(t, tm + 0.4,
                  f"±10: {tw10:.0f}%\n±20: {tw20:.0f}%\nn={n}",
                  ha="center", va="bottom", fontsize=7, color=P["dark"])

    ax_E.set_xticks(range(n_tiers))
    ax_E.set_xticklabels(tier_labels, fontsize=7.5)
    ax_E.set_xlabel("Price Quintile (Actual Price)")
    ax_E.set_ylabel("MAPE (%)")
    ax_E.set_title("[E] Accuracy by Price Tier\n(Even bars = no price-range bias)",
                   fontweight="bold")
    ax_E.legend(fontsize=8)
    ax_E.text(0.97, 0.97,
              f"Overall MAPE: {mape:.1f}%\nBar labels: % of listings\nwithin ±10 % / ±20 %",
              transform=ax_E.transAxes, ha="right", va="top", fontsize=8.5,
              color=P["dark"],
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.85))

    # ── [F] Spatial Residuals (lat/lon) ────────────────────────────────────────
    # Plots each listing by its GPS coordinates and colours it by prediction
    # error. Clusters of the same colour reveal geographic bias — location
    # sub-markets the model has not fully captured (e.g. a particular
    # subdivision that commands a premium not explained by the features).
    lat_idx = feature_cols.index("latitude")  if "latitude"  in feature_cols else None
    lon_idx = feature_cols.index("longitude") if "longitude" in feature_cols else None

    if lat_idx is not None and lon_idx is not None:
        lat_F = X[:, lat_idx].astype(float)
        lon_F = X[:, lon_idx].astype(float)
        valid_F = (
            np.isfinite(lat_F) & np.isfinite(lon_F) &
            (lat_F != 0) & (lon_F != 0)
        )
        if valid_F.sum() >= 10:
            res_k = residuals[valid_F] / 1e3
            vmax_F = float(np.percentile(np.abs(res_k), 90))
            sc_F = ax_F.scatter(
                lon_F[valid_F], lat_F[valid_F], c=res_k,
                cmap="RdBu_r", vmin=-vmax_F, vmax=vmax_F,
                alpha=0.65, s=22, edgecolors="none", zorder=3,
            )
            plt.colorbar(sc_F, ax=ax_F, label="Residual ($K)",
                         shrink=0.75, pad=0.02)
            ax_F.set_xlabel("Longitude")
            ax_F.set_ylabel("Latitude")
            ax_F.set_title("[F] Spatial Residuals\n(Clustered blue/red = geographic bias)",
                           fontweight="bold")
            ax_F.text(0.03, 0.97,
                      "Blue: model over-predicts\nRed: model under-predicts\n"
                      "Clusters → location bias\nnot captured by features.",
                      transform=ax_F.transAxes, va="top", fontsize=8.5,
                      color=P["dark"],
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.85))
        else:
            ax_F.text(0.5, 0.5, "Insufficient valid\nlat/lon coordinates",
                      ha="center", va="center", transform=ax_F.transAxes, fontsize=11)
            ax_F.set_title("[F] Spatial Residuals", fontweight="bold")
    else:
        ax_F.text(0.5, 0.5, "latitude/longitude not\nin feature set",
                  ha="center", va="center", transform=ax_F.transAxes, fontsize=11)
        ax_F.set_title("[F] Spatial Residuals", fontweight="bold")

    # ── [G] City Actual vs Predicted median prices ─────────────────────────────
    city_summary = []
    for city in unique_cities:
        mask = cities == city
        if mask.sum() < 3:
            continue
        city_summary.append({
            "city":    city,
            "actual":  np.median(y[mask]) / 1e3,
            "predicted": np.median(y_pred[mask]) / 1e3,
            "n":       mask.sum(),
        })
    cdf = pd.DataFrame(city_summary).sort_values("actual")

    x_pos = np.arange(len(cdf))
    width = 0.38
    b1 = ax_G.barh(x_pos - width/2, cdf["actual"],    width, color=P["blue"],   label="Actual median",    alpha=0.85)
    b2 = ax_G.barh(x_pos + width/2, cdf["predicted"], width, color=P["orange"], label="Predicted median", alpha=0.85)
    ax_G.set_yticks(x_pos)
    ax_G.set_yticklabels([f"{r['city']} (n={r['n']})" for _, r in cdf.iterrows()],
                         fontsize=8)
    ax_G.set_xlabel("Median Price ($K)")
    ax_G.set_title("[G] City Medians: Actual vs Predicted\n(Paired bars — close = low city-level bias)",
                   fontweight="bold")
    ax_G.legend(fontsize=9)

    # ── [H] Residuals by City (box plot) ──────────────────────────────────────
    city_res = {c: residuals[cities == c] / 1e3 for c in unique_cities if (cities == c).sum() >= 3}
    sorted_cities = sorted(city_res, key=lambda c: np.median(city_res[c]))
    data_for_box = [city_res[c] for c in sorted_cities]

    bp = ax_H.boxplot(
        data_for_box,
        vert=False,
        tick_labels=sorted_cities,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(city_color_map.get(sorted_cities[i], P["blue"]))
        patch.set_alpha(0.6)

    ax_H.axvline(0, color="red", linewidth=1.5, linestyle="--", label="No bias")
    ax_H.set_xlabel("Residual ($K)  [Predicted − Actual]")
    ax_H.set_title("[H] Residuals by City\n(Median on 0 = no systematic bias)",
                   fontweight="bold")
    ax_H.tick_params(axis="y", labelsize=8)
    ax_H.legend(fontsize=9)

    # Annotation
    ax_H.text(0.97, 0.03,
              "✓ Good: boxes centred on 0\n✗ Bad: city median far from 0\n   = location bias",
              transform=ax_H.transAxes, ha="right", va="bottom", fontsize=8.5,
              color=P["dark"],
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.85))

    # ── [I] Normal Q-Q Plot ────────────────────────────────────────────────────
    theoretical, sample = _qq_data(residuals)
    ax_I.scatter(theoretical, sample, alpha=0.45, s=18,
                 color=P["purple"], edgecolors="white", linewidths=0.3)

    q25_t, q75_t = np.percentile(theoretical, [25, 75])
    q25_s, q75_s = np.percentile(sample, [25, 75])
    slope = (q75_s - q25_s) / (q75_t - q25_t)
    intercept = q25_s - slope * q25_t
    ref_x = np.array([theoretical.min(), theoretical.max()])
    ax_I.plot(ref_x, slope * ref_x + intercept, "r--", linewidth=2,
              label="Normal reference line")

    ax_I.set_xlabel("Theoretical Quantiles (Normal)")
    ax_I.set_ylabel("Sample Quantiles (Std. Residuals)")
    sw_label = "✓ approximately normal" if sw_p > 0.05 else "✗ non-normal (expected\nwith real estate data)"
    ax_I.set_title(f"[I] Normal Q-Q Plot\nShapiro-Wilk p = {sw_p:.4f}  →  {sw_label}",
                   fontweight="bold")
    ax_I.legend(fontsize=9)
    ax_I.text(0.03, 0.97,
              "✓ Good: points follow red line\n✗ Bad: S-curve = heavy tails\n"
              "Tree models don't require\nnormality — just a guide.",
              transform=ax_I.transAxes, va="top", fontsize=8.5, color=P["dark"],
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.85))

    # ── Style all axes ─────────────────────────────────────────────────────────
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.title.set_fontsize(10.5)

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.005,
        f"{len(y):,} listings  ·  [{data_label}]  ·  "
        f"R² = {r2:.3f}  ·  RMSE = ${rmse/1e3:.0f}K  ·  "
        f"MAE = ${mae/1e3:.0f}K  ·  MAPE = {mape:.1f}%  ·  "
        f"Within ±10%: {within10:.0f}%  ±20%: {within20:.0f}%  ±30%: {within30:.0f}%",
        ha="center", fontsize=9.5, color=P["gray"],
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved → {OUTPUT_FILE}  ({OUTPUT_FILE.stat().st_size / 1024:.0f} KB)")
    return str(OUTPUT_FILE)


if __name__ == "__main__":
    run()
