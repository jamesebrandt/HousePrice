"""
Model Diagnostics — saves a multi-panel PNG to diagnostics/model_diagnostics.png

Layout (3 rows × 3 columns):

  Row 1 — Core Fit
    [A] Predicted vs Actual, colored by city  — does it track reality?
    [B] % Prediction Error distribution       — practical accuracy bands
    [C] Feature Importances (top 15)          — what the model actually learned

  Row 2 — What Drives Price (Partial Dependence)
    [D] Price vs Square Feet                  — most intuitive driver
    [E] Price vs ZIP Median Income            — wealth / neighbourhood signal
    [F] Price vs ZHVI (Zillow Home Value)     — market-level appreciation signal

  Row 3 — Location & Bias
    [G] City medians: Actual vs Predicted     — does it nail each market?
    [H] Residuals by City                     — systematic over/under-prediction
    [I] Normal Q-Q Plot                       — are residuals bell-shaped?

Underlying issues for [D] and [E] (e.g. flat curves, odd shapes) are explained
in diagnostics/DIAGNOSTICS_ISSUES.md.

Run:  python -m src.model_diagnostics
"""

import sys
import logging
from pathlib import Path

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
from sklearn.inspection import partial_dependence

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


def _partial_dep(model, X: np.ndarray, feature_cols: list[str], feat_name: str,
                 grid_resolution: int = 50):
    """Return (grid_values, avg_prediction) for a single feature."""
    if feat_name not in feature_cols:
        return None, None
    idx = feature_cols.index(feat_name)
    try:
        result = partial_dependence(model, X, features=[idx],
                                    kind="average", grid_resolution=grid_resolution)
        return result["grid_values"][0], result["average"][0]
    except Exception:
        return None, None


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    raw_dir: str  = "data/raw",
    model_path:   str = "models/hedonic_model.joblib",
    feature_path: str = "models/hedonic_model_features.joblib",
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

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading prepared data…")
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

    print("Loading model…")
    model, feature_cols = load_model(model_path, feature_path)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].astype(float).values
    y = df["price"].values
    y_pred = model.predict(X)
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

    print(f"R²={r2:.3f}  RMSE=${rmse:,.0f}  MAPE={mape:.1f}%")
    print(f"Within ±10%: {within10:.0f}%  ±20%: {within20:.0f}%  ±30%: {within30:.0f}%")

    city_col = "city" if "city" in df.columns else None
    cities   = df[city_col].values if city_col else np.full(len(df), "unknown")
    unique_cities = sorted(set(cities))
    city_color_map = {c: CITY_COLORS[i % len(CITY_COLORS)]
                      for i, c in enumerate(unique_cities)}

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22), facecolor="white")
    fig.suptitle(
        f"Home Price Model — Diagnostic Report\n"
        f"RandomForest / XGBoost  ·  {len(df):,} listings  ·  "
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
        ("Row 1 — Core Fit",                     0),
        ("Row 2 — What Drives Price",             1),
        ("Row 3 — Location & Bias",               2),
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

    # ── [D] Partial Dependence: Square Feet ────────────────────────────────────
    grid_sqft, pd_sqft = _partial_dep(model, X, feature_cols, "sqft")
    if grid_sqft is not None:
        ax_D.plot(grid_sqft, pd_sqft / 1e3, color=P["blue"], linewidth=2.5)
        ax_D.fill_between(grid_sqft, pd_sqft / 1e3 * 0.9, pd_sqft / 1e3 * 1.1,
                          alpha=0.15, color=P["blue"])
        ax_D.set_xlabel("Square Feet")
        ax_D.set_ylabel("Predicted Price ($K)")
        ax_D.set_title("[D] Price vs Square Feet\n(Partial Dependence — all else equal)",
                       fontweight="bold")
        ax_D.text(0.03, 0.97,
                  "Each point: expected price if\nonly sqft changed, holding\n"
                  "everything else at its median.",
                  transform=ax_D.transAxes, va="top", fontsize=8.5,
                  color=P["dark"],
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.85))
    else:
        ax_D.text(0.5, 0.5, "'sqft' not in feature set",
                  ha="center", va="center", transform=ax_D.transAxes)
        ax_D.set_title("[D] Price vs Sq Ft", fontweight="bold")

    # ── [E] Partial Dependence: ZIP Median Income ──────────────────────────────
    grid_inc, pd_inc = _partial_dep(model, X, feature_cols, "feat_median_income")
    if grid_inc is not None and grid_inc.max() > 0:
        ax_E.plot(grid_inc / 1e3, pd_inc / 1e3, color=P["purple"], linewidth=2.5)
        ax_E.fill_between(grid_inc / 1e3, pd_inc / 1e3 * 0.9, pd_inc / 1e3 * 1.1,
                          alpha=0.15, color=P["purple"])
        ax_E.set_xlabel("ZIP Median Household Income ($K)")
        ax_E.set_ylabel("Predicted Price ($K)")
        ax_E.set_title("[E] Price vs ZIP Median Income\n(Partial Dependence)",
                       fontweight="bold")
        ax_E.text(0.03, 0.97,
                  "Wealthier neighbourhoods → higher\nhome prices, even for the same\n"
                  "physical home features.",
                  transform=ax_E.transAxes, va="top", fontsize=8.5,
                  color=P["dark"],
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.85))
    else:
        ax_E.text(0.5, 0.5, "Income feature not available\nor no variation in data",
                  ha="center", va="center", transform=ax_E.transAxes, fontsize=10)
        ax_E.set_title("[E] Price vs ZIP Income", fontweight="bold")

    # ── [F] Partial Dependence: ZHVI Current ───────────────────────────────────
    grid_zhvi, pd_zhvi = _partial_dep(model, X, feature_cols, "feat_zhvi_current")
    if grid_zhvi is not None and grid_zhvi.max() > 0:
        ax_F.plot(grid_zhvi / 1e3, pd_zhvi / 1e3, color=P["green"], linewidth=2.5)
        ax_F.fill_between(grid_zhvi / 1e3, pd_zhvi / 1e3 * 0.9, pd_zhvi / 1e3 * 1.1,
                          alpha=0.15, color=P["green"])
        ax_F.set_xlabel("Zillow Home Value Index ($K, city median)")
        ax_F.set_ylabel("Predicted Price ($K)")
        ax_F.set_title("[F] Price vs Zillow HVI\n(Partial Dependence)",
                       fontweight="bold")
        ax_F.text(0.03, 0.97,
                  "ZHVI captures the city's overall\nmarket level — a strong anchor\n"
                  "for individual home prices.",
                  transform=ax_F.transAxes, va="top", fontsize=8.5,
                  color=P["dark"],
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.85))
    else:
        ax_F.text(0.5, 0.5, "ZHVI feature not available",
                  ha="center", va="center", transform=ax_F.transAxes, fontsize=10)
        ax_F.set_title("[F] Price vs ZHVI", fontweight="bold")

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
        f"{len(df):,} listings  ·  "
        f"R² = {r2:.3f} (in-sample)  ·  RMSE = ${rmse/1e3:.0f}K  ·  "
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
