"""
Home Finder Dashboard — Interactive Streamlit App

Run: streamlit run dashboard.py

Tabs:
  Overview  — market summary stats and price distributions
  Map       — geographic scatter colored by value score
  Deals     — ranked table of top deals matching your criteria
  Model     — predicted vs actual, feature importance, residuals
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.features import prepare_dataset, get_model_feature_cols
from src.scorer import compute_value_scores, top_deals
from src.filter import apply_criteria

logging.basicConfig(level=logging.WARNING)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Home Finder",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(raw_dir: str = "data/raw") -> pd.DataFrame:
    from src.scraper import load_all_raw_csv
    return load_all_raw_csv(raw_dir)


@st.cache_data(show_spinner=False)
def get_prepared(raw_dir: str = "data/raw") -> pd.DataFrame:
    raw = load_data(raw_dir)
    return prepare_dataset(raw)


@st.cache_resource(show_spinner=False)
def get_model():
    from src.model import load_model
    try:
        return load_model(
            "models/hedonic_model.joblib",
            "models/hedonic_model_features.joblib",
        )
    except FileNotFoundError:
        return None, None


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fmt_price(v) -> str:
    try:
        return f"${float(v):,.0f}"
    except Exception:
        return str(v)


def fmt_pct(v) -> str:
    try:
        return f"{float(v):+.1f}%"
    except Exception:
        return "—"


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏠 Home Finder")
st.sidebar.caption("Saratoga Springs & Eagle Mountain, UT")
st.sidebar.divider()

config = load_config()

df_all = get_prepared()

if df_all.empty:
    st.error(
        "No data found in `data/raw/`. "
        "Run `python -m src.generate_sample_data` for sample data, "
        "or `python main.py --run-now --no-email` to fetch live data."
    )
    st.stop()

# City filter
all_cities = sorted(df_all["city"].dropna().unique().tolist()) if "city" in df_all.columns else []
target_cities = config.get("search", {}).get("cities", all_cities)
selected_cities = st.sidebar.multiselect(
    "Cities", all_cities, default=target_cities
)

# Price range
price_min = int(df_all["price"].min()) if "price" in df_all.columns else 100_000
price_max = int(df_all["price"].max()) if "price" in df_all.columns else 1_000_000
price_range = st.sidebar.slider(
    "Price Range ($)",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, min(price_max, 550_000)),
    step=10_000,
    format="$%d",
)

# Beds / Baths
col1, col2 = st.sidebar.columns(2)
min_beds  = col1.number_input("Min Beds",  min_value=1, max_value=8, value=3)
min_baths = col2.number_input("Min Baths", min_value=1.0, max_value=6.0, value=2.0, step=0.5)

# Sqft
min_sqft = st.sidebar.slider("Min Sqft", 1000, 5000, 1800, step=100)

# Home age
max_age = st.sidebar.slider("Max Home Age (years)", 5, 50, 25)

# Value threshold (only in Deals tab)
st.sidebar.divider()
st.sidebar.markdown("**Scoring**")
min_value_score = st.sidebar.slider("Min % Below Market Price", 0, 20, 5) / 100
top_n = st.sidebar.number_input("Max Deals to Show", 1, 50, 10)

# ── Apply sidebar filters to data ────────────────────────────────────────────
active_df = df_all.copy()
if "sold" in active_df.columns:
    active_df = active_df[active_df["sold"] == False]  # noqa: E712
if selected_cities:
    active_df = active_df[active_df["city"].isin(selected_cities)]

active_df = active_df[
    (active_df["price"] >= price_range[0]) &
    (active_df["price"] <= price_range[1])
]
if "beds" in active_df.columns:
    active_df = active_df[active_df["beds"] >= min_beds]
if "baths" in active_df.columns:
    active_df = active_df[active_df["baths"] >= min_baths]
if "sqft" in active_df.columns:
    active_df = active_df[active_df["sqft"] >= min_sqft]
if "feat_home_age" in active_df.columns:
    active_df = active_df[active_df["feat_home_age"] <= max_age]

# Score listings if model is available
model, feature_cols = get_model()
if model is not None and len(active_df) > 0:
    from src.model import predict
    for col in feature_cols:
        if col not in active_df.columns:
            active_df[col] = 0
    preds = predict(model, feature_cols, active_df)
    active_df = compute_value_scores(active_df, preds)
    has_scores = True
else:
    has_scores = False

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_map, tab_deals, tab_model = st.tabs(
    ["📊 Overview", "🗺️ Map", "🏆 Deals", "🤖 Model"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Market Overview")

    if active_df.empty:
        st.warning("No listings match your current filters.")
    else:
        # KPI cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Listings", f"{len(active_df):,}")
        col2.metric("Median Price", fmt_price(active_df["price"].median()))
        col3.metric("Median $/sqft", f"${(active_df['price'] / active_df['sqft']).median():.0f}" if "sqft" in active_df.columns else "—")
        col4.metric("Median Sqft", f"{active_df['sqft'].median():,.0f}" if "sqft" in active_df.columns else "—")
        col5.metric("Median Days on Mkt", f"{active_df['days_on_market'].median():.0f}" if "days_on_market" in active_df.columns else "—")

        st.divider()

        col_l, col_r = st.columns(2)

        # Price histogram
        with col_l:
            fig = px.histogram(
                active_df,
                x="price",
                nbins=40,
                color="city" if "city" in active_df.columns else None,
                title="Price Distribution",
                labels={"price": "Price ($)", "count": "Listings"},
                barmode="overlay",
                opacity=0.75,
            )
            fig.update_layout(legend_title="City", height=380)
            fig.update_xaxes(tickformat="$,.0f")
            st.plotly_chart(fig, use_container_width=True)

        # Price by city box
        with col_r:
            if "city" in active_df.columns and active_df["city"].nunique() > 1:
                fig = px.box(
                    active_df,
                    x="city",
                    y="price",
                    color="city",
                    title="Price by City",
                    labels={"price": "Price ($)", "city": ""},
                    points="outliers",
                )
                fig.update_layout(showlegend=False, height=380)
                fig.update_yaxes(tickformat="$,.0f")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.histogram(
                    active_df,
                    x="sqft",
                    nbins=35,
                    title="Square Footage Distribution",
                    labels={"sqft": "Sqft"},
                    color_discrete_sequence=["#2ecc71"],
                )
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

        col_l2, col_r2 = st.columns(2)

        # Sqft vs Price scatter
        with col_l2:
            plot_df = active_df.dropna(subset=["sqft", "price"])
            fig = px.scatter(
                plot_df,
                x="sqft",
                y="price",
                color="city" if "city" in plot_df.columns else None,
                hover_data=["address", "beds", "baths", "year_built"] if "address" in plot_df.columns else None,
                trendline="ols",
                title="Sqft vs Price",
                labels={"sqft": "Square Feet", "price": "Price ($)"},
                opacity=0.6,
            )
            fig.update_layout(height=380)
            fig.update_yaxes(tickformat="$,.0f")
            st.plotly_chart(fig, use_container_width=True)

        # $/sqft by city
        with col_r2:
            active_df["price_per_sqft_calc"] = active_df["price"] / active_df["sqft"]
            if "city" in active_df.columns and active_df["city"].nunique() > 1:
                city_stats = (
                    active_df.groupby("city")["price_per_sqft_calc"]
                    .agg(["median", "mean", "count"])
                    .reset_index()
                    .sort_values("median", ascending=False)
                )
                fig = px.bar(
                    city_stats,
                    x="city",
                    y="median",
                    color="city",
                    title="Median $/sqft by City",
                    labels={"median": "Median $/sqft", "city": ""},
                    text="median",
                )
                fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
                fig.update_layout(showlegend=False, height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.histogram(
                    active_df,
                    x="price_per_sqft_calc",
                    nbins=35,
                    title="$/sqft Distribution",
                    labels={"price_per_sqft_calc": "$/sqft"},
                    color_discrete_sequence=["#9b59b6"],
                )
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

        # Days on market
        st.subheader("Days on Market")
        if "days_on_market" in active_df.columns:
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.histogram(
                    active_df,
                    x=active_df["days_on_market"].clip(0, 120),
                    nbins=30,
                    title="Days on Market (capped at 120)",
                    labels={"x": "Days"},
                    color="city" if "city" in active_df.columns else None,
                    barmode="overlay",
                    opacity=0.75,
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                dom_city = active_df.groupby("city")["days_on_market"].median().reset_index()
                fig = px.bar(
                    dom_city,
                    x="city",
                    y="days_on_market",
                    color="city",
                    title="Median Days on Market by City",
                    labels={"days_on_market": "Median Days", "city": ""},
                    text="days_on_market",
                )
                fig.update_traces(texttemplate="%{text:.0f}d", textposition="outside")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.header("Geographic View")

    map_df = active_df.dropna(subset=["latitude", "longitude", "price"])

    if map_df.empty:
        st.warning("No listings with location data to display.")
    else:
        color_by = st.selectbox(
            "Color by",
            options=["pct_below_market", "price", "sqft", "price_per_sqft_calc", "days_on_market"],
            format_func=lambda x: {
                "pct_below_market": "% Below Market (Value Score)",
                "price": "Listing Price",
                "sqft": "Square Feet",
                "price_per_sqft_calc": "$/sqft",
                "days_on_market": "Days on Market",
            }.get(x, x),
        ) if has_scores else "price"

        if color_by not in map_df.columns:
            color_by = "price"

        hover_cols = [c for c in ["address", "city", "price", "beds", "baths", "sqft",
                                   "year_built", "pct_below_market", "composite_score"]
                      if c in map_df.columns]

        fig = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color=color_by,
            color_continuous_scale="RdYlGn" if "below" in color_by or "score" in color_by else "Viridis",
            size="sqft" if "sqft" in map_df.columns else None,
            size_max=18,
            zoom=11,
            center={"lat": map_df["latitude"].mean(), "lon": map_df["longitude"].mean()},
            mapbox_style="open-street-map",
            hover_data=hover_cols,
            title=f"Listings colored by: {color_by.replace('_', ' ').title()}",
            height=600,
        )
        fig.update_layout(coloraxis_colorbar_title=color_by.replace("_", " "))
        st.plotly_chart(fig, use_container_width=True)

        if has_scores:
            st.caption(
                "Green = underpriced relative to model prediction (potential deal). "
                "Red = overpriced. Size = square footage."
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEALS
# ══════════════════════════════════════════════════════════════════════════════
with tab_deals:
    st.header("Top Deals")

    if not has_scores:
        st.warning(
            "No trained model found. Run `python main.py --run-now --no-email` first to train the model, "
            "then refresh this page."
        )
    elif active_df.empty:
        st.warning("No listings match your current filters.")
    else:
        deals = top_deals(active_df, min_value_score=min_value_score, top_n=int(top_n))

        if deals.empty:
            st.info(
                f"No listings are {min_value_score:.0%} or more below predicted price. "
                "Try lowering the 'Min % Below Market Price' slider."
            )
        else:
            # Summary banner
            col1, col2, col3 = st.columns(3)
            col1.metric("Deals Found", len(deals))
            col2.metric("Avg % Below Market", f"{deals['pct_below_market'].mean():.1f}%")
            col3.metric("Avg Savings vs Model", fmt_price(
                (deals["predicted_price"] - deals["price"]).mean()
            ))

            st.divider()

            # Ranked deal cards
            for rank, (_, row) in enumerate(deals.iterrows(), 1):
                pct = row.get("pct_below_market", 0)
                score = row.get("composite_score", 0)
                color = "🟢" if pct >= 10 else ("🟡" if pct >= 5 else "⚪")

                with st.expander(
                    f"{color} #{rank}  ·  {row.get('address', '?')} ({row.get('city', '')})  ·  "
                    f"{fmt_price(row.get('price'))}  ·  {pct:+.1f}% vs market",
                    expanded=(rank <= 3),
                ):
                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        st.markdown("**Pricing**")
                        st.metric("List Price", fmt_price(row.get("price")),
                                  delta=f"{pct:+.1f}% vs model")
                        st.metric("Model Predicts", fmt_price(row.get("predicted_price")))
                        st.metric("Savings", fmt_price(
                            row.get("predicted_price", 0) - row.get("price", 0)
                        ))

                    with col_b:
                        st.markdown("**Details**")
                        st.write(f"🛏  {row.get('beds', '?')} beds / {row.get('baths', '?')} baths")
                        st.write(f"📐 {row.get('sqft', '?'):,.0f} sqft" if pd.notna(row.get("sqft")) else "📐 sqft: ?")
                        st.write(f"🌿 Lot: {row.get('lot_sqft', '?'):,.0f} sqft" if pd.notna(row.get("lot_sqft")) else "🌿 Lot: ?")
                        st.write(f"🏗️ Built: {int(row['year_built'])}" if pd.notna(row.get("year_built")) else "🏗️ Built: ?")
                        st.write(f"📅 {int(row.get('days_on_market', 0))} days on market")
                        if row.get("hoa_monthly", 0) > 0:
                            st.write(f"🏢 HOA: ${row['hoa_monthly']:,.0f}/mo")

                    with col_c:
                        st.markdown("**Score Breakdown**")
                        st.metric("Composite Score", f"{score:.3f}")
                        if "sqft_per_dollar" in row:
                            st.write(f"$/sqft: ${row['price'] / row['sqft']:,.0f}" if pd.notna(row.get("sqft")) else "")
                        if row.get("url"):
                            st.link_button("View on Redfin →", row["url"])

            st.divider()
            st.subheader("Full Deals Table")

            display_cols = [c for c in ["address", "city", "price", "predicted_price",
                                         "pct_below_market", "beds", "baths", "sqft",
                                         "year_built", "days_on_market", "composite_score"]
                            if c in deals.columns]
            st.dataframe(
                deals[display_cols].style.format({
                    "price": "${:,.0f}",
                    "predicted_price": "${:,.0f}",
                    "pct_below_market": "{:+.1f}%",
                    "sqft": "{:,.0f}",
                    "composite_score": "{:.3f}",
                    "year_built": "{:.0f}",
                    "days_on_market": "{:.0f}",
                }).background_gradient(subset=["composite_score"], cmap="Greens").background_gradient(
                    subset=["pct_below_market"], cmap="RdYlGn"
                ),
                use_container_width=True,
                height=400,
            )

            # Value score distribution
            st.subheader("Value Score Distribution — All Filtered Listings")
            fig = px.histogram(
                active_df,
                x="pct_below_market",
                nbins=50,
                color="city" if "city" in active_df.columns else None,
                barmode="overlay",
                opacity=0.75,
                title="% Below Predicted Market Price",
                labels={"pct_below_market": "% Below Predicted Price"},
            )
            fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Fairly priced")
            fig.add_vline(x=min_value_score * 100, line_dash="dot", line_color="green",
                          annotation_text=f"Alert threshold ({min_value_score:.0%})")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            # Composite score vs price scatter — highlight top deals
            fig = px.scatter(
                active_df.dropna(subset=["price", "composite_score"]),
                x="price",
                y="composite_score",
                color="city" if "city" in active_df.columns else None,
                hover_data=["address", "pct_below_market"] if "address" in active_df.columns else None,
                opacity=0.5,
                title="Composite Score vs Price — Stars = Your Top Deals",
                labels={"price": "Listing Price ($)", "composite_score": "Composite Score"},
                height=400,
            )
            fig.add_trace(go.Scatter(
                x=deals["price"],
                y=deals["composite_score"],
                mode="markers+text",
                marker=dict(symbol="star", size=16, color="gold",
                            line=dict(color="black", width=1)),
                text=[str(i + 1) for i in range(len(deals))],
                textposition="top center",
                name="Top Deals",
            ))
            fig.update_xaxes(tickformat="$,.0f")
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.header("Model Diagnostics")

    if model is None:
        st.warning(
            "No trained model found in `models/`. "
            "Run `python main.py --run-now --no-email` to train the model first."
        )
    elif not has_scores:
        st.info("Model loaded but no active listings scored yet.")
    else:
        col1, col2 = st.columns(2)

        # Predicted vs Actual
        with col1:
            plot_df = active_df.dropna(subset=["price", "predicted_price"])
            fig = px.scatter(
                plot_df,
                x="predicted_price",
                y="price",
                color="pct_below_market" if has_scores else None,
                color_continuous_scale="RdYlGn",
                hover_data=["address"] if "address" in plot_df.columns else None,
                opacity=0.6,
                title="Predicted vs Listed Price",
                labels={"predicted_price": "Model Prediction ($)", "price": "List Price ($)"},
                height=420,
            )
            lim_max = max(plot_df["predicted_price"].max(), plot_df["price"].max()) * 1.05
            lim_min = min(plot_df["predicted_price"].min(), plot_df["price"].min()) * 0.95
            fig.add_trace(go.Scatter(
                x=[lim_min, lim_max], y=[lim_min, lim_max],
                mode="lines", line=dict(dash="dash", color="gray", width=1.5),
                name="Fairly priced",
            ))
            fig.update_xaxes(tickformat="$,.0f")
            fig.update_yaxes(tickformat="$,.0f")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Points below the dashed line are listed below predicted price (potential deals). Above = overpriced.")

        # Residual distribution
        with col2:
            residuals = active_df["predicted_price"] - active_df["price"]
            fig = px.histogram(
                x=residuals / 1000,
                nbins=40,
                title="Prediction Error Distribution",
                labels={"x": "Predicted − Listed ($K)"},
                color_discrete_sequence=["#3498db"],
                height=420,
            )
            fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Fairly priced")
            fig.add_vline(x=residuals.mean() / 1000, line_dash="dot", line_color="red",
                          annotation_text=f"Mean: ${residuals.mean()/1000:.1f}K")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Positive = model thinks home is worth more than list price.  "
                f"Median error: ${residuals.median()/1000:.1f}K"
            )

        # Feature importance via permutation (if accessible from model)
        st.subheader("Feature Importance")
        try:
            last_step = model[-1] if hasattr(model, "__getitem__") else model
            if hasattr(last_step, "feature_importances_"):
                imp = last_step.feature_importances_
                imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp})
                imp_df = imp_df.sort_values("importance", ascending=True).tail(20)
                fig = px.bar(
                    imp_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 20 Feature Importances",
                    labels={"importance": "Importance", "feature": "Feature"},
                    color="importance",
                    color_continuous_scale="Blues",
                    height=500,
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Importance = how much each feature reduces prediction error. "
                    "Higher = more influential."
                )
            elif hasattr(last_step, "coef_"):
                coefs = pd.DataFrame({
                    "feature": feature_cols,
                    "coefficient": last_step.coef_,
                }).sort_values("coefficient", ascending=True)
                fig = px.bar(
                    coefs,
                    x="coefficient",
                    y="feature",
                    orientation="h",
                    title="Linear Model Coefficients ($ change per unit)",
                    color="coefficient",
                    color_continuous_scale="RdBu",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Feature importance not available for this model type. ({e})")

        # Score distribution stats
        st.subheader("Value Score Statistics")
        col_a, col_b, col_c, col_d = st.columns(4)
        vs = active_df["value_score"] if "value_score" in active_df.columns else pd.Series(dtype=float)
        col_a.metric("Median Value Score", f"{vs.median():.2%}" if not vs.empty else "—")
        col_b.metric("% Listings Below Market", f"{(vs > 0).mean():.0%}" if not vs.empty else "—")
        col_c.metric(">5% Below Market", f"{(vs >= 0.05).sum()}" if not vs.empty else "—")
        col_d.metric(">10% Below Market", f"{(vs >= 0.10).sum()}" if not vs.empty else "—")
