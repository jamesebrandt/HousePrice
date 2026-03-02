"""
Home Finder — main entry point.

Usage:
  # Run once immediately:
  python main.py --run-now

  # Run on a daily schedule (keeps process alive):
  python main.py --schedule

  # Retrain the model before running:
  python main.py --run-now --retrain

  # Just print to console, skip email:
  python main.py --run-now --no-email

  # Use a custom config file:
  python main.py --run-now --config my_config.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import schedule

from src.filter import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(config_path: str = "config.yaml", retrain: bool = False, send_email: bool = True, use_sample: bool = False, refetch: bool = False, export_report: bool = True):
    """
    Full pipeline:
      1. Fetch fresh listings from Redfin
      2. Clean + engineer features
      3. Load (or retrain) the pricing model
      4. Score all active listings
      5. Filter to your criteria
      6. Email (or print) the top deals
    """
    from src.scraper import fetch_listings, load_all_raw_csv, fetch_listing_descriptions
    from src.features import prepare_dataset
    from src.model import train, load_model, predict
    from src.scorer import compute_value_scores, top_deals, score_summary
    from src.filter import apply_criteria
    from src.notifier import send_email as do_send_email, print_deals_to_console, send_staleness_alert
    from src.reporter import build_html_report
    from src.bps_loader import load_bps_data, compute_permit_features, permit_feature_summary
    from src.zhvi_loader import load_zhvi_data, compute_zhvi_features, zhvi_feature_summary
    from src.acs_loader import load_acs_income, acs_income_summary
    from src.data_staleness import check_all_staleness, stale_sources, staleness_summary
    from src.adu import detect_adu_potential, estimate_adu_rent, compute_adu_affordability, adu_summary

    config = load_config(config_path)
    paths           = config.get("paths", {})
    model_path      = paths.get("model_path", "models/hedonic_model.joblib")
    feature_path    = model_path.replace(".joblib", "_features.joblib")
    holdout_path    = model_path.replace(".joblib", "_holdout.joblib")
    max_train_price = config.get("training", {}).get("max_price")
    scoring_cfg     = config.get("scoring", {})
    min_value_score = scoring_cfg.get("min_value_score", 0.05)
    top_n           = scoring_cfg.get("top_n_alerts", 5)
    adu_cfg         = config.get("adu", {})
    adu_enabled     = adu_cfg.get("enabled", True)
    notif_cfg       = config.get("notifications", {})
    email_to        = notif_cfg.get("email_to")
    email_from      = notif_cfg.get("email_from")

    logger.info("=" * 60)
    logger.info(f"Home Finder run started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)

    search_cfg      = config.get("search", {})
    training_cities = search_cfg.get("training_cities", search_cfg.get("cities", []))
    target_cities   = search_cfg.get("cities", [])
    raw_dir         = paths.get("raw_data_dir", "data/raw")

    # ── Step 1: Fetch listings ────────────────────────────────────────────────
    logger.info("Step 1: Fetching listings...")
    cached_csvs = list(Path(raw_dir).glob("*.csv"))
    if refetch:
        cached_csvs = []

    if use_sample:
        logger.info("Generating sample data (--use-sample flag).")
        from src.generate_sample_data import generate
        raw_df = generate(save_dir=raw_dir, verbose=True)
    elif cached_csvs:
        logger.info(f"Found {len(cached_csvs)} cached CSV(s) in {raw_dir} — loading without live fetch.")
        logger.info("(Pass --refetch to force a fresh API download instead.)")
        raw_df = load_all_raw_csv(raw_dir)
    else:
        logger.info("No cached CSVs found — attempting live fetch from Redfin...")
        raw_df = fetch_listings(cities=training_cities, include_sold=True, save_dir=raw_dir)
        if raw_df.empty:
            logger.warning("Live fetch returned no Utah data. Falling back to sample data.")
            logger.warning("See data/DOWNLOAD_GUIDE.md to download real listings manually.")
            from src.generate_sample_data import generate
            raw_df = generate(save_dir=raw_dir, verbose=True)

    # ── Step 1b–1d: Load supplemental data sources ───────────────────────────
    permit_features: dict = {}
    bps_dir = str(Path(raw_dir) / "BPS Data")
    if Path(bps_dir).exists():
        logger.info("Step 1b: Loading building permit data (BPS)...")
        bps_df = load_bps_data(bps_dir)
        if not bps_df.empty:
            permit_features = compute_permit_features(bps_df, bps_dir=bps_dir)
            logger.info(permit_feature_summary(permit_features))
    else:
        logger.info("No BPS Data folder found — permit features will be skipped.")

    zhvi_features: dict = {}
    zhvi_dir = str(Path(raw_dir) / "Zillow Data")
    if Path(zhvi_dir).exists():
        logger.info("Step 1c: Loading Zillow ZHVI appreciation data...")
        zhvi_df = load_zhvi_data(zhvi_dir, state=search_cfg.get("state", "UT"))
        if not zhvi_df.empty:
            zhvi_features = compute_zhvi_features(zhvi_df)
            logger.info(zhvi_feature_summary(zhvi_features, cities=target_cities))
    else:
        logger.info("No Zillow Data folder found — ZHVI features will be skipped.")

    income_map: dict = {}
    census_dir = str(Path(raw_dir) / "Census Data")
    if Path(census_dir).exists():
        logger.info("Step 1d: Loading Census ACS median household income by ZIP...")
        income_map = load_acs_income(census_dir)
        if income_map:
            logger.info(acs_income_summary(income_map))
    else:
        logger.info("No Census Data folder found — median income feature will be skipped.")

    logger.info("Step 1e: Checking data freshness...")
    staleness_results = check_all_staleness(raw_dir=raw_dir, cfg=config)
    logger.info(staleness_summary(staleness_results))

    stale = stale_sources(staleness_results, exclude_redfin=True)
    if stale and send_email and email_to and email_from:
        send_staleness_alert(stale, email_to=email_to, email_from=email_from)
    elif stale:
        logger.warning(
            f"{len(stale)} supplemental data source(s) are stale and need updating: "
            + ", ".join(r["source"] for r in stale)
        )

    # ── Step 2: Prepare features ──────────────────────────────────────────────
    logger.info("Step 2: Cleaning and engineering features...")
    exclude_cities = search_cfg.get("exclude_cities") or []
    prepared_df = prepare_dataset(
        raw_df,
        permit_features=permit_features,
        zhvi_features=zhvi_features,
        income_map=income_map,
        exclude_cities=exclude_cities if exclude_cities else None,
    )
    # Keyword ADU score requires descriptions (fetched for active listings only);
    # structural score works everywhere, so add it to the full training set here.
    if adu_enabled:
        from src.adu import _structural_score
        prepared_df["adu_structural_score"] = prepared_df.apply(_structural_score, axis=1)

    prepared_df.to_csv(
        Path(paths.get("processed_data_dir", "data/processed")) / "prepared_listings.csv",
        index=False,
    )

    # ── Step 3: Train or load model ───────────────────────────────────────────
    if retrain or not Path(model_path).exists():
        logger.info("Step 3: Training hedonic pricing model...")
        # Utah MLS rules mean sold CSVs often have blank PRICE. Require ≥500 sold rows
        # with confirmed prices before training on sold-only; otherwise fall back to
        # all priced listings (active asking prices are a valid proxy for value).
        MIN_SOLD_ROWS = 500
        training_mode: str
        if "sold" in prepared_df.columns:
            sold_df = prepared_df[prepared_df["sold"] == True]  # noqa: E712
        else:
            sold_df = prepared_df

        if len(sold_df) >= MIN_SOLD_ROWS:
            training_mode = "sold-only"
            logger.info(
                f"Training mode: SOLD-ONLY — {len(sold_df):,} sold listings "
                f"with confirmed sale prices (≥{MIN_SOLD_ROWS} threshold met)."
            )
        else:
            logger.warning(
                f"Training mode: MIXED — only {len(sold_df):,} sold listings "
                f"(need ≥{MIN_SOLD_ROWS}). Falling back to all {len(prepared_df):,} "
                "priced listings, which includes active asking prices. "
                "Model may learn asking-price patterns rather than sale prices. "
                "Add more sold CSV exports to fix this."
            )
            sold_df = prepared_df
            training_mode = "mixed (asking + sold)"

        logger.info(f"Training on {len(sold_df):,} rows  [mode: {training_mode}].")
        model, feature_cols, results = train(
            sold_df,
            model_path=model_path,
            feature_path=feature_path,
            holdout_path=holdout_path,
            max_price=max_train_price,
        )
    else:
        logger.info("Step 3: Loading existing model...")
        model, feature_cols = load_model(model_path, feature_path)

    # ── Step 4: Score active listings ─────────────────────────────────────────
    logger.info("Step 4: Scoring active listings...")
    active_df = prepared_df[
        (prepared_df.get("sold", False) == False) &  # noqa: E712
        (prepared_df["city"].isin(target_cities))
    ] if "sold" in prepared_df.columns else prepared_df[prepared_df["city"].isin(target_cities)]

    if active_df.empty:
        logger.warning("No active listings found in target cities after filtering.")
        return

    predicted = predict(model, feature_cols, active_df, model_path=model_path)
    scored_df = compute_value_scores(active_df, predicted)
    logger.info(score_summary(scored_df))

    # ── Step 4b: ADU detection ────────────────────────────────────────────────
    if adu_enabled:
        logger.info("Step 4b: ADU detection and affordability analysis...")
        url_col = "url" if "url" in scored_df.columns else None
        if url_col:
            max_fetches = adu_cfg.get("max_description_fetches", 200)
            scored_df = fetch_listing_descriptions(scored_df, max_listings=max_fetches, url_col=url_col)
        scored_df = detect_adu_potential(scored_df)
        default_rent = adu_cfg.get("default_adu_rent")
        scored_df = estimate_adu_rent(scored_df, default_rent=default_rent)
        fred_path = str(Path(raw_dir) / "FRED Data" / "MORTGAGE30US.csv")
        scored_df = compute_adu_affordability(scored_df, adu_cfg=adu_cfg, fred_path=fred_path)
        logger.info(adu_summary(scored_df))
    else:
        logger.info("Step 4b: ADU detection disabled in config.")

    # ── Step 5: Filter ────────────────────────────────────────────────────────
    logger.info("Step 5: Applying your criteria...")
    filtered_df = apply_criteria(scored_df, config)
    if filtered_df.empty:
        logger.info("No listings matched your criteria today.")
        return

    # ── Step 6: Notify ────────────────────────────────────────────────────────
    logger.info("Step 6: Selecting top deals and sending alert...")
    deals = top_deals(filtered_df, min_value_score=min_value_score, top_n=top_n)
    print_deals_to_console(deals)

    if send_email and email_to and email_from:
        do_send_email(deals, email_to=email_to, email_from=email_from)
    elif send_email:
        logger.warning("Email skipped — set email_to and email_from in config.yaml")

    if export_report:
        report_name = f"home_finder_{datetime.now().strftime('%Y-%m-%d')}.html"
        report_path = Path("reports") / report_name
        saved = build_html_report(deals, filtered_df, output_path=report_path)
        logger.info(f"HTML report → {saved.resolve()}")

    logger.info("Run complete.")


def main():
    parser = argparse.ArgumentParser(description="Home Finder — find high-value homes in your target area.")
    parser.add_argument("--run-now", action="store_true", help="Run the pipeline immediately.")
    parser.add_argument("--schedule", action="store_true", help="Run on a daily schedule.")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model before running.")
    parser.add_argument("--no-email", action="store_true", help="Print results to console only, skip email.")
    parser.add_argument("--no-report", action="store_true", help="Skip saving the HTML report to reports/.")
    parser.add_argument("--use-sample", action="store_true", help="Use generated sample data instead of live Redfin fetch.")
    parser.add_argument("--refetch", action="store_true", help="Force live Redfin API fetch even if cached CSVs exist.")
    parser.add_argument("--serve", action="store_true", help="Serve the HTML report locally after running.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML (default: config.yaml).")
    parser.add_argument("--time", default="07:00", help="Daily run time in HH:MM format (default: 07:00).")
    args = parser.parse_args()

    send_email_flag = not args.no_email
    use_sample_flag = args.use_sample
    refetch_flag = args.refetch
    export_report_flag = not args.no_report

    if args.run_now:
        run_pipeline(config_path=args.config, retrain=args.retrain,
                     send_email=send_email_flag, use_sample=use_sample_flag,
                     refetch=refetch_flag, export_report=export_report_flag)

    if args.serve:
        import http.server
        import socketserver
        import os

        start_port = 8000
        end_port = 8010
        web_dir = Path("reports")
        
        if not web_dir.exists():
            logger.error(f"Reports directory {web_dir} does not exist. Run with --run-now first.")
            sys.exit(1)

        os.chdir(web_dir)
        Handler = http.server.SimpleHTTPRequestHandler
        socketserver.TCPServer.allow_reuse_address = True
        
        server_started = False
        for port in range(start_port, end_port + 1):
            try:
                with socketserver.TCPServer(("", port), Handler) as httpd:
                    report_name = f"home_finder_{datetime.now().strftime('%Y-%m-%d')}.html"
                    print(f"\nServing reports at http://localhost:{port}")
                    print(f"Latest report: http://localhost:{port}/{report_name}")
                    print("Press Ctrl+C to stop serving.")
                    httpd.serve_forever()
                    server_started = True
                    break
            except OSError as e:
                if e.errno == 48: # Address already in use
                    logger.warning(f"Port {port} is already in use. Trying next port...")
                    continue
                else:
                    raise
            except KeyboardInterrupt:
                print("\nServer stopped.")
                server_started = True
                break

        if not server_started:
            logger.error(f"Could not find an open port between {start_port} and {end_port}. Please free up a port and try again.")

    if args.schedule:
        schedule.every().day.at(args.time).do(
            run_pipeline,
            config_path=args.config,
            retrain=False,
            send_email=send_email_flag,
            use_sample=use_sample_flag,
            refetch=True,   # always pull fresh Redfin listings on scheduled runs
            export_report=export_report_flag,
        )
        logger.info(f"Scheduler started. Will refetch + run daily at {args.time}. Press Ctrl+C to stop.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped.")

    if not args.run_now and not args.schedule and not args.serve:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
