"""
Email notifier module.

Sends an HTML email summarizing the top high-value homes found.
Uses Gmail via yagmail (which wraps smtplib with OAuth/App Password support).

Setup:
  1. Enable 2FA on your Google account
  2. Create an App Password: https://myaccount.google.com/apppasswords
  3. Set GMAIL_APP_PASSWORD in your .env file
"""

import os
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

try:
    import yagmail
    YAGMAIL_AVAILABLE = True
except ImportError:
    YAGMAIL_AVAILABLE = False
    logger.warning("yagmail not installed. Run: pip install yagmail")


def _format_price(val) -> str:
    try:
        return f"${float(val):,.0f}"
    except (ValueError, TypeError):
        return str(val)


def _format_pct(val) -> str:
    try:
        return f"{float(val):.1f}%"
    except (ValueError, TypeError):
        return "—"


def build_html_email(df: pd.DataFrame, run_date: str = None) -> str:
    """
    Build an HTML email body from a DataFrame of top-scored listings.
    """
    run_date = run_date or datetime.now().strftime("%B %d, %Y")

    rows_html = ""
    for i, row in df.iterrows():
        url = row.get("url", "#") or "#"
        address = row.get("address", "Unknown address")
        city = row.get("city", "")
        price = _format_price(row.get("price"))
        predicted = _format_price(row.get("predicted_price"))
        pct_below = _format_pct(row.get("pct_below_market"))
        beds = row.get("beds", "?")
        baths = row.get("baths", "?")
        sqft = f"{int(row['sqft']):,}" if pd.notna(row.get("sqft")) else "?"
        year = int(row["year_built"]) if pd.notna(row.get("year_built")) else "?"
        dom = int(row["days_on_market"]) if pd.notna(row.get("days_on_market")) else "?"
        score = f"{row.get('composite_score', 0):.3f}"

        badge_color = "#27ae60" if row.get("value_score", 0) > 0 else "#e74c3c"

        adu_html = ""
        if row.get("adu_likely", False):
            adu_rent = row.get("estimated_adu_rent", 0)
            adu_html = (
                '<br><span style="background:#8e44ad;color:white;padding:2px 8px;'
                'border-radius:10px;font-size:11px;font-weight:bold;">'
                f'ADU ~${adu_rent:,.0f}/mo</span>'
            )

        rows_html += f"""
        <tr style="border-bottom: 1px solid #eee;">
          <td style="padding: 14px 10px;">
            <strong><a href="{url}" style="color:#2c3e50;text-decoration:none;">{address}</a></strong><br>
            <span style="color:#7f8c8d;font-size:13px;">{city}, UT</span>
          </td>
          <td style="padding: 14px 10px; text-align:right;">
            <strong style="font-size:16px;">{price}</strong><br>
            <span style="color:#7f8c8d;font-size:12px;">Model: {predicted}</span>
          </td>
          <td style="padding: 14px 10px; text-align:center;">
            <span style="background:{badge_color};color:white;padding:4px 10px;border-radius:12px;font-size:13px;font-weight:bold;">
              {pct_below} below
            </span>
          </td>
          <td style="padding: 14px 10px; text-align:center; color:#555; font-size:13px;">
            {beds}bd / {baths}ba<br>
            {sqft} sqft<br>
            Built {year}<br>
            {dom} days on mkt{adu_html}
          </td>
          <td style="padding: 14px 10px; text-align:center; font-size:13px; color:#555;">
            {score}
          </td>
        </tr>
        """

    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:900px;margin:auto;color:#2c3e50;">
      <div style="background:#2c3e50;padding:24px;border-radius:8px 8px 0 0;">
        <h1 style="color:white;margin:0;font-size:22px;">Home Finder Alert</h1>
        <p style="color:#bdc3c7;margin:4px 0 0;">{run_date} — Top value listings in Saratoga Springs & Eagle Mountain</p>
      </div>
      <div style="background:#f8f9fa;padding:16px;border-left:4px solid #3498db;margin:16px 0;border-radius:4px;">
        <strong>How to read this:</strong> "% below" means the home is priced that much below what the model
        predicts it <em>should</em> cost based on comparable homes. Higher composite score = better overall value.
      </div>
      <table style="width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
        <thead>
          <tr style="background:#34495e;color:white;text-align:left;">
            <th style="padding:12px 10px;">Address</th>
            <th style="padding:12px 10px;text-align:right;">Price</th>
            <th style="padding:12px 10px;text-align:center;">Value</th>
            <th style="padding:12px 10px;text-align:center;">Details</th>
            <th style="padding:12px 10px;text-align:center;">Score</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
      <p style="color:#95a5a6;font-size:12px;margin-top:20px;text-align:center;">
        Powered by your Home Finder model · Data from Redfin<br>
        This is not financial advice. Always do your own due diligence.
      </p>
    </body></html>
    """
    return html


def send_email(
    df: pd.DataFrame,
    email_to: str,
    email_from: str,
    subject: str = None,
    app_password: str = None,
) -> bool:
    """
    Send the top-deals email.

    Args:
        df:           DataFrame of top listings (output of top_deals()).
        email_to:     Recipient email address.
        email_from:   Sender Gmail address.
        subject:      Email subject line (auto-generated if None).
        app_password: Gmail App Password. Reads GMAIL_APP_PASSWORD env var if None.

    Returns:
        True on success, False on failure.
    """
    if not YAGMAIL_AVAILABLE:
        logger.error("yagmail not installed. Run: pip install yagmail")
        return False

    app_password = app_password or os.getenv("GMAIL_APP_PASSWORD")
    if not app_password:
        logger.error(
            "No Gmail App Password found. Set GMAIL_APP_PASSWORD in your .env file.\n"
            "Generate one at: https://myaccount.google.com/apppasswords"
        )
        return False

    subject = subject or (
        f"Home Finder: {len(df)} high-value listing{'s' if len(df) != 1 else ''} "
        f"— {datetime.now().strftime('%b %d')}"
    )

    html_body = build_html_email(df)

    try:
        yag = yagmail.SMTP(email_from, app_password)
        yag.send(to=email_to, subject=subject, contents=html_body)
        logger.info(f"Email sent to {email_to}: '{subject}'")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def build_staleness_email(stale_results: list) -> str:
    """
    Build an HTML email body listing data sources that need to be updated.

    Args:
        stale_results: List of dicts from data_staleness.check_all_staleness()
                       filtered to only stale entries.
    """
    rows_html = ""
    for r in stale_results:
        source   = r.get("source", "Unknown")
        latest   = r.get("latest") or "—"
        age_days = r.get("age_days")
        age_str  = f"{age_days:,} days" if age_days is not None else "unknown"
        message  = r.get("message", "")
        dl_url   = r.get("download_url")

        dl_btn = (
            f'<a href="{dl_url}" style="background:#3498db;color:white;padding:5px 12px;'
            'border-radius:4px;text-decoration:none;font-size:12px;">Download</a>'
            if dl_url else "—"
        )

        rows_html += f"""
        <tr style="border-bottom:1px solid #eee;">
          <td style="padding:12px 10px;font-weight:bold;">{source}</td>
          <td style="padding:12px 10px;text-align:center;">{latest}</td>
          <td style="padding:12px 10px;text-align:center;">{age_str}</td>
          <td style="padding:12px 10px;font-size:13px;color:#555;">{message}</td>
          <td style="padding:12px 10px;text-align:center;">{dl_btn}</td>
        </tr>
        """

    run_date = datetime.now().strftime("%B %d, %Y")
    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:860px;margin:auto;color:#2c3e50;">
      <div style="background:#e74c3c;padding:24px;border-radius:8px 8px 0 0;">
        <h1 style="color:white;margin:0;font-size:22px;">Home Finder — Data Update Required</h1>
        <p style="color:#fadbd8;margin:4px 0 0;">{run_date} — {len(stale_results)} data source(s) need attention</p>
      </div>
      <div style="background:#fef9e7;padding:16px;border-left:4px solid #f39c12;margin:16px 0;border-radius:4px;">
        <strong>Action needed:</strong> One or more supplemental data sources used by the model
        are out of date. Download the latest files and place them in the correct subfolder
        inside <code>data/raw/</code>, then run <code>python main.py --run-now --retrain</code>
        to rebuild the model with fresh data.
      </div>
      <table style="width:100%;border-collapse:collapse;background:white;border-radius:8px;
                    overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
        <thead>
          <tr style="background:#34495e;color:white;text-align:left;">
            <th style="padding:12px 10px;">Source</th>
            <th style="padding:12px 10px;text-align:center;">Latest Data</th>
            <th style="padding:12px 10px;text-align:center;">Age</th>
            <th style="padding:12px 10px;">Status</th>
            <th style="padding:12px 10px;text-align:center;">Update</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
      <div style="background:#f8f9fa;padding:14px;border-radius:4px;margin-top:16px;font-size:13px;">
        <strong>Where to put the files:</strong><br>
        &bull; <strong>BPS Permits</strong> → <code>data/raw/BPS Data/co&lt;YY&gt;&lt;MM&gt;c.txt</code>
          (e.g. <code>co2601c.txt</code> for Jan 2026)<br>
        &bull; <strong>Zillow ZHVI</strong> → <code>data/raw/Zillow Data/City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv</code><br>
        &bull; <strong>Census ACS Income</strong> → <code>data/raw/Census Data/acs_median_income_by_zip_&lt;YEAR&gt;.csv</code>
      </div>
      <p style="color:#95a5a6;font-size:12px;margin-top:20px;text-align:center;">
        Powered by your Home Finder model · This is an automated maintenance alert
      </p>
    </body></html>
    """
    return html


def send_staleness_alert(
    stale_results: list,
    email_to: str,
    email_from: str,
    app_password: str = None,
) -> bool:
    """
    Send an email listing data sources that need to be updated.

    Args:
        stale_results: Filtered list of stale result dicts from data_staleness.
        email_to:      Recipient address.
        email_from:    Sender Gmail address.
        app_password:  Gmail App Password (falls back to GMAIL_APP_PASSWORD env var).

    Returns:
        True on success, False on failure.
    """
    if not stale_results:
        return True  # nothing to send

    if not YAGMAIL_AVAILABLE:
        logger.error("yagmail not installed. Run: pip install yagmail")
        return False

    app_password = app_password or os.getenv("GMAIL_APP_PASSWORD")
    if not app_password:
        logger.error(
            "No Gmail App Password found. Set GMAIL_APP_PASSWORD in your .env file."
        )
        return False

    subject = (
        f"Home Finder: {len(stale_results)} data source(s) need updating "
        f"— {datetime.now().strftime('%b %d')}"
    )
    html_body = build_staleness_email(stale_results)

    try:
        yag = yagmail.SMTP(email_from, app_password)
        yag.send(to=email_to, subject=subject, contents=html_body)
        logger.info(f"Staleness alert sent to {email_to}: '{subject}'")
        return True
    except Exception as e:
        logger.error(f"Failed to send staleness alert: {e}")
        return False


def print_deals_to_console(df: pd.DataFrame) -> None:
    """Pretty-print top deals to stdout — useful when testing without email."""
    print("\n" + "=" * 70)
    print(f"  TOP {len(df)} HIGH-VALUE LISTINGS")
    print("=" * 70)

    for _, row in df.iterrows():
        price = _format_price(row.get("price"))
        predicted = _format_price(row.get("predicted_price"))
        pct_below = _format_pct(row.get("pct_below_market"))
        print(f"\n  {row.get('address', 'Unknown')} ({row.get('city', '')})")
        print(f"  Price:     {price}  |  Model predicted: {predicted}  |  {pct_below} below market")
        print(f"  Beds/Baths: {row.get('beds')}bd / {row.get('baths')}ba  |  Sqft: {row.get('sqft', '?')}")
        if row.get("adu_likely", False):
            adu_rent = row.get("estimated_adu_rent", 0)
            mortgage = row.get("estimated_mortgage", 0)
            net = row.get("net_monthly_cost", 0)
            conf = row.get("adu_confidence", 0) * 100
            print(f"  ADU:       ~${adu_rent:,.0f}/mo rent  |  Mortgage: ${mortgage:,.0f}/mo  |  Net: ${net:,.0f}/mo  ({conf:.0f}% confidence)")
        print(f"  Composite Score: {row.get('composite_score', 0):.3f}")
        if row.get("url"):
            print(f"  URL: {row['url']}")
    print("\n" + "=" * 70)
