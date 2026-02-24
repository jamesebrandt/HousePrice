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
            {dom} days on mkt
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
        print(f"  Composite Score: {row.get('composite_score', 0):.3f}")
        if row.get("url"):
            print(f"  URL: {row['url']}")
    print("\n" + "=" * 70)
