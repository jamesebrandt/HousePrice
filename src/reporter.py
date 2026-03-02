"""
HTML Report Generator.

Builds a self-contained HTML report from scored listings.
Open the output file in any browser — no server required.

Usage:
    from src.reporter import build_html_report
    path = build_html_report(deals_df, filtered_df, output_path="reports/home_finder_2026-03-01.html")
"""

import base64
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DIAGNOSTICS_PNG = Path(__file__).parent.parent / "diagnostics" / "model_diagnostics.png"


def _fmt_price(val) -> str:
    try:
        return f"${float(val):,.0f}"
    except (ValueError, TypeError):
        return "—"


def _fmt_pct(val) -> str:
    try:
        return f"{float(val):.1f}%"
    except (ValueError, TypeError):
        return "—"


def _fmt_int(val) -> str:
    try:
        return f"{int(float(val)):,}"
    except (ValueError, TypeError):
        return "—"


def _safe(val, default="—") -> str:
    if pd.isna(val) if not isinstance(val, str) else val == "":
        return default
    return str(val)


def _encode_png(png_path: Path) -> str | None:
    try:
        data = png_path.read_bytes()
        return base64.b64encode(data).decode("ascii")
    except Exception:
        return None


def _deal_cards(deals: pd.DataFrame) -> str:
    if deals.empty:
        return "<p style='color:#7f8c8d;'>No deals found matching your criteria.</p>"

    cards = []
    for rank, (_, row) in enumerate(deals.iterrows(), 1):
        url = row.get("url", "#") or "#"
        address = _safe(row.get("address"), "Unknown address")
        city = _safe(row.get("city"), "")
        price = _fmt_price(row.get("price"))
        predicted = _fmt_price(row.get("predicted_price"))
        pct_below = row.get("pct_below_market", 0)
        pct_below_str = _fmt_pct(pct_below)
        beds = _safe(row.get("beds"))
        baths = _safe(row.get("baths"))
        sqft = _fmt_int(row.get("sqft")) if pd.notna(row.get("sqft")) else "—"
        year = _fmt_int(row.get("year_built")) if pd.notna(row.get("year_built")) else "—"
        dom = _fmt_int(row.get("days_on_market")) if pd.notna(row.get("days_on_market")) else "—"
        score = f"{row.get('composite_score', 0):.3f}"

        try:
            pct_val = float(pct_below)
        except (ValueError, TypeError):
            pct_val = 0
        badge_bg = "#27ae60" if pct_val >= 5 else ("#f39c12" if pct_val > 0 else "#e74c3c")
        border_color = "#27ae60" if pct_val >= 5 else ("#f39c12" if pct_val > 0 else "#e74c3c")

        # ADU info
        adu_likely = row.get("adu_likely", False)
        adu_conf = row.get("adu_confidence", 0)
        adu_rent = row.get("estimated_adu_rent", 0)
        mortgage = row.get("estimated_mortgage", 0)
        net_cost = row.get("net_monthly_cost", 0)

        adu_html = ""
        if adu_likely:
            conf_pct = f"{adu_conf * 100:.0f}%" if adu_conf else "—"
            adu_html = f"""
              <div class="adu-badge">
                <span class="adu-tag">ADU Potential</span>
                <span class="adu-detail">~${adu_rent:,.0f}/mo rent · ${mortgage:,.0f}/mo mortgage · <strong>${net_cost:,.0f}/mo net</strong> · {conf_pct} confidence</span>
              </div>"""

        link_btn = (
            f'<a href="{url}" target="_blank" class="redfin-btn">View on Redfin ↗</a>'
            if url != "#" else ""
        )

        cards.append(f"""
        <div class="deal-card" style="border-left: 4px solid {border_color};">
          <div class="deal-rank">#{rank}</div>
          <div class="deal-body">
            <div class="deal-address">
              <strong>{address}</strong>
              <span class="deal-city">{city}, UT</span>
            </div>
            <div class="deal-meta">
              <div class="deal-price">
                <span class="price-main">{price}</span>
                <span class="price-sub">Model: {predicted}</span>
              </div>
              <div class="deal-badge" style="background:{badge_bg};">{pct_below_str} below market</div>
              <div class="deal-details">
                {beds}bd / {baths}ba &nbsp;·&nbsp; {sqft} sqft &nbsp;·&nbsp; Built {year} &nbsp;·&nbsp; {dom} days on mkt
              </div>{adu_html}
              <div class="deal-score">Composite score: <strong>{score}</strong></div>
            </div>
          </div>
          <div class="deal-actions">{link_btn}</div>
        </div>""")

    return "\n".join(cards)


def _listings_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p style='color:#7f8c8d;'>No listings to display.</p>"

    rows = []
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        url = row.get("url", "#") or "#"
        address = _safe(row.get("address"), "Unknown")
        city = _safe(row.get("city"), "")
        price = _fmt_price(row.get("price"))
        predicted = _fmt_price(row.get("predicted_price"))

        pct_val = row.get("pct_below_market", 0)
        try:
            pct_float = float(pct_val)
        except (ValueError, TypeError):
            pct_float = 0.0
        pct_str = _fmt_pct(pct_float)

        # Cell color: green for deals, neutral for fairly priced, red for overpriced
        if pct_float >= 10:
            pct_bg = "rgba(39,174,96,0.20)"
        elif pct_float >= 5:
            pct_bg = "rgba(39,174,96,0.10)"
        elif pct_float > 0:
            pct_bg = "rgba(243,156,18,0.12)"
        else:
            pct_bg = "rgba(231,76,60,0.12)"

        beds = _safe(row.get("beds"))
        baths = _safe(row.get("baths"))
        sqft_val = _fmt_int(row.get("sqft")) if pd.notna(row.get("sqft")) else "—"

        price_num = row.get("price", None)
        sqft_num = row.get("sqft", None)
        try:
            price_per_sqft = _fmt_price(float(price_num) / float(sqft_num)) if (
                price_num and sqft_num and float(sqft_num) > 0
            ) else "—"
        except (ValueError, TypeError):
            price_per_sqft = "—"

        year = _fmt_int(row.get("year_built")) if pd.notna(row.get("year_built")) else "—"
        dom = _fmt_int(row.get("days_on_market")) if pd.notna(row.get("days_on_market")) else "—"
        score = f"{row.get('composite_score', 0):.3f}"

        # ADU column
        adu_cell = ""
        if row.get("adu_likely", False):
            rent_est = row.get("estimated_adu_rent", 0)
            net_est = row.get("net_monthly_cost", 0)
            adu_cell = f'<span class="adu-table-tag">${rent_est:,.0f}/mo → ${net_est:,.0f} net</span>'
        else:
            adu_cell = "—"

        link_cell = (
            f'<a href="{url}" target="_blank" class="table-link">↗</a>'
            if url != "#" else "—"
        )

        rows.append(f"""
        <tr>
          <td class="rank-cell">{rank}</td>
          <td class="address-cell">
            <span class="addr-main">{address}</span>
          </td>
          <td>{city}</td>
          <td class="num-cell">{price}</td>
          <td class="num-cell">{predicted}</td>
          <td class="num-cell pct-cell" style="background:{pct_bg};">{pct_str}</td>
          <td class="num-cell">{beds}</td>
          <td class="num-cell">{baths}</td>
          <td class="num-cell">{sqft_val}</td>
          <td class="num-cell">{price_per_sqft}</td>
          <td class="num-cell">{year}</td>
          <td class="num-cell">{dom}</td>
          <td class="num-cell">{adu_cell}</td>
          <td class="num-cell score-cell">{score}</td>
          <td class="link-cell">{link_cell}</td>
        </tr>""")

    return "\n".join(rows)


def build_html_report(
    deals_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    output_path: str | Path = "reports/home_finder.html",
    diagnostics_png_path: str | Path | None = None,
    run_date: str | None = None,
) -> Path:
    """
    Build and save a self-contained HTML report.

    Args:
        deals_df:            Top-N deals (output of top_deals()).
        filtered_df:         All listings passing criteria filters (output of apply_criteria()).
        output_path:         Where to save the .html file.
        diagnostics_png_path: Path to model_diagnostics.png; uses default location if None.
        run_date:            Display date string; defaults to today.

    Returns:
        Path to the saved HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_date = run_date or datetime.now().strftime("%B %d, %Y at %I:%M %p")

    png_path = Path(diagnostics_png_path) if diagnostics_png_path else _DIAGNOSTICS_PNG
    png_b64 = _encode_png(png_path)
    diag_section = (
        f'<img src="data:image/png;base64,{png_b64}" alt="Model diagnostics" class="diag-img">'
        if png_b64
        else '<p class="no-diag">Diagnostics image not found. Run: <code>python -m src.model_diagnostics</code></p>'
    )

    n_scored = len(filtered_df)
    n_deals = len(deals_df)
    med_price = _fmt_price(filtered_df["price"].median()) if not filtered_df.empty and "price" in filtered_df.columns else "—"
    med_pct = _fmt_pct(filtered_df["pct_below_market"].median()) if not filtered_df.empty and "pct_below_market" in filtered_df.columns else "—"
    cities_list = ", ".join(sorted(filtered_df["city"].dropna().unique())) if not filtered_df.empty and "city" in filtered_df.columns else "—"

    n_adu = int(filtered_df["adu_likely"].sum()) if not filtered_df.empty and "adu_likely" in filtered_df.columns else 0

    deal_cards_html = _deal_cards(deals_df)
    table_rows_html = _listings_table(filtered_df)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Home Finder Report — {run_date}</title>
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f0f2f5;
      color: #2c3e50;
      margin: 0;
      padding: 0;
    }}

    /* ── Header ── */
    .header {{
      background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
      padding: 28px 40px;
      color: white;
    }}
    .header h1 {{ margin: 0 0 4px; font-size: 26px; font-weight: 700; }}
    .header p {{ margin: 0; color: #bdc3c7; font-size: 14px; }}

    /* ── KPI bar ── */
    .kpi-bar {{
      display: flex;
      gap: 16px;
      padding: 20px 40px;
      background: white;
      border-bottom: 1px solid #e0e4ea;
      flex-wrap: wrap;
    }}
    .kpi {{
      flex: 1;
      min-width: 120px;
      text-align: center;
      padding: 12px 16px;
      background: #f8f9fa;
      border-radius: 8px;
      border: 1px solid #e9ecef;
    }}
    .kpi-value {{ font-size: 22px; font-weight: 700; color: #2c3e50; }}
    .kpi-label {{ font-size: 12px; color: #7f8c8d; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }}

    /* ── Tabs ── */
    .tabs {{
      display: flex;
      gap: 0;
      padding: 0 40px;
      background: white;
      border-bottom: 2px solid #e0e4ea;
    }}
    .tab-btn {{
      padding: 14px 24px;
      border: none;
      background: none;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
      color: #7f8c8d;
      border-bottom: 3px solid transparent;
      margin-bottom: -2px;
      transition: color 0.15s, border-color 0.15s;
    }}
    .tab-btn.active {{ color: #2980b9; border-bottom-color: #2980b9; }}
    .tab-btn:hover:not(.active) {{ color: #2c3e50; }}

    /* ── Content ── */
    .content {{ max-width: 1400px; margin: 0 auto; padding: 32px 40px; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}

    /* ── Section headings ── */
    .section-title {{
      font-size: 18px;
      font-weight: 700;
      margin: 0 0 20px;
      padding-bottom: 10px;
      border-bottom: 2px solid #e9ecef;
      color: #2c3e50;
    }}

    /* ── Deal cards ── */
    .deal-card {{
      display: flex;
      align-items: flex-start;
      gap: 16px;
      background: white;
      border-radius: 10px;
      padding: 20px 24px;
      margin-bottom: 14px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.07);
      position: relative;
    }}
    .deal-rank {{
      font-size: 22px;
      font-weight: 800;
      color: #bdc3c7;
      min-width: 36px;
      padding-top: 2px;
    }}
    .deal-body {{ flex: 1; }}
    .deal-address {{ margin-bottom: 10px; }}
    .deal-address strong {{ font-size: 16px; }}
    .deal-city {{ color: #7f8c8d; font-size: 13px; margin-left: 8px; }}
    .deal-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
    }}
    .deal-price {{ line-height: 1.3; }}
    .price-main {{ font-size: 20px; font-weight: 700; display: block; }}
    .price-sub {{ font-size: 12px; color: #7f8c8d; }}
    .deal-badge {{
      color: white;
      padding: 5px 14px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .deal-details {{ font-size: 13px; color: #555; }}
    .deal-score {{ font-size: 13px; color: #7f8c8d; }}
    .adu-badge {{ margin-top: 6px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
    .adu-tag {{
      background: #8e44ad; color: white; padding: 3px 10px; border-radius: 12px;
      font-size: 11px; font-weight: 700; white-space: nowrap;
    }}
    .adu-detail {{ font-size: 12px; color: #555; }}
    .adu-table-tag {{
      background: rgba(142,68,173,0.12); color: #8e44ad; padding: 2px 8px;
      border-radius: 8px; font-size: 11px; font-weight: 600; white-space: nowrap;
    }}
    .deal-actions {{ display: flex; align-items: center; }}
    .redfin-btn {{
      display: inline-block;
      background: #d0021b;
      color: white;
      padding: 8px 16px;
      border-radius: 6px;
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
      white-space: nowrap;
    }}
    .redfin-btn:hover {{ background: #b0011a; }}

    /* ── Listings table ── */
    .table-wrap {{
      background: white;
      border-radius: 10px;
      padding: 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.07);
      overflow-x: auto;
    }}
    #listingsTable {{
      width: 100%;
      font-size: 13px;
      border-collapse: collapse;
    }}
    #listingsTable thead th {{
      background: #34495e;
      color: white;
      padding: 10px 12px;
      text-align: left;
      white-space: nowrap;
    }}
    #listingsTable tbody tr:nth-child(even) {{ background: #f8f9fa; }}
    #listingsTable tbody tr:hover {{ background: #eaf4fd; }}
    #listingsTable tbody td {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }}
    .rank-cell {{ font-weight: 700; color: #bdc3c7; text-align: center; }}
    .num-cell {{ text-align: right; white-space: nowrap; }}
    .pct-cell {{ font-weight: 600; }}
    .score-cell {{ font-weight: 600; color: #2980b9; }}
    .address-cell {{ max-width: 200px; }}
    .addr-main {{ display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px; }}
    .link-cell {{ text-align: center; }}
    .table-link {{
      color: #2980b9;
      font-weight: 700;
      font-size: 15px;
      text-decoration: none;
    }}
    .table-link:hover {{ color: #d0021b; }}

    /* ── Diagnostics ── */
    .diag-wrap {{
      background: white;
      border-radius: 10px;
      padding: 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.07);
      text-align: center;
    }}
    .diag-img {{ max-width: 100%; border-radius: 6px; }}
    .no-diag {{ color: #7f8c8d; font-style: italic; }}

    /* ── Cities note ── */
    .cities-note {{
      background: #eaf4fd;
      border-left: 4px solid #2980b9;
      padding: 12px 16px;
      border-radius: 4px;
      font-size: 13px;
      margin-bottom: 20px;
      color: #2c3e50;
    }}

    /* DataTables overrides */
    .dataTables_wrapper .dataTables_filter input {{
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 4px 8px;
      font-size: 13px;
    }}
    .dataTables_wrapper .dataTables_length select {{
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 2px 6px;
    }}

    @media (max-width: 700px) {{
      .content {{ padding: 20px 16px; }}
      .kpi-bar {{ padding: 16px; }}
      .tabs {{ padding: 0 16px; }}
      .header {{ padding: 20px 16px; }}
    }}
  </style>
</head>
<body>

<div class="header">
  <h1>🏠 Home Finder Report</h1>
  <p>Generated {run_date}</p>
</div>

<div class="kpi-bar">
  <div class="kpi">
    <div class="kpi-value">{n_scored}</div>
    <div class="kpi-label">Listings matching criteria</div>
  </div>
  <div class="kpi">
    <div class="kpi-value">{n_deals}</div>
    <div class="kpi-label">Top deals</div>
  </div>
  <div class="kpi">
    <div class="kpi-value">{med_price}</div>
    <div class="kpi-label">Median listing price</div>
  </div>
  <div class="kpi">
    <div class="kpi-value">{med_pct}</div>
    <div class="kpi-label">Median % below predicted</div>
  </div>
  <div class="kpi">
    <div class="kpi-value">{n_adu}</div>
    <div class="kpi-label">Likely ADU / MIL</div>
  </div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('deals', this)">Top Deals</button>
  <button class="tab-btn" onclick="switchTab('listings', this)">All Listings ({n_scored})</button>
  <button class="tab-btn" onclick="switchTab('diagnostics', this)">Model Diagnostics</button>
</div>

<div class="content">

  <!-- ── Top Deals ── -->
  <div id="tab-deals" class="tab-panel active">
    <h2 class="section-title">Top Deals</h2>
    <div class="cities-note">
      Cities: {cities_list} &nbsp;·&nbsp;
      Homes ranked by composite score (60% value vs. model + 25% sqft/dollar + 15% rooms/dollar).
    </div>
    {deal_cards_html}
  </div>

  <!-- ── All Listings ── -->
  <div id="tab-listings" class="tab-panel">
    <h2 class="section-title">All Listings Matching Criteria</h2>
    <div class="table-wrap">
      <table id="listingsTable" class="display">
        <thead>
          <tr>
            <th>#</th>
            <th>Address</th>
            <th>City</th>
            <th>Price</th>
            <th>Predicted</th>
            <th>% Below</th>
            <th>Beds</th>
            <th>Baths</th>
            <th>Sqft</th>
            <th>$/Sqft</th>
            <th>Built</th>
            <th>DOM</th>
            <th>ADU</th>
            <th>Score</th>
            <th>Link</th>
          </tr>
        </thead>
        <tbody>
          {table_rows_html}
        </tbody>
      </table>
    </div>
  </div>

  <!-- ── Model Diagnostics ── -->
  <div id="tab-diagnostics" class="tab-panel">
    <h2 class="section-title">Model Diagnostics</h2>
    <div class="diag-wrap">
      {diag_section}
    </div>
  </div>

</div>

<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
<script>
  function switchTab(name, btn) {{
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
    // Redraw DataTable if switching to listings (fixes column widths)
    if (name === 'listings' && $.fn.DataTable.isDataTable('#listingsTable')) {{
      $('#listingsTable').DataTable().columns.adjust();
    }}
  }}

  $(document).ready(function() {{
    $('#listingsTable').DataTable({{
      pageLength: 25,
      order: [[13, 'desc']],
      columnDefs: [
        {{ orderable: false, targets: [14] }},
        {{ type: 'num-fmt', targets: [3, 4, 9] }},
      ],
      language: {{
        search: 'Filter:',
        lengthMenu: 'Show _MENU_ listings',
        info: 'Showing _START_–_END_ of _TOTAL_ listings',
        emptyTable: 'No listings match your criteria.',
      }}
    }});
  }});
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML report saved → {output_path}")
    return output_path
