"""
Send a simple, human, student-voice summary email about the FITB project.
Usage:  python send_project_email.py --repo https://github.com/<user>/<repo>
"""
from __future__ import annotations
import argparse
from fitb_analysis import send_email, OUT, C_INK, C_ACCENT, C_GOLD, C_MUTE, C_BG


def build(repo_url: str) -> str:
    link = repo_url or "(GitHub link — coming once the repo is pushed)"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="margin:0;background:{C_BG};font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:{C_INK};">
<div style="max-width:640px;margin:0 auto;background:#fff;padding:26px 30px;line-height:1.65;font-size:15px;">
  <p>Hi,</p>

  <p>Sharing a project I built this week — a fundamental analysis of <b>Fifth Third Bancorp ($FITB)</b>.
  I wanted to try valuing a bank the way an investment bank actually would, and learn the tools properly rather than
  just reading about them.</p>

  <p><b>Why this stock:</b> Fifth Third just bought <b>Comerica</b> in a ~$12.7B all-stock deal (closed Feb 2026),
  which made it the 9th-largest US bank (~$288B in assets). Right after the deal, reported EPS fell to $0.15 — but only
  because of ~$635M of one-time merger costs. So the real question was: what is the business worth once you look past
  that noise?</p>

  <p><b>What I did (in plain English, but keeping the real terms):</b></p>
  <ul style="padding-left:18px;">
    <li>Built <b>6 valuation methods</b> — Dividend Discount Model (DDM), Residual Income, Justified P/TBV
        = (ROTCE − g) / (COE − g), Justified P/E, peer comps, and analyst consensus.</li>
    <li>Did a <b>consumer-banking deep dive</b> — the loan book by product (auto, mortgage, home equity, credit card, solar),
        their yields (finance charges), interchange + deposit fees, and the segment profit &amp; loss.</li>
    <li>Ran the <b>standard trend analysis</b> — horizontal analysis, common-size (vertical) analysis, and a
        <b>rate/volume variance</b> decomposition (the same one banks show in their 10-K).</li>
    <li>Made a <b>driver-based earnings forecast</b> — EPS about <b>$3.68 (2026E) → $4.80 (2028E)</b>.</li>
    <li>Added <b>investment-bank techniques</b> — trading comps, <b>M&amp;A accretion/dilution</b> and tangible-book
        <b>earnback</b> on the Comerica deal, <b>Sum-of-the-Parts</b>, and a two-way sensitivity table.</li>
    <li>And two I was proud of: a <b>P/TBV-vs-ROTCE peer regression</b> (the "warranted multiple") and a
        <b>reverse DCF</b> that backs out the growth the market is already pricing in.</li>
  </ul>

  <p><b>What I found:</b> putting all <b>14 methods</b> together, the weighted target came out around <b>$58</b> vs a
  price of about <b>$58</b> — so, roughly <b>fairly valued (HOLD)</b>. The reverse DCF was the clincher: the market is
  pricing in ~<b>3.9% long-term growth</b>, almost exactly my base-case 4.0%. The deal is dilutive on day one but
  ~<b>+13.5% accretive by 2027E</b> once the $850M of cost synergies land, with a &lt;3-year tangible-book earnback.</p>

  <p><b>Biggest lesson:</b> the methods disagree with each other, and that's actually the point — a DDM lowballs a bank
  that keeps most of its earnings, while residual income and P/TBV capture the compounding. Showing the range is more
  honest than picking one number.</p>

  <p>All the code is open-source and reproducible (it pulls live data with <code>yfinance</code> and has offline
  fallbacks):<br><a href="{link}" style="color:{C_ACCENT};font-weight:700;">{link}</a></p>

  <p style="color:{C_MUTE};font-size:13px;">P.S. This is a learning / portfolio project — <b>not investment advice</b>.
  Numbers are from FITB's FY2025 10-K and Q4-2025 earnings release; a few M&amp;A/forecast inputs are clearly-labeled estimates.</p>

  <p>Thanks for reading,<br>Hanmanth</p>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, default="")
    ap.add_argument("--to", type=str, default=None)
    args = ap.parse_args()
    html = build(args.repo)
    (OUT / "project_summary_email.html").write_text(html, encoding="utf-8")
    ok = send_email(html, {}, to_override=args.to,
                    subject="My FITB equity research project (Fifth Third Bancorp) — code + writeup")
    print("email sent" if ok else "email NOT sent")


if __name__ == "__main__":
    main()
