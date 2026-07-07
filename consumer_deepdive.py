"""
================================================================================
 FITB — Consumer & Small Business Banking (CSBB): Deep-Dive Analytics
================================================================================
A granular, filing-sourced deep dive into Fifth Third's consumer franchise:
the loan book by product (balances, interest income / finance charges, yields),
the fee engine (interchange + deposit service charges), the segment P&L, the
deposit franchise, and consumer credit quality.

All figures hand-verified from Fifth Third's 2025 Form 10-K and Q4-2025 earnings
release (SEC EDGAR). $ in millions unless noted.

Run:
  python consumer_deepdive.py            # build charts + HTML, then email
  python consumer_deepdive.py --no-email # build only
================================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import date

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse the email plumbing + palette from the main engine
from fitb_analysis import send_email, C_INK, C_ACCENT, C_GOLD, C_GOOD, C_BAD, C_MUTE, C_BG, OUT

# ─────────────────────────────────────────────────────────────────────────────
# CONSUMER DATA — verified from FITB 2025 10-K + Q4-2025 earnings release
# ─────────────────────────────────────────────────────────────────────────────

# Consumer loan book. avg_q4/prior are average balances ($M) from the Q4-25
# release (Q4-25 | Q3-25 | Q4-24). fy_bal/fy_income/fy_yield are FY2025 average
# balance, interest income (finance charges), and yield from the 10-K.
LOAN_BOOK = [
    # product,                 avg_q4,  yoy%,  fy_bal,  fy_income, fy_yield
    ("Residential mortgage",   17660,  0.02,  18194,   727,   4.00),
    ("Home equity",             4769,  0.16,   4491,   332,   7.40),
    ("Indirect (auto)",        17879,  0.11,  17338,   974,   5.62),
    ("Credit card",             1694,  0.02,   1665,   239,  14.34),
    ("Solar installation",      4486,  0.08,   4333,   368,   8.48),
    ("Other consumer",          2345, -0.08,   2435,   225,   9.26),
]
CONSUMER_TOTAL = {"avg_q4": 48833, "yoy": 0.06, "fy_bal": 48456,
                  "fy_income": 2865, "fy_yield": 5.91}
TOTAL_PORTFOLIO_LOANS = 123430          # total avg portfolio loans Q4-25

# Consumer fee income ($M) — FY2025 | FY2024 | FY2023 (10-K noninterest income)
FEE_LINES = {
    "Consumer banking revenue": (571, 555, 546),   # interchange + deposit svc chgs
    "Wealth & asset mgmt":      (704, 647, 581),
    "Mortgage banking net":     (227, 211, 250),
}
TOTAL_NONINT_INCOME = (3035, 2849, 2881)

# CSBB segment P&L, Q4-2025 ($M, FTE) — from Q4-25 release segment table
SEGMENT_Q4 = {
    "nii": 1026, "provision": -84, "noninterest_income": 311,
    "noninterest_expense": -645, "pretax": 608,
}
TOTAL_NII_Q4 = 1533
SEG_NII_Q4 = {"Commercial": 581, "Consumer & SB": 1026, "Wealth": 52, "Corp/Other": -126}
CSBB_PRETAX_TREND = {"Q1-25": 648, "Q2-25": 665, "Q4-25": 608}  # $M pretax (FTE)

# Deposits & franchise
DEPOSITS = {"total_avg": 168384, "total_eop": 171819,
            "loan_to_core_deposit": 0.72, "demand_growth_yoy": 0.04}

# Cards / interchange / distribution
CARDS = {
    "durbin_cap": "21c + 0.05% of ticket + 1c fraud-prevention (regulated debit)",
    "credit_card_book_b": 1.694,
    "credit_card_share_of_consumer": 1694 / 48833,
    "credit_card_share_of_total": 1694 / 123430,
    "credit_card_yield": 14.34,
    "payments_processed_2024_tn": 17.0,     # total payments franchise (incl. commercial)
    "direct_express_cardholders_m": 3.4,    # federal benefits prepaid program (won Sep-2025)
    "branches": "1,000+ in 12 states",
}

# Consumer growth / share
GROWTH = {
    "households_yoy": 0.025, "southeast_yoy": 0.07,
    "georgia": 0.10, "carolinas": 0.09,
    "heloc_rank": "#2 origination share in footprint (up from #4)",
    "auto_growth": 0.11, "home_equity_growth": 0.16,
}

# Credit quality
CREDIT = {"nco_fy2025": 0.0062, "nco_fy2024": 0.0033,
          "nco_q4_total": 0.0040, "nco_q4_commercial": 0.0027,
          "acl_ratio": 0.0179}


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25); ax.tick_params(colors=C_MUTE)


def chart_loan_mix() -> Path:
    names = [x[0] for x in LOAN_BOOK]; bal = [x[2] for x in LOAN_BOOK]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=130)
    cols = [C_ACCENT, C_GOOD, C_GOLD, C_BAD, "#6a8caf", C_MUTE]
    order = np.argsort(bal)
    ax.barh([names[i] for i in order], [bal[i] for i in order],
            color=[cols[i] for i in order])
    for i, idx in enumerate(order):
        ax.text(bal[idx] + 150, i, f"${bal[idx]/1000:.1f}B", va="center", fontsize=8, color=C_INK)
    ax.set_title("Consumer loan book by product (avg, Q4-2025)", color=C_INK, fontweight="bold")
    _style(ax); ax.grid(axis="x", alpha=0.2); ax.grid(axis="y", visible=False)
    p = OUT / "cd_loan_mix.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


def chart_yields() -> Path:
    names = [x[0] for x in LOAN_BOOK]; yld = [x[5] for x in LOAN_BOOK]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=130)
    cols = [C_BAD if y >= 12 else (C_GOLD if y >= 8 else C_ACCENT) for y in yld]
    ax.bar(names, yld, color=cols)
    ax.axhline(CONSUMER_TOTAL["fy_yield"], color=C_INK, ls="--", lw=1.2,
               label=f"Blended {CONSUMER_TOTAL['fy_yield']:.2f}%")
    for i, y in enumerate(yld):
        ax.text(i, y + 0.2, f"{y:.2f}%", ha="center", fontsize=8, color=C_INK)
    ax.set_title("Portfolio yield by product — the finance-charge engine (FY2025)", color=C_INK, fontweight="bold")
    ax.set_ylabel("Yield %"); ax.legend(frameon=False, fontsize=8)
    plt.xticks(rotation=20, ha="right", fontsize=8); _style(ax)
    p = OUT / "cd_yields.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


def chart_interest_income() -> Path:
    names = [x[0] for x in LOAN_BOOK]; inc = [x[4] for x in LOAN_BOOK]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=130)
    ax.bar(names, inc, color=C_ACCENT)
    for i, v in enumerate(inc):
        ax.text(i, v + 8, f"${v}M", ha="center", fontsize=8, color=C_INK)
    ax.set_title("Interest income (finance charges) by product — FY2025 ($M)", color=C_INK, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=8); _style(ax)
    p = OUT / "cd_interest.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


def chart_nii_contribution() -> Path:
    labels = list(SEG_NII_Q4.keys()); vals = [max(v, 0) for v in SEG_NII_Q4.values()]
    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=130)
    cols = [C_MUTE, C_ACCENT, C_GOLD, "#cdd5e0"]
    wedges, _, _ = ax.pie(vals, labels=labels, autopct=lambda p: f"{p:.0f}%",
                          colors=cols, textprops={"fontsize": 8, "color": C_INK},
                          wedgeprops={"width": 0.45})
    ax.set_title("Net interest income by segment (Q4-2025)\nConsumer = the funding engine", color=C_INK, fontweight="bold", fontsize=10)
    p = OUT / "cd_nii.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


# ─────────────────────────────────────────────────────────────────────────────
# HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────
def build_html(images: dict) -> str:
    today = date.today().strftime("%d %b %Y")
    seg = SEGMENT_Q4
    nii_share = seg["nii"] / TOTAL_NII_Q4 * 100

    def img(k, alt):
        return (f'<img src="cid:{k}" alt="{alt}" style="width:100%;max-width:720px;'
                f'border:1px solid #e5e9f0;border-radius:8px;margin:8px 0;">') if k in images else ""

    loan_rows = ""
    for name, avg_q4, yoy, fy_bal, inc, yld in LOAN_BOOK:
        loan_rows += (
            f"<tr><td>{name}</td>"
            f"<td style='text-align:right'>${avg_q4/1000:.1f}B</td>"
            f"<td style='text-align:right;color:{C_GOOD if yoy>=0 else C_BAD}'>{yoy*100:+.0f}%</td>"
            f"<td style='text-align:right'>${inc}M</td>"
            f"<td style='text-align:right;font-weight:700'>{yld:.2f}%</td></tr>")
    loan_rows += (
        f"<tr style='border-top:2px solid {C_INK};font-weight:800'><td>Total consumer</td>"
        f"<td style='text-align:right'>${CONSUMER_TOTAL['avg_q4']/1000:.1f}B</td>"
        f"<td style='text-align:right;color:{C_GOOD}'>+{CONSUMER_TOTAL['yoy']*100:.0f}%</td>"
        f"<td style='text-align:right'>${CONSUMER_TOTAL['fy_income']:,}M</td>"
        f"<td style='text-align:right'>{CONSUMER_TOTAL['fy_yield']:.2f}%</td></tr>")

    fee_rows = ""
    for name, (y25, y24, y23) in FEE_LINES.items():
        fee_rows += (f"<tr><td>{name}</td><td style='text-align:right'>${y25}M</td>"
                     f"<td style='text-align:right'>${y24}M</td>"
                     f"<td style='text-align:right;color:{C_GOOD if y25>=y24 else C_BAD}'>{(y25/y24-1)*100:+.1f}%</td></tr>")

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;background:{C_BG};font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:{C_INK};">
<div style="max-width:760px;margin:0 auto;background:#fff;">
  <div style="background:{C_INK};color:#fff;padding:26px 30px;">
    <div style="font-size:12px;letter-spacing:2px;color:{C_GOLD};font-weight:700;">SEGMENT DEEP DIVE</div>
    <div style="font-size:24px;font-weight:800;margin-top:4px;">Fifth Third — Consumer &amp; Small Business Banking</div>
    <div style="font-size:13px;color:#c7d0dc;margin-top:6px;">{today} · figures from FY2025 10-K + Q4-2025 release</div>
  </div>

  <div style="display:flex;flex-wrap:wrap;padding:18px 30px;gap:14px;border-bottom:1px solid #eef1f6;">
    <div style="flex:1;min-width:110px"><div style="font-size:11px;color:{C_MUTE}">CSBB PRETAX (Q4)</div><div style="font-size:20px;font-weight:800">${seg['pretax']}M</div></div>
    <div style="flex:1;min-width:110px"><div style="font-size:11px;color:{C_MUTE}">SEGMENT NII (Q4)</div><div style="font-size:20px;font-weight:800;color:{C_ACCENT}">${seg['nii']:,}M</div></div>
    <div style="flex:1;min-width:110px"><div style="font-size:11px;color:{C_MUTE}">% OF BANK NII</div><div style="font-size:20px;font-weight:800;color:{C_GOOD}">{nii_share:.0f}%</div></div>
    <div style="flex:1;min-width:110px"><div style="font-size:11px;color:{C_MUTE}">CONSUMER LOANS</div><div style="font-size:20px;font-weight:800">${CONSUMER_TOTAL['avg_q4']/1000:.0f}B</div></div>
  </div>

  <div style="padding:22px 30px;">
    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">1 · The one-line story</h2>
    <p style="line-height:1.6;font-size:14px">Consumer &amp; Small Business Banking is Fifth Third's <b>profit and funding engine</b>. In Q4-2025 it generated
    <b>${seg['nii']:,}M of net interest income — {nii_share:.0f}% of the entire bank's NII</b> — plus <b>${seg['noninterest_income']}M</b> of fee income, for
    <b>${seg['pretax']}M of pretax profit</b>, the largest of any segment (Commercial was ${445}M). The deposits this franchise gathers fund the whole balance
    sheet; the loans it makes earn a blended <b>{CONSUMER_TOTAL['fy_yield']:.2f}%</b>. Growth is skewing to the high-yield, high-growth products
    (auto +{GROWTH['auto_growth']*100:.0f}%, home equity +{GROWTH['home_equity_growth']*100:.0f}%) and to the Southeast.</p>
    {img('cd_nii','nii by segment')}

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">2 · The loan book — balances, finance charges &amp; yields</h2>
    <p style="line-height:1.6;font-size:14px">Consumer is <b>${CONSUMER_TOTAL['avg_q4']/1000:.0f}B</b> of loans (~{CONSUMER_TOTAL['avg_q4']/TOTAL_PORTFOLIO_LOANS*100:.0f}% of the bank's
    ${TOTAL_PORTFOLIO_LOANS/1000:.0f}B book), throwing off <b>${CONSUMER_TOTAL['fy_income']:,}M</b> of interest income (finance charges) in FY2025. The mix matters: the
    <b>credit-card book is tiny (${CARDS['credit_card_book_b']:.1f}B, {CARDS['credit_card_share_of_consumer']*100:.1f}% of consumer loans)</b> but yields <b>{CARDS['credit_card_yield']:.2f}%</b> —
    Fifth Third is a <b>deposit/debit-led</b> consumer bank, not a card lender.</p>
    {img('cd_loan_mix','loan mix')}
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:{C_BG}"><th style="text-align:left;padding:6px">Product</th><th style="text-align:right;padding:6px">Avg bal (Q4)</th><th style="text-align:right;padding:6px">YoY</th><th style="text-align:right;padding:6px">Int. inc (FY25)</th><th style="text-align:right;padding:6px">Yield</th></tr>
      {loan_rows}
    </table>
    {img('cd_yields','yields')}
    {img('cd_interest','interest income')}

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">3 · The fee engine — interchange &amp; deposit charges</h2>
    <p style="line-height:1.6;font-size:14px"><b>Consumer banking revenue was ${FEE_LINES['Consumer banking revenue'][0]}M in FY2025.</b> Per the 10-K it "consists primarily of
    <b>interchange fees</b> earned when consumer credit &amp; debit cards are processed through card-association networks, <b>monthly service charges on consumer
    deposit accounts</b>, and other deposit-related charges." The FY25 rise was driven by higher deposit/overdraft fees. Debit interchange is <b>Durbin-capped</b>
    ({CARDS['durbin_cap']}) because Fifth Third is over $10B in assets — so this line grows with card <i>volume</i> and account count, not price.</p>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:{C_BG}"><th style="text-align:left;padding:6px">Fee line</th><th style="text-align:right;padding:6px">FY2025</th><th style="text-align:right;padding:6px">FY2024</th><th style="text-align:right;padding:6px">Δ</th></tr>
      {fee_rows}
    </table>
    <p style="line-height:1.6;font-size:13px;color:{C_MUTE};margin-top:6px">Context: the bank processed ~<b>${CARDS['payments_processed_2024_tn']:.0f}T</b> of total payments in 2024 (incl. commercial), and in Sep-2025
    won the <b>Direct Express</b> federal-benefits program (~{CARDS['direct_express_cardholders_m']:.1f}M prepaid debit cardholders) — a large, sticky, fee-and-deposit generator.</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">4 · The deposit franchise (why NII is so high)</h2>
    <p style="line-height:1.6;font-size:14px">The segment's NII is outsized because it owns the deposits. Bank-wide average deposits were <b>${DEPOSITS['total_avg']/1000:.0f}B</b>,
    the <b>loan-to-core-deposit ratio is {DEPOSITS['loan_to_core_deposit']*100:.0f}%</b>, and demand deposits grew <b>{DEPOSITS['demand_growth_yoy']*100:.0f}% YoY</b>. Under funds-transfer pricing,
    the consumer segment is credited for supplying this low-cost funding to the rest of the bank — which is why CSBB shows ${seg['nii']:,}M of NII versus ${SEG_NII_Q4['Commercial']}M for Commercial.</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">5 · Growth &amp; distribution</h2>
    <p style="line-height:1.6;font-size:14px">{CARDS['branches']}. Net-new consumer households <b>+{GROWTH['households_yoy']*100:.1f}% YoY</b>, led by the Southeast
    <b>(+{GROWTH['southeast_yoy']*100:.0f}%; Georgia +{GROWTH['georgia']*100:.0f}%, Carolinas +{GROWTH['carolinas']*100:.0f}%)</b>. Home-equity is now <b>{GROWTH['heloc_rank']}</b>.
    The strategy is textbook: gather cheap deposits in a growing region, then lend into higher-yield consumer products.</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">6 · Consumer credit quality</h2>
    <p style="line-height:1.6;font-size:14px">Total net charge-offs were <b>{CREDIT['nco_fy2025']*100:.0f} bps</b> in FY2025 (vs {CREDIT['nco_fy2024']*100:.0f} bps in FY2024). In Q4-25, total NCOs were
    {CREDIT['nco_q4_total']*100:.0f} bps and commercial only {CREDIT['nco_q4_commercial']*100:.0f} bps — so consumer runs a bit hotter, as expected given the card/solar/other mix. Allowance for credit losses is a
    conservative <b>{CREDIT['acl_ratio']*100:.2f}% of loans</b>. Credit is normalizing off unusually low levels, not deteriorating.</p>

    <div style="background:{C_INK};color:#fff;padding:18px 22px;border-radius:8px;margin-top:16px">
      <div style="font-size:12px;letter-spacing:1px;color:{C_GOLD}">DEEP-DIVE TAKEAWAY</div>
      <div style="font-size:15px;line-height:1.5;margin-top:6px">CSBB is the crown jewel: it funds the bank with cheap, growing deposits and lends into a well-diversified, {CONSUMER_TOTAL['fy_yield']:.2f}%-yield
      consumer book. The upside levers are auto/home-equity growth and Southeast household gains; the watch items are deposit-fee dependence (Durbin-capped) and
      normalizing consumer credit. It is a spread-and-deposit story, not a credit-card story.</div>
    </div>

    <p style="font-size:11px;color:{C_MUTE};margin-top:16px;line-height:1.5">Source: Fifth Third Bancorp FY2025 Form 10-K and Q4-2025 earnings release (SEC EDGAR).
    Educational portfolio project — <b>not investment advice</b>.</p>
  </div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--to", type=str, default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("  FITB — CONSUMER & SMALL BUSINESS BANKING DEEP DIVE")
    print("=" * 70)
    print(f"CSBB Q4 pretax ${SEGMENT_Q4['pretax']}M | NII ${SEGMENT_Q4['nii']}M "
          f"({SEGMENT_Q4['nii']/TOTAL_NII_Q4*100:.0f}% of bank) | consumer loans ${CONSUMER_TOTAL['avg_q4']/1000:.0f}B "
          f"@ {CONSUMER_TOTAL['fy_yield']:.2f}%")
    print("Interest income by product (FY2025):")
    for name, _, _, _, inc, yld in LOAN_BOOK:
        print(f"   {name:22s} ${inc:>4}M   {yld:>6.2f}%")

    images = {}
    for k, fn in (("cd_nii", chart_nii_contribution), ("cd_loan_mix", chart_loan_mix),
                  ("cd_yields", chart_yields), ("cd_interest", chart_interest_income)):
        images[k] = str(fn())

    html = build_html(images)
    report = OUT / "FITB_consumer_deepdive.html"
    report.write_text(html, encoding="utf-8")
    print(f"\nReport: {report}")

    if not args.no_email:
        send_email(html, images, to_override=args.to,
                   subject="FITB Deep Dive — Consumer & Small Business Banking: loan book, yields, interchange & credit")
    else:
        print("[--no-email] skipped.")
    print("=" * 70, "\nDONE.")


if __name__ == "__main__":
    main()
