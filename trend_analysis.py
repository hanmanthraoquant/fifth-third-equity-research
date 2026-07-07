"""
================================================================================
 FITB Consumer Banking — Multi-Year Trend Analysis (FY2023 → FY2025)
================================================================================
Applies the STANDARD financial-statement-analysis toolkit that equity analysts
and the CFA curriculum use, so the output is textbook-defensible:

  1. HORIZONTAL ANALYSIS  — year-over-year % change + compound annual growth
                             rate (CAGR) for every line.
  2. VERTICAL / COMMON-SIZE ANALYSIS — each item as a % of its total (loan mix,
                             fee mix), to show shifting composition.
  3. RATE / VOLUME VARIANCE DECOMPOSITION — the method banks are REQUIRED to
                             disclose in the 10-K "Analysis of Net Interest
                             Income": split the change in interest income into
                             the part driven by balance growth (volume) vs. the
                             part driven by yield change (rate), with the mixed
                             term allocated proportionally.
  4. INTERCHANGE DRIVER ESTIMATE — a transparent, clearly-labeled model that
                             splits the bundled "Consumer banking revenue" into
                             interchange vs. deposit service charges, with a
                             sensitivity range (FITB does not disclose the split).

All raw figures from FITB's FY2025 Form 10-K (3-year average-balance / yield
tables and noninterest-income detail). $ in millions. NOT investment advice.

Run:  python trend_analysis.py            # build + email
      python trend_analysis.py --no-email
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

from fitb_analysis import send_email, C_INK, C_ACCENT, C_GOLD, C_GOOD, C_BAD, C_MUTE, C_BG, OUT

YEARS = [2023, 2024, 2025]  # ascending

# ── 3-year consumer loan data (10-K avg-balance / yield table) ────────────────
# Each: balances[FY23,FY24,FY25], income[...], yield_pct[...]
LOANS = {
    "Residential mortgage": {"bal": [18002, 17537, 18194], "inc": [621, 645, 727], "yld": [3.45, 3.68, 4.00]},
    "Home equity":          {"bal": [3936, 4002, 4491],     "inc": [298, 330, 332], "yld": [7.58, 8.25, 7.40]},
    "Indirect (auto)":      {"bal": [15944, 15583, 17338],  "inc": [687, 822, 974], "yld": [4.31, 5.27, 5.62]},
    "Credit card":          {"bal": [1800, 1719, 1665],     "inc": [252, 236, 239], "yld": [14.00, 13.70, 14.34]},
    "Solar installation":   {"bal": [2958, 3960, 4333],     "inc": [180, 318, 368], "yld": [6.09, 8.04, 8.48]},
    "Other consumer":       {"bal": [3164, 2700, 2435],     "inc": [277, 248, 225], "yld": [8.74, 9.19, 9.26]},
}
CONSUMER_TOTAL = {"bal": [45804, 45501, 48456], "inc": [2315, 2599, 2865], "yld": [5.05, 5.71, 5.91]}

# ── 3-year fee income (noninterest income detail) ─────────────────────────────
FEES = {
    "Consumer banking revenue": [546, 555, 571],  # interchange + deposit svc chgs
    "Wealth & asset mgmt":      [581, 647, 704],
    "Mortgage banking net":     [250, 211, 227],
}
TOTAL_NONINT_INCOME = [2881, 2849, 3035]

# ── 3-year deposit franchise (avg bal, interest, rate paid) — deposit beta ─────
DEPOSITS = {
    "Interest checking": {"bal": [52536, 58757, 57484], "int": [1555, 1927, 1514], "rate": [2.96, 3.28, 2.63]},
    "Money market":      {"bal": [30943, 36165, 37406], "int": [666, 1050, 900],   "rate": [2.15, 2.90, 2.41]},
    "Savings":           {"bal": [20872, 17594, 16663], "int": [147, 119, 78],     "rate": [0.71, 0.68, 0.47]},
}
DEMAND_DEPOSITS_AVG = [46195, 40314, 40926]  # noninterest-bearing (free funding)

# ── Interchange model assumptions (transparent, sensitivity-tested) ───────────
INTERCHANGE_SHARES = [0.40, 0.45, 0.50, 0.55, 0.60]  # of Consumer banking revenue
INTERCHANGE_BASE = 0.45
DURBIN = "Regulated debit cap: 21c + 0.05% of ticket + 1c fraud adjustment / transaction"


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def cagr(first: float, last: float, periods: int) -> float:
    """Compound annual growth rate — standard horizontal-analysis metric."""
    if first <= 0:
        return float("nan")
    return (last / first) ** (1 / periods) - 1


def yoy(series: list[float]) -> list[float]:
    """Year-over-year % change for each step."""
    return [(series[i] / series[i - 1] - 1) if series[i - 1] else float("nan")
            for i in range(1, len(series))]


def rate_volume_decomposition(bal0, yld0, bal1, yld1):
    """
    Classic 10-K rate/volume analysis. Decompose the change in interest income
    into volume, rate and a proportionally-allocated mixed term.
      income = balance x yield
      ΔVolume = ΔBalance x prior_yield
      ΔRate   = ΔYield   x prior_balance
      ΔMixed  = ΔBalance x ΔYield  (allocated pro-rata to volume & rate)
    yields passed as percent (e.g. 5.91). Returns (volume$, rate$, total$).
    """
    r0, r1 = yld0 / 100.0, yld1 / 100.0
    d_bal, d_rate = bal1 - bal0, r1 - r0
    vol = d_bal * r0
    rate = d_rate * bal0
    mixed = d_bal * d_rate
    denom = abs(vol) + abs(rate)
    if denom > 0:
        vol += mixed * abs(vol) / denom
        rate += mixed * abs(rate) / denom
    return vol, rate, (bal1 * r1 - bal0 * r0)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25); ax.tick_params(colors=C_MUTE)


def chart_income_trend() -> Path:
    """Stacked interest income by product across 3 years (horizontal analysis)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=130)
    bottoms = np.zeros(3)
    cols = [C_ACCENT, C_GOOD, C_GOLD, C_BAD, "#6a8caf", C_MUTE]
    for (name, d), col in zip(LOANS.items(), cols):
        ax.bar([str(y) for y in YEARS], d["inc"], bottom=bottoms, label=name, color=col)
        bottoms += np.array(d["inc"])
    for i, tot in enumerate(CONSUMER_TOTAL["inc"]):
        ax.text(i, tot + 30, f"${tot:,}M", ha="center", fontsize=9, fontweight="bold", color=C_INK)
    ax.set_title("Consumer interest income (finance charges) by product, FY23–25 ($M)",
                 color=C_INK, fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2); _style(ax)
    p = OUT / "tr_income.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


def chart_yield_trend() -> Path:
    """Yield trend lines by product (rate analysis)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=130)
    cols = [C_ACCENT, C_GOOD, C_GOLD, C_BAD, "#6a8caf", C_MUTE]
    for (name, d), col in zip(LOANS.items(), cols):
        ax.plot(YEARS, d["yld"], marker="o", lw=1.8, color=col, label=name)
    ax.plot(YEARS, CONSUMER_TOTAL["yld"], marker="s", lw=2.4, color=C_INK, ls="--", label="Blended")
    ax.set_xticks(YEARS); ax.set_ylabel("Yield %")
    ax.set_title("Portfolio yield by product, FY23–25", color=C_INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2); _style(ax)
    p = OUT / "tr_yield.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


def chart_rate_volume() -> Path:
    """Rate/volume decomposition of FY24->FY25 interest-income growth, by product."""
    names, vols, rates = [], [], []
    for name, d in LOANS.items():
        v, r, _ = rate_volume_decomposition(d["bal"][1], d["yld"][1], d["bal"][2], d["yld"][2])
        names.append(name); vols.append(v); rates.append(r)
    x = np.arange(len(names)); w = 0.4
    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=130)
    ax.bar(x - w/2, vols, w, label="Volume effect", color=C_ACCENT)
    ax.bar(x + w/2, rates, w, label="Rate effect", color=C_GOLD)
    ax.axhline(0, color=C_MUTE, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("$M contribution")
    ax.set_title("What drove FY24→FY25 interest-income growth? (Rate vs Volume)",
                 color=C_INK, fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8); _style(ax)
    p = OUT / "tr_ratevol.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


def chart_common_size() -> Path:
    """Vertical/common-size loan mix: 2023 vs 2025."""
    names = list(LOANS.keys())
    mix23 = [LOANS[n]["bal"][0] / CONSUMER_TOTAL["bal"][0] * 100 for n in names]
    mix25 = [LOANS[n]["bal"][2] / CONSUMER_TOTAL["bal"][2] * 100 for n in names]
    x = np.arange(len(names)); w = 0.4
    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=130)
    ax.bar(x - w/2, mix23, w, label="FY2023", color=C_MUTE)
    ax.bar(x + w/2, mix25, w, label="FY2025", color=C_ACCENT)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("% of consumer loans")
    ax.set_title("Common-size loan mix shift, FY2023 vs FY2025", color=C_INK, fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8); _style(ax)
    p = OUT / "tr_mix.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


def chart_deposit_beta() -> Path:
    """Rate paid on interest-bearing deposits (deposit-cost / beta story)."""
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=130)
    cols = [C_ACCENT, C_GOLD, C_GOOD]
    for (name, d), col in zip(DEPOSITS.items(), cols):
        ax.plot(YEARS, d["rate"], marker="o", lw=1.8, color=col, label=name)
    ax.set_xticks(YEARS); ax.set_ylabel("Rate paid %")
    ax.set_title("Deposit cost by product, FY23–25 (funding-cost trend)", color=C_INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=8); _style(ax)
    p = OUT / "tr_deposit.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────
def build_html(images: dict) -> str:
    today = date.today().strftime("%d %b %Y")

    def img(k, alt):
        return (f'<img src="cid:{k}" alt="{alt}" style="width:100%;max-width:720px;'
                f'border:1px solid #e5e9f0;border-radius:8px;margin:8px 0;">') if k in images else ""

    # horizontal analysis table (loans): balance, income, yield with YoY + CAGR
    loan_rows = ""
    for name, d in LOANS.items():
        g = cagr(d["inc"][0], d["inc"][2], 2) * 100
        loan_rows += (
            f"<tr><td>{name}</td>"
            f"<td style='text-align:right'>${d['inc'][0]}</td>"
            f"<td style='text-align:right'>${d['inc'][1]}</td>"
            f"<td style='text-align:right'>${d['inc'][2]}</td>"
            f"<td style='text-align:right;color:{C_GOOD if g>=0 else C_BAD};font-weight:700'>{g:+.1f}%</td>"
            f"<td style='text-align:right'>{d['yld'][0]:.2f}→{d['yld'][2]:.2f}%</td></tr>")
    gt = cagr(CONSUMER_TOTAL["inc"][0], CONSUMER_TOTAL["inc"][2], 2) * 100
    loan_rows += (
        f"<tr style='border-top:2px solid {C_INK};font-weight:800'><td>Total consumer</td>"
        f"<td style='text-align:right'>${CONSUMER_TOTAL['inc'][0]:,}</td>"
        f"<td style='text-align:right'>${CONSUMER_TOTAL['inc'][1]:,}</td>"
        f"<td style='text-align:right'>${CONSUMER_TOTAL['inc'][2]:,}</td>"
        f"<td style='text-align:right;color:{C_GOOD}'>{gt:+.1f}%</td>"
        f"<td style='text-align:right'>{CONSUMER_TOTAL['yld'][0]:.2f}→{CONSUMER_TOTAL['yld'][2]:.2f}%</td></tr>")

    # rate/volume decomposition table (FY24->FY25)
    rv_rows = ""; tot_v = tot_r = 0.0
    for name, d in LOANS.items():
        v, r, tot = rate_volume_decomposition(d["bal"][1], d["yld"][1], d["bal"][2], d["yld"][2])
        tot_v += v; tot_r += r
        rv_rows += (f"<tr><td>{name}</td>"
                    f"<td style='text-align:right'>${v:+.0f}M</td>"
                    f"<td style='text-align:right'>${r:+.0f}M</td>"
                    f"<td style='text-align:right;font-weight:700'>${tot:+.0f}M</td></tr>")
    tot_all = tot_v + tot_r
    vol_pct = tot_v / tot_all * 100 if tot_all else 0
    rv_rows += (f"<tr style='border-top:2px solid {C_INK};font-weight:800'><td>Total</td>"
                f"<td style='text-align:right'>${tot_v:+.0f}M</td>"
                f"<td style='text-align:right'>${tot_r:+.0f}M</td>"
                f"<td style='text-align:right'>${tot_all:+.0f}M</td></tr>")

    # fee horizontal analysis
    fee_rows = ""
    for name, s in FEES.items():
        g = cagr(s[0], s[2], 2) * 100
        fee_rows += (f"<tr><td>{name}</td><td style='text-align:right'>${s[0]}</td>"
                     f"<td style='text-align:right'>${s[1]}</td><td style='text-align:right'>${s[2]}</td>"
                     f"<td style='text-align:right;color:{C_GOOD if g>=0 else C_BAD};font-weight:700'>{g:+.1f}%</td></tr>")

    # interchange sensitivity
    cbr = FEES["Consumer banking revenue"][2]
    ic_rows = ""
    for sh in INTERCHANGE_SHARES:
        ic = cbr * sh; dep = cbr * (1 - sh)
        hi = "font-weight:800;background:#fff7e6;" if abs(sh - INTERCHANGE_BASE) < 1e-9 else ""
        ic_rows += (f"<tr style='{hi}'><td style='text-align:right'>{sh*100:.0f}%</td>"
                    f"<td style='text-align:right'>${ic:.0f}M</td>"
                    f"<td style='text-align:right'>${dep:.0f}M</td></tr>")
    ic_base = cbr * INTERCHANGE_BASE

    # deposit cost trend summary
    dep_beta_txt = (f"blended interest-bearing deposit cost fell as the Fed peaked — e.g. interest-checking "
                    f"{DEPOSITS['Interest checking']['rate'][1]:.2f}%→{DEPOSITS['Interest checking']['rate'][2]:.2f}%, "
                    f"money-market {DEPOSITS['Money market']['rate'][1]:.2f}%→{DEPOSITS['Money market']['rate'][2]:.2f}% "
                    f"from FY24 to FY25 — the tailwind behind rising NIM.")
    demand_cagr = cagr(DEMAND_DEPOSITS_AVG[0], DEMAND_DEPOSITS_AVG[2], 2) * 100

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;background:{C_BG};font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:{C_INK};">
<div style="max-width:760px;margin:0 auto;background:#fff;">
  <div style="background:{C_INK};color:#fff;padding:26px 30px;">
    <div style="font-size:12px;letter-spacing:2px;color:{C_GOLD};font-weight:700;">CONSUMER BANKING · MULTI-YEAR TREND ANALYSIS</div>
    <div style="font-size:24px;font-weight:800;margin-top:4px;">Fifth Third — FY2023 → FY2025</div>
    <div style="font-size:13px;color:#c7d0dc;margin-top:6px;">{today} · methods: horizontal · common-size · rate/volume variance</div>
  </div>

  <div style="padding:22px 30px;">
    <div style="background:{C_BG};border-radius:8px;padding:12px 16px;font-size:12.5px;color:{C_MUTE};line-height:1.55;">
      <b style="color:{C_INK}">Methodology.</b> This note uses the standard financial-statement-analysis toolkit:
      <b>horizontal analysis</b> (YoY change + 2-yr CAGR), <b>vertical / common-size analysis</b> (mix as % of total), and the
      <b>rate/volume variance decomposition</b> that banks are required to disclose in the 10-K "Analysis of Net Interest Income."
      Interest income = average balance × yield, so its change decomposes cleanly into a <i>volume</i> effect (balance growth) and a
      <i>rate</i> effect (yield change). Data: FITB FY2025 10-K.
    </div>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">1 · Horizontal analysis — the finance-charge engine (interest income)</h2>
    {img('tr_income','income trend')}
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:{C_BG}"><th style="text-align:left;padding:6px">Product</th><th style="text-align:right;padding:6px">FY23</th><th style="text-align:right;padding:6px">FY24</th><th style="text-align:right;padding:6px">FY25</th><th style="text-align:right;padding:6px">CAGR</th><th style="text-align:right;padding:6px">Yield Δ</th></tr>
      {loan_rows}
    </table>
    <p style="line-height:1.6;font-size:13px;color:{C_MUTE};margin-top:6px">Consumer interest income compounded at <b>{gt:.1f}%</b> over two years.
    The engines of growth are <b>indirect/auto</b> and <b>solar</b> — the credit-card line barely moved (Fifth Third is not a card lender).</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">2 · Rate / Volume decomposition (FY24 → FY25)</h2>
    <p style="line-height:1.6;font-size:14px">This is the standard bank disclosure: was interest-income growth driven by lending <b>more</b> (volume)
    or by earning a <b>higher yield</b> (rate)? For the consumer book, growth was <b>~{vol_pct:.0f}% volume-driven</b> — Fifth Third grew the balance sheet,
    it didn't just ride higher rates.</p>
    {img('tr_ratevol','rate volume')}
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:{C_BG}"><th style="text-align:left;padding:6px">Product</th><th style="text-align:right;padding:6px">Volume effect</th><th style="text-align:right;padding:6px">Rate effect</th><th style="text-align:right;padding:6px">Total Δ</th></tr>
      {rv_rows}
    </table>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">3 · Vertical / common-size analysis — mix shift</h2>
    <p style="line-height:1.6;font-size:14px">Holding the book to 100%, the mix is tilting toward <b>auto and solar</b> and away from <b>legacy "other consumer"</b> —
    a deliberate move up the yield curve while keeping it secured.</p>
    {img('tr_mix','common size')}

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">4 · Yield trend (rate analysis)</h2>
    {img('tr_yield','yield trend')}
    <p style="line-height:1.6;font-size:13px;color:{C_MUTE}">Auto yields re-priced sharply higher (4.31%→5.62% over two years) as old low-rate paper rolled off;
    credit-card yield is stable ~14%; mortgage yields drift up slowly (long-duration, slow to reprice).</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">5 · Fee income trend (horizontal analysis)</h2>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:{C_BG}"><th style="text-align:left;padding:6px">Fee line</th><th style="text-align:right;padding:6px">FY23</th><th style="text-align:right;padding:6px">FY24</th><th style="text-align:right;padding:6px">FY25</th><th style="text-align:right;padding:6px">CAGR</th></tr>
      {fee_rows}
    </table>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">6 · Interchange vs. deposit-fee split — driver model</h2>
    <p style="line-height:1.6;font-size:14px">Fifth Third bundles interchange and deposit service charges into one line ("Consumer banking revenue", <b>${cbr}M</b> FY25),
    so the split must be <b>modeled, not read off</b>. Per the 10-K this line "consists primarily of interchange fees... [and] monthly service charges on
    consumer deposit accounts." Debit interchange is capped by Durbin ({DURBIN}), so it scales with card <i>volume</i>, not price. Base case assumes
    interchange ≈ <b>{INTERCHANGE_BASE*100:.0f}%</b> of the line (≈ <b>${ic_base:.0f}M</b>); sensitivity:</p>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:{C_BG}"><th style="text-align:right;padding:6px">Interchange share</th><th style="text-align:right;padding:6px">Interchange $</th><th style="text-align:right;padding:6px">Deposit svc chg + other $</th></tr>
      {ic_rows}
    </table>
    <p style="line-height:1.6;font-size:12px;color:{C_BAD};margin-top:6px"><b>Estimate — not disclosed by the company.</b> Shown as a transparent range;
    the FY25 increase in this line was attributed by management to higher deposit/overdraft fees, which argues the deposit-fee share is rising.</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">7 · Deposit cost / funding trend</h2>
    <p style="line-height:1.6;font-size:14px">The other side of the ledger: {dep_beta_txt} Non-interest-bearing <b>demand deposits</b> averaged
    ${DEMAND_DEPOSITS_AVG[2]/1000:.1f}B ({demand_cagr:+.1f}% 2-yr CAGR) — free funding that lifts the consumer segment's spread.</p>
    {img('tr_deposit','deposit cost')}

    <div style="background:{C_INK};color:#fff;padding:18px 22px;border-radius:8px;margin-top:16px">
      <div style="font-size:12px;letter-spacing:1px;color:{C_GOLD}">TREND TAKEAWAY</div>
      <div style="font-size:15px;line-height:1.5;margin-top:6px">Two-year story: consumer interest income compounded at {gt:.1f}% and the growth was <b>{vol_pct:.0f}% volume-driven</b>
      (balance growth, led by auto &amp; solar), <b>not</b> just rate-driven — a higher-quality growth signal. Deposit costs are now <i>falling</i> off the rate peak,
      widening the spread. Fee income is led by Wealth &amp; Asset Management; the interchange line is Durbin-capped and volume-linked.</div>
    </div>

    <p style="font-size:11px;color:{C_MUTE};margin-top:16px;line-height:1.5">Methods: horizontal analysis (YoY, CAGR), vertical/common-size analysis,
    rate/volume variance decomposition (10-K standard). Source: FITB FY2025 Form 10-K (SEC EDGAR). Interchange split is a modeled estimate.
    Educational portfolio project — <b>not investment advice</b>.</p>
  </div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--to", type=str, default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("  FITB CONSUMER — MULTI-YEAR TREND ANALYSIS (FY2023->FY2025)")
    print("=" * 70)
    gt = cagr(CONSUMER_TOTAL["inc"][0], CONSUMER_TOTAL["inc"][2], 2) * 100
    print(f"Consumer interest income: ${CONSUMER_TOTAL['inc'][0]:,}M -> ${CONSUMER_TOTAL['inc'][2]:,}M  (CAGR {gt:+.1f}%)")
    print("\nRate/Volume decomposition (FY24->FY25):")
    tv = tr = 0.0
    for name, d in LOANS.items():
        v, r, tot = rate_volume_decomposition(d["bal"][1], d["yld"][1], d["bal"][2], d["yld"][2])
        tv += v; tr += r
        print(f"   {name:22s} vol ${v:+6.0f}M | rate ${r:+6.0f}M | total ${tot:+6.0f}M")
    print(f"   {'TOTAL':22s} vol ${tv:+6.0f}M | rate ${tr:+6.0f}M  ->  {tv/(tv+tr)*100:.0f}% volume-driven")
    print(f"\nInterchange base estimate ({INTERCHANGE_BASE*100:.0f}% of ${FEES['Consumer banking revenue'][2]}M): "
          f"${FEES['Consumer banking revenue'][2]*INTERCHANGE_BASE:.0f}M (modeled, not disclosed)")

    images = {}
    for k, fn in (("tr_income", chart_income_trend), ("tr_ratevol", chart_rate_volume),
                  ("tr_mix", chart_common_size), ("tr_yield", chart_yield_trend),
                  ("tr_deposit", chart_deposit_beta)):
        images[k] = str(fn())

    html = build_html(images)
    report = OUT / "FITB_consumer_trend.html"
    report.write_text(html, encoding="utf-8")
    print(f"\nReport: {report}")

    if not args.no_email:
        send_email(html, images, to_override=args.to,
                   subject="FITB Consumer — Multi-Year Trend Analysis (FY23-25): horizontal, common-size & rate/volume")
    else:
        print("[--no-email] skipped.")
    print("=" * 70, "\nDONE.")


if __name__ == "__main__":
    main()
