"""
================================================================================
 FITB — Full-Fledged View & Forecast (Fundamentals + Consumer Trends → Outlook)
================================================================================
Ties the whole project together:
  • the intrinsic valuation (6-method blend from fitb_analysis.py),
  • the consumer-banking growth engine (trend_analysis.py),
  • and a forward EARNINGS FORECAST that produces a 12-month price target.

FORECAST METHODOLOGY (standard sell-side "earnings build" + forward-multiple TP):
  1. Driver-based pro-forma P&L, FY2026E–FY2028E:
        Revenue  = Net interest income (guidance-anchored, grown by driver)
                 + Fee income (grown at trend CAGR)
        PPNR     = Revenue − Operating expense (efficiency ratio × revenue)
        Pretax   = PPNR − Provision (net charge-off rate × loans)
        EPS      = (Pretax × (1−tax) − preferred divs) ÷ diluted shares
     Reflects the Comerica cost synergies ($850M by Q4-26) via a falling
     efficiency ratio, and buybacks via a shrinking share count.
  2. 12-month target = forward P/E × NTM EPS, cross-checked with P/TBV.
  3. Blended target = average of (intrinsic fair value, forward-P/E target,
     P/TBV target) — bull / base / bear.

Data: FITB FY2025 10-K + Q4-2025 release + 2026 management guidance; live price
via yfinance. $ in millions unless noted. NOT investment advice.

Run:  python full_view_forecast.py            # build + email
      python full_view_forecast.py --no-email
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

from fitb_analysis import (
    get_live_market_data, get_peer_multiples, cost_of_equity,
    scenario_valuation, peer_relative_value, blended_fair_value,
    FUND, ASSUMPTIONS, send_email, OUT,
    C_INK, C_ACCENT, C_GOLD, C_GOOD, C_BAD, C_MUTE, C_BG,
)
from trend_analysis import CONSUMER_TOTAL, cagr

TICKER = "FITB"

# ── FORECAST DRIVERS (documented, adjustable) ─────────────────────────────────
# Base year = FY2026E, the first full combined (post-Comerica) year.
FORECAST = {
    "nii_2026": 8750.0,        # mgmt 2026 NII guidance midpoint ($8.7–8.8B)
    "fees_2026": 3700.0,       # combined co. noninterest income estimate
    "avg_loans": 180000.0,     # combined avg loans (~$180B)
    "preferred_div": 180.0,    # annual preferred dividends
    "tax_rate": 0.22,
    "shares_2026": None,       # filled from live data (diluted, millions)
    "buyback_shrink": 0.015,   # ~1.5% share count reduction / yr
    # scenario drivers
    "nii_growth":  {"bear": 0.02, "base": 0.04, "bull": 0.06},
    "fee_growth":  {"bear": 0.03, "base": 0.05, "bull": 0.07},
    # efficiency ratio path (synergies pull it down over time)
    "efficiency":  {"bear": [0.590, 0.575, 0.565],
                    "base": [0.580, 0.545, 0.530],
                    "bull": [0.565, 0.525, 0.510]},
    "nco_rate":    {"bear": 0.0055, "base": 0.0040, "bull": 0.0032},
    # 12-month target multiples
    "fwd_pe":      {"bear": 10.5, "base": 12.5, "bull": 13.5},
    "target_ptbv": {"bear": 1.6, "base": 2.0, "bull": 2.3},
}
FY = [2026, 2027, 2028]


def project(case: str, shares0: float) -> dict:
    """Driver-based 3-year P&L → EPS path for one scenario."""
    nii = FORECAST["nii_2026"]; fees = FORECAST["fees_2026"]
    g_nii = FORECAST["nii_growth"][case]; g_fee = FORECAST["fee_growth"][case]
    eff = FORECAST["efficiency"][case]; nco = FORECAST["nco_rate"][case]
    shares = shares0
    rows = []
    for i, yr in enumerate(FY):
        if i > 0:
            nii *= (1 + g_nii); fees *= (1 + g_fee)
            shares *= (1 - FORECAST["buyback_shrink"])
        revenue = nii + fees
        opex = eff[i] * revenue
        ppnr = revenue - opex
        provision = nco * FORECAST["avg_loans"]
        pretax = ppnr - provision
        net = pretax * (1 - FORECAST["tax_rate"])
        to_common = net - FORECAST["preferred_div"]
        eps = to_common / shares
        rows.append({"year": yr, "nii": nii, "fees": fees, "revenue": revenue,
                     "opex": opex, "ppnr": ppnr, "provision": provision,
                     "pretax": pretax, "eps": eps, "shares": shares, "eff": eff[i]})
    return {"case": case, "rows": rows}


def price_targets(mkt: dict, proj: dict, tbvps: float, intrinsic_fv: float) -> dict:
    """12-month target: forward P/E × NTM EPS, P/TBV × TBVPS, and blend w/ intrinsic."""
    case = proj["case"]
    ntm_eps = proj["rows"][1]["eps"]        # FY2027E ≈ next-twelve-months anchor
    pe_target = FORECAST["fwd_pe"][case] * ntm_eps
    ptbv_target = FORECAST["target_ptbv"][case] * tbvps
    blended = float(np.mean([intrinsic_fv, pe_target, ptbv_target]))
    return {"ntm_eps": ntm_eps, "pe_target": pe_target,
            "ptbv_target": ptbv_target, "blended": blended}


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25); ax.tick_params(colors=C_MUTE)


def chart_eps(projs: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=130)
    x = np.arange(len(FY))
    base = [r["eps"] for r in projs["base"]["rows"]]
    bull = [r["eps"] for r in projs["bull"]["rows"]]
    bear = [r["eps"] for r in projs["bear"]["rows"]]
    ax.bar(x, base, 0.5, color=C_ACCENT, label="Base")
    ax.plot(x, bull, "o--", color=C_GOOD, lw=1.4, label="Bull")
    ax.plot(x, bear, "o--", color=C_BAD, lw=1.4, label="Bear")
    # 2025 actual anchor
    ax.axhline(FUND["fy2025"]["eps_diluted"], color=C_MUTE, ls=":", lw=1,
               label=f"FY25A ${FUND['fy2025']['eps_diluted']:.2f}")
    for i, v in enumerate(base):
        ax.text(i, v + 0.08, f"${v:.2f}", ha="center", fontsize=8, color=C_INK)
    ax.set_xticks(x); ax.set_xticklabels([f"{y}E" for y in FY])
    ax.set_ylabel("Diluted EPS ($)")
    ax.set_title("EPS forecast, FY2026E–FY2028E", color=C_INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=8); _style(ax)
    p = OUT / "fv_eps.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


def chart_revenue_build(proj_base: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=130)
    x = np.arange(len(FY))
    nii = [r["nii"] for r in proj_base["rows"]]
    fees = [r["fees"] for r in proj_base["rows"]]
    ppnr = [r["ppnr"] for r in proj_base["rows"]]
    ax.bar(x, nii, 0.5, label="Net interest income", color=C_ACCENT)
    ax.bar(x, fees, 0.5, bottom=nii, label="Fee income", color=C_GOLD)
    ax.plot(x, ppnr, "o-", color=C_INK, lw=1.8, label="PPNR")
    ax.set_xticks(x); ax.set_xticklabels([f"{y}E" for y in FY])
    ax.set_ylabel("$M")
    ax.set_title("Revenue build & pre-provision profit (base case)", color=C_INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=8); _style(ax)
    p = OUT / "fv_revenue.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


def chart_target_bridge(price, intrinsic, pe_t, ptbv_t, blended) -> Path:
    labels = ["Current price", "Intrinsic FV", "Fwd P/E target", "P/TBV target", "Blended 12-mo target"]
    vals = [price, intrinsic, pe_t, ptbv_t, blended]
    cols = [C_INK, C_ACCENT, C_ACCENT, C_ACCENT, C_GOLD]
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=130)
    ax.bar(labels, vals, color=cols)
    ax.axhline(price, color=C_MUTE, ls="--", lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"${v:.0f}", ha="center", fontsize=8, color=C_INK)
    ax.set_ylabel("$/share")
    ax.set_title("Price target bridge (base case)", color=C_INK, fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=8); _style(ax)
    p = OUT / "fv_bridge.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────
def build_html(mkt, projs, targets_all, intrinsic_fv, tbvps, images) -> str:
    price = mkt["price"]; today = date.today().strftime("%d %b %Y")
    base_t = targets_all["base"]["blended"]
    upside = (base_t / price - 1) * 100
    rec = "BUY" if upside > 12 else ("HOLD" if upside > -5 else "REDUCE")
    rec_col = C_GOOD if rec == "BUY" else (C_GOLD if rec == "HOLD" else C_BAD)
    cons_cagr = cagr(CONSUMER_TOTAL["inc"][0], CONSUMER_TOTAL["inc"][2], 2) * 100

    def img(k, alt):
        return (f'<img src="cid:{k}" alt="{alt}" style="width:100%;max-width:720px;'
                f'border:1px solid #e5e9f0;border-radius:8px;margin:8px 0;">') if k in images else ""

    # EPS forecast table
    eps_rows = ""
    for i, yr in enumerate(FY):
        b = projs["base"]["rows"][i]; bl = projs["bull"]["rows"][i]; be = projs["bear"]["rows"][i]
        eps_rows += (f"<tr><td>{yr}E</td>"
                     f"<td style='text-align:right'>${b['revenue']:,.0f}M</td>"
                     f"<td style='text-align:right'>{b['eff']*100:.1f}%</td>"
                     f"<td style='text-align:right'>${b['ppnr']:,.0f}M</td>"
                     f"<td style='text-align:right;color:{C_BAD}'>${be['eps']:.2f}</td>"
                     f"<td style='text-align:right;font-weight:800'>${b['eps']:.2f}</td>"
                     f"<td style='text-align:right;color:{C_GOOD}'>${bl['eps']:.2f}</td></tr>")

    # target table by scenario
    tgt_rows = ""
    for case in ("bear", "base", "bull"):
        t = targets_all[case]; up = (t["blended"] / price - 1) * 100
        tgt_rows += (f"<tr><td style='text-transform:capitalize'>{case}</td>"
                     f"<td style='text-align:right'>${t['ntm_eps']:.2f}</td>"
                     f"<td style='text-align:right'>${t['pe_target']:.0f}</td>"
                     f"<td style='text-align:right'>${t['ptbv_target']:.0f}</td>"
                     f"<td style='text-align:right;font-weight:800'>${t['blended']:.0f}</td>"
                     f"<td style='text-align:right;color:{C_GOOD if up>0 else C_BAD}'>{up:+.1f}%</td></tr>")

    b26 = projs["base"]["rows"][0]; b28 = projs["base"]["rows"][2]
    eps_cagr = cagr(b26["eps"], b28["eps"], 2) * 100

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;background:{C_BG};font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:{C_INK};">
<div style="max-width:760px;margin:0 auto;background:#fff;">
  <div style="background:{C_INK};color:#fff;padding:26px 30px;">
    <div style="font-size:12px;letter-spacing:2px;color:{C_GOLD};font-weight:700;">FULL-FLEDGED VIEW &amp; FORECAST</div>
    <div style="font-size:24px;font-weight:800;margin-top:4px;">Fifth Third Bancorp (NASDAQ: FITB)</div>
    <div style="font-size:13px;color:#c7d0dc;margin-top:6px;">{today} · fundamentals + consumer trends + earnings forecast</div>
  </div>

  <div style="display:flex;flex-wrap:wrap;padding:18px 30px;gap:14px;border-bottom:1px solid #eef1f6;">
    <div style="flex:1;min-width:110px"><div style="font-size:11px;color:{C_MUTE}">PRICE</div><div style="font-size:20px;font-weight:800">${price:,.2f}</div></div>
    <div style="flex:1;min-width:110px"><div style="font-size:11px;color:{C_MUTE}">12-MO TARGET (BASE)</div><div style="font-size:20px;font-weight:800;color:{C_ACCENT}">${base_t:,.0f}</div></div>
    <div style="flex:1;min-width:110px"><div style="font-size:11px;color:{C_MUTE}">UPSIDE</div><div style="font-size:20px;font-weight:800;color:{C_GOOD if upside>0 else C_BAD}">{upside:+.1f}%</div></div>
    <div style="flex:1;min-width:110px"><div style="font-size:11px;color:{C_MUTE}">RATING</div><div style="font-size:20px;font-weight:800;color:{rec_col}">{rec}</div></div>
  </div>

  <div style="padding:22px 30px;">
    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">1 · The full picture in one paragraph</h2>
    <p style="line-height:1.6;font-size:14px">Fifth Third enters 2026 as a newly-enlarged super-regional (post-Comerica, ~$288B assets) with a
    <b>consumer franchise that is both the funding engine and the growth engine</b>: consumer interest income compounded <b>{cons_cagr:.0f}%</b>/yr and that growth was
    ~60% <b>volume-driven</b> (balance sheet growth, led by auto &amp; solar), not just higher rates. On top of that, deposit costs are now <i>falling</i> off the rate peak
    and management is targeting <b>${FUND['guide_2026']['cost_synergy_pretax_m']:.0f}M</b> of cost synergies. Feed those drivers through a standard earnings build and EPS
    recovers from a merger-charge trough toward <b>${b26['eps']:.2f} (2026E) → ${b28['eps']:.2f} (2028E)</b>, a {eps_cagr:.0f}% CAGR. Putting the intrinsic valuation and the
    forward-multiple target together gives a base-case 12-month target of <b>${base_t:,.0f}</b> ({upside:+.1f}% vs ${price:,.2f}).</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">2 · Earnings forecast (driver-based build)</h2>
    <p style="line-height:1.6;font-size:14px">Method: Revenue = NII (2026 guidance ${FORECAST['nii_2026']/1000:.1f}B, grown by driver) + fees (grown at trend CAGR);
    PPNR = revenue − efficiency×revenue (efficiency falls as synergies land); pretax = PPNR − NCO×loans; EPS after tax, preferred dividends and buybacks.</p>
    {img('fv_revenue','revenue build')}
    {img('fv_eps','eps forecast')}
    <table style="width:100%;border-collapse:collapse;font-size:12.5px">
      <tr style="background:{C_BG}"><th style="text-align:left;padding:6px">Year</th><th style="text-align:right;padding:6px">Revenue</th><th style="text-align:right;padding:6px">Efficiency</th><th style="text-align:right;padding:6px">PPNR</th><th style="text-align:right;padding:6px">EPS bear</th><th style="text-align:right;padding:6px">EPS base</th><th style="text-align:right;padding:6px">EPS bull</th></tr>
      {eps_rows}
    </table>
    <p style="font-size:12px;color:{C_MUTE};margin-top:5px">FY2025 actual EPS was ${FUND['fy2025']['eps_diluted']:.2f}; 2026E is diluted by a larger post-merger share count and residual integration costs, then recovers as synergies land.</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">3 · Price target — forecast × valuation, blended</h2>
    <p style="line-height:1.6;font-size:14px">12-month target = <b>forward P/E × NTM EPS</b>, cross-checked with a <b>P/TBV</b> target, then averaged with the
    <b>intrinsic fair value</b> (${intrinsic_fv:,.0f}, from the 6-method blend). This fuses the forecast with the fundamental valuation.</p>
    {img('fv_bridge','target bridge')}
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:{C_BG}"><th style="text-align:left;padding:6px">Scenario</th><th style="text-align:right;padding:6px">NTM EPS</th><th style="text-align:right;padding:6px">P/E target</th><th style="text-align:right;padding:6px">P/TBV target</th><th style="text-align:right;padding:6px">Blended TP</th><th style="text-align:right;padding:6px">Upside</th></tr>
      {tgt_rows}
    </table>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">4 · What has to go right (and wrong)</h2>
    <p style="line-height:1.6;font-size:14px"><b>Bull levers:</b> synergies land on schedule (efficiency to low-50s), auto/home-equity &amp; Southeast households keep compounding,
    Fed cuts widen NIM, buybacks resume. <b>Bear risks:</b> integration slips, consumer credit normalizes faster (NCOs already rising 33→62bps),
    deposit competition re-tightens funding costs, CRE/C&amp;I stress. The forecast bakes these into the efficiency, NCO and growth drivers per scenario.</p>

    <div style="background:{rec_col};color:#fff;padding:18px 22px;border-radius:8px;margin-top:16px">
      <div style="font-size:12px;letter-spacing:1px;opacity:.85">FULL-VIEW CONCLUSION</div>
      <div style="font-size:20px;font-weight:800">{rec} · 12-mo target ${base_t:,.0f} · {upside:+.1f}% vs ${price:,.2f}</div>
      <div style="font-size:12px;opacity:.9;margin-top:6px">Scenario range ${targets_all['bear']['blended']:.0f} (bear) – ${targets_all['bull']['blended']:.0f} (bull). Sell-side consensus: {FUND['analyst']['rating']}, avg ${FUND['analyst']['target_avg']:.0f}.</div>
    </div>

    <p style="font-size:11px;color:{C_MUTE};margin-top:16px;line-height:1.5">Methodology: driver-based pro-forma earnings build; 12-month target = forward P/E × NTM EPS and target P/TBV × TBVPS,
    blended with a 6-method intrinsic valuation; bull/base/bear scenarios. Inputs: FITB FY2025 10-K, Q4-2025 release, 2026 management guidance; live price via yfinance.
    Educational portfolio project — <b>not investment advice</b>. Forecasts are estimates and will differ from actual results.</p>
  </div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--to", type=str, default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("  FITB — FULL-FLEDGED VIEW & FORECAST")
    print("=" * 70)

    mkt = get_live_market_data(TICKER)
    shares0 = mkt["shares_m"]
    FORECAST["shares_2026"] = shares0
    tbvps = round(0.5 * mkt["book_value_ps"] + 0.5 * ASSUMPTIONS["tbvps_estimate"], 2)

    # intrinsic valuation (reuse the 6-method engine)
    peers = get_peer_multiples(["KEY", "HBAN", "RF", "CFG", "MTB", "PNC", "USB"])
    scen = scenario_valuation(mkt, tbvps, mkt["book_value_ps"])
    peer_rel = peer_relative_value(mkt, peers)
    intrinsic = blended_fair_value(scen, peer_rel, FUND["analyst"]["target_avg"])["fair_value"]

    # forecast
    projs = {c: project(c, shares0) for c in ("bear", "base", "bull")}
    targets_all = {c: price_targets(mkt, projs[c], tbvps, intrinsic) for c in ("bear", "base", "bull")}

    print(f"Price ${mkt['price']:.2f} | shares {shares0:.0f}M | intrinsic FV ${intrinsic:.2f}")
    print("\nEPS forecast (base):")
    for r in projs["base"]["rows"]:
        print(f"   {r['year']}E  rev ${r['revenue']:,.0f}M  eff {r['eff']*100:.1f}%  EPS ${r['eps']:.2f}")
    bt = targets_all["base"]
    print(f"\n12-mo target (base): fwdP/E ${bt['pe_target']:.0f} | P/TBV ${bt['ptbv_target']:.0f} | "
          f"blended ${bt['blended']:.0f} ({(bt['blended']/mkt['price']-1)*100:+.1f}%)")
    print(f"Range: bear ${targets_all['bear']['blended']:.0f} — bull ${targets_all['bull']['blended']:.0f}")

    images = {
        "fv_revenue": str(chart_revenue_build(projs["base"])),
        "fv_eps": str(chart_eps(projs)),
        "fv_bridge": str(chart_target_bridge(mkt["price"], intrinsic, bt["pe_target"], bt["ptbv_target"], bt["blended"])),
    }
    html = build_html(mkt, projs, targets_all, intrinsic, tbvps, images)
    report = OUT / "FITB_full_view_forecast.html"
    report.write_text(html, encoding="utf-8")
    print(f"\nReport: {report}")

    if not args.no_email:
        send_email(html, images, to_override=args.to,
                   subject="FITB — Full-Fledged View & Forecast: fundamentals + consumer trends + EPS/price outlook")
    else:
        print("[--no-email] skipped.")
    print("=" * 70, "\nDONE.")


if __name__ == "__main__":
    main()
