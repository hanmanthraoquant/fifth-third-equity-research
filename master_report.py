"""
================================================================================
 FITB — MASTER EQUITY RESEARCH REPORT (Initiating-Coverage style)
================================================================================
The capstone. Consolidates EVERY method in this project into one investment-bank-
grade note and adds three more advanced techniques, then produces a single
weighted valuation and recommendation.

VALUATION METHODS INCLUDED (14):
  Intrinsic          1) Dividend Discount Model (Gordon)
                     2) Justified P/TBV = (ROTCE-g)/(COE-g)
                     3) Residual Income (Excess Returns)
                     4) Justified P/E
  Relative           5) Peer P/E   6) Peer P/TBV
                     7) P/TBV-vs-ROTCE peer REGRESSION (warranted multiple)   [NEW]
  Structural         8) Sum-of-the-Parts (SOTP)
  Forecast-driven    9) Forward P/E × NTM EPS   10) Target P/TBV × TBVPS
  Market check      11) Analyst consensus
  Expectations      12) Reverse DCF — implied growth priced in                [NEW]
  M&A               13) Comerica accretion/dilution & TBV earnback
  Yield             14) Total capital-return (dividend + buyback) yield        [NEW]

Also carries the consumer-banking trend engine and a bull/base/bear synthesis.
Live price via yfinance; fundamentals from FITB FY2025 10-K + Q4-2025 + guidance.
$ in millions. NOT investment advice.

Run:  python master_report.py            # build + email
      python master_report.py --no-email
================================================================================
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fitb_analysis import (
    get_live_market_data, get_peer_multiples, cost_of_equity,
    scenario_valuation, peer_relative_value, blended_fair_value,
    ddm_gordon, justified_ptbv, residual_income, justified_pe,
    FUND, ASSUMPTIONS, send_email, OUT,
    C_INK, C_ACCENT, C_GOLD, C_GOOD, C_BAD, C_MUTE, C_BG,
)
from full_view_forecast import project, price_targets, FORECAST
from ib_deepdive import trading_comps, accretion_dilution, sotp, sensitivity, PEERS
from trend_analysis import CONSUMER_TOTAL, cagr

TICKER = "FITB"

# Weights for the master blended target (sum to 1.0). Weighted toward the
# methods most appropriate for a bank (tangible-book / residual income /
# forward multiples), lighter on DDM (understates retention) and single checks.
WEIGHTS = {
    "DDM (Gordon)": 0.04, "Justified P/TBV": 0.11, "Residual Income": 0.13,
    "Justified P/E": 0.04, "Peer P/E": 0.08, "Peer P/TBV": 0.08,
    "P/TBV~ROTCE regression": 0.10, "Sum-of-the-Parts": 0.07,
    "Forward P/E target": 0.15, "Target P/TBV": 0.10,
    "Analyst consensus": 0.05, "Reverse-DCF (fair @ base g)": 0.05,
}


# ── NEW METHOD 7: P/TBV vs ROTCE peer regression (warranted multiple) ─────────
def ptbv_rotce_regression(peers_df, mkt, tbvps, rotce):
    """Fit P/TBV = a + b*ROTCE across peers; FITB warranted P/TBV -> value."""
    x = peers_df["roe"].astype(float).values
    y = peers_df["pb"].astype(float).values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return {"ok": False}
    b, a = np.polyfit(x[mask], y[mask], 1)          # slope, intercept
    warranted_pb = a + b * rotce
    return {"ok": True, "slope": b, "intercept": a,
            "warranted_pb": warranted_pb, "value": max(warranted_pb, 0) * tbvps,
            "x": x[mask], "y": y[mask], "rotce": rotce}


# ── NEW METHOD 12: Reverse DCF — implied growth priced into the stock ──────────
def reverse_dcf(price, tbvps, rotce, coe):
    """
    From the bank identity  P/TBV = (ROTCE - g)/(COE - g), solve for the g the
    market is implying at the current price, and report the fair value at our
    base-case g for comparison.
    """
    cur_ptbv = price / tbvps
    # g = (ROTCE - P/TBV*COE) / (1 - P/TBV)
    denom = 1 - cur_ptbv
    implied_g = (rotce - cur_ptbv * coe) / denom if abs(denom) > 1e-6 else float("nan")
    base_g = ASSUMPTIONS["growth_g"]["base"]
    fair_at_base = justified_ptbv(rotce, base_g, coe, tbvps)
    return {"cur_ptbv": cur_ptbv, "implied_g": implied_g, "base_g": base_g,
            "fair_at_base": fair_at_base}


# ── NEW METHOD 14: total capital-return (shareholder) yield ───────────────────
def capital_return_yield(mkt):
    div_yield = (mkt["dividend_rate"] / mkt["price"]) if mkt["price"] else 0.0
    buyback_yield = FORECAST["buyback_shrink"]
    earnings_yield = (1 / mkt["pe_ttm"]) if mkt["pe_ttm"] else 0.0
    return {"div_yield": div_yield, "buyback_yield": buyback_yield,
            "total_yield": div_yield + buyback_yield, "earnings_yield": earnings_yield}


# ─────────────────────────────────────────────────────────────────────────────
def assemble(mkt):
    tbvps = round(0.5 * mkt["book_value_ps"] + 0.5 * ASSUMPTIONS["tbvps_estimate"], 2)
    bvps = mkt["book_value_ps"]
    coe = cost_of_equity(mkt["beta"])
    rotce = FUND["fy2025"]["rotce"]

    peers_df = get_peer_multiples(PEERS)
    scen = scenario_valuation(mkt, tbvps, bvps)
    peer_rel = peer_relative_value(mkt, peers_df)
    intrinsic_blend = blended_fair_value(scen, peer_rel, FUND["analyst"]["target_avg"])
    base = scen[scen["scenario"] == "base"].iloc[0]

    comps = trading_comps(mkt, peers_df)
    ad = accretion_dilution(mkt)
    s = sotp(mkt)
    sens = sensitivity(mkt)
    projs = {c: project(c, mkt["shares_m"]) for c in ("bear", "base", "bull")}
    tgt = price_targets(mkt, projs["base"], tbvps, intrinsic_blend["fair_value"])

    reg = ptbv_rotce_regression(peers_df, mkt, tbvps, rotce)
    rdcf = reverse_dcf(mkt["price"], tbvps, rotce, coe)
    cry = capital_return_yield(mkt)

    methods = {
        "DDM (Gordon)": float(base["DDM"]),
        "Justified P/TBV": float(base["Justified P/TBV"]),
        "Residual Income": float(base["Residual Income"]),
        "Justified P/E": float(base["Justified P/E"]),
        "Peer P/E": comps["implied_pe"],
        "Peer P/TBV": comps["implied_pb"],
        "P/TBV~ROTCE regression": reg["value"] if reg.get("ok") else float("nan"),
        "Sum-of-the-Parts": s["per_share"],
        "Forward P/E target": tgt["pe_target"],
        "Target P/TBV": tgt["ptbv_target"],
        "Analyst consensus": FUND["analyst"]["target_avg"],
        "Reverse-DCF (fair @ base g)": rdcf["fair_at_base"],
    }
    # weighted blended target (skip NaNs, renormalize weights)
    num = den = 0.0
    for k, v in methods.items():
        if v and not np.isnan(v) and v > 0:
            w = WEIGHTS.get(k, 0.0); num += w * v; den += w
    weighted = num / den if den else float("nan")

    return dict(tbvps=tbvps, coe=coe, rotce=rotce, peers_df=peers_df, comps=comps,
                ad=ad, sotp=s, sens=sens, projs=projs, tgt=tgt, reg=reg, rdcf=rdcf,
                cry=cry, methods=methods, weighted=weighted,
                intrinsic=intrinsic_blend["fair_value"])


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25); ax.tick_params(colors=C_MUTE)


def chart_master_field(methods, price, weighted) -> str:
    items = [(k, v) for k, v in methods.items() if v and not np.isnan(v) and v > 0]
    names = [k for k, _ in items]; vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=130)
    cols = [C_GOOD if v >= price else C_BAD for v in vals]
    ax.barh(names[::-1], vals[::-1], color=cols[::-1], alpha=0.9)
    ax.axvline(price, color=C_INK, ls="--", lw=1.5, label=f"Price ${price:.0f}")
    ax.axvline(weighted, color=C_GOLD, lw=2.2, label=f"Weighted target ${weighted:.0f}")
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.4, i, f"${v:.0f}", va="center", fontsize=8, color=C_INK)
    ax.set_title("Master valuation — 12 methods vs price", color=C_INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=9); _style(ax); ax.grid(axis="x", alpha=0.2); ax.grid(axis="y", visible=False)
    p = OUT / "ms_field.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return str(p)


def chart_regression(reg, mkt) -> str | None:
    if not reg.get("ok"):
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=130)
    x, y = reg["x"] * 100, reg["y"]
    ax.scatter(x, y, color=C_ACCENT, s=45, label="Peers")
    xs = np.linspace(min(x.min(), reg["rotce"] * 100) - 1, x.max() + 1, 50)
    ax.plot(xs, reg["intercept"] + reg["slope"] * (xs / 100), color=C_MUTE, lw=1.4,
            label="Fitted warranted line")
    fx = reg["rotce"] * 100; fy = reg["warranted_pb"]
    ax.scatter([fx], [mkt["pb"]], color=C_BAD, s=80, marker="*", label=f"FITB actual ({mkt['pb']:.2f}x)")
    ax.scatter([fx], [fy], color=C_GOLD, s=80, marker="D", label=f"FITB warranted ({fy:.2f}x)")
    ax.set_xlabel("ROTCE / ROE (%)"); ax.set_ylabel("P/TBV (x)")
    ax.set_title("Warranted multiple: P/TBV vs ROTCE regression", color=C_INK, fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8); _style(ax)
    p = OUT / "ms_regression.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return str(p)


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────
def build_html(mkt, A, images):
    price = mkt["price"]; today = date.today().strftime("%d %b %Y")
    weighted = A["weighted"]; upside = (weighted / price - 1) * 100
    rec = "BUY" if upside > 12 else ("HOLD" if upside > -5 else "REDUCE")
    rec_col = C_GOOD if rec == "BUY" else (C_GOLD if rec == "HOLD" else C_BAD)
    cons_cagr = cagr(CONSUMER_TOTAL["inc"][0], CONSUMER_TOTAL["inc"][2], 2) * 100
    b = A["projs"]["base"]["rows"]

    def img(k, alt):
        return (f'<img src="cid:{k}" alt="{alt}" style="width:100%;max-width:720px;'
                f'border:1px solid #e5e9f0;border-radius:8px;margin:8px 0;">') if k in images else ""

    method_rows = ""
    for k, v in A["methods"].items():
        if v and not np.isnan(v) and v > 0:
            up = (v / price - 1) * 100
            method_rows += (f"<tr><td>{k}</td><td style='text-align:right'>${v:,.0f}</td>"
                            f"<td style='text-align:right;color:{C_GOOD if up>=0 else C_BAD}'>{up:+.1f}%</td>"
                            f"<td style='text-align:right'>{WEIGHTS.get(k,0)*100:.0f}%</td></tr>")
    method_rows += (f"<tr style='border-top:2px solid {C_INK};font-weight:800'><td>Weighted target</td>"
                    f"<td style='text-align:right'>${weighted:,.0f}</td>"
                    f"<td style='text-align:right;color:{C_GOOD if upside>=0 else C_BAD}'>{upside:+.1f}%</td><td></td></tr>")

    reg = A["reg"]; rdcf = A["rdcf"]; cry = A["cry"]; ad = A["ad"]
    reg_txt = (f"Peers price roughly {reg['slope']:.1f}x of P/TBV per 1.0 of ROTCE. At FITB's {A['rotce']*100:.1f}% ROTCE the regression "
               f"warrants ~<b>{reg['warranted_pb']:.2f}x</b> P/TBV (≈ ${reg['value']:.0f}); FITB actually trades at {mkt['pb']:.2f}x."
               ) if reg.get("ok") else "Insufficient peer data for regression."

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;background:{C_BG};font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:{C_INK};">
<div style="max-width:760px;margin:0 auto;background:#fff;">
  <div style="background:{C_INK};color:#fff;padding:28px 30px;">
    <div style="font-size:12px;letter-spacing:2px;color:{C_GOLD};font-weight:700;">MASTER EQUITY RESEARCH · INITIATING COVERAGE</div>
    <div style="font-size:26px;font-weight:800;margin-top:4px;">Fifth Third Bancorp <span style="color:{C_GOLD}">(NASDAQ: FITB)</span></div>
    <div style="font-size:13px;color:#c7d0dc;margin-top:6px;">{today} · 14 valuation methods · full fundamentals + forecast</div>
  </div>

  <div style="display:flex;flex-wrap:wrap;padding:18px 30px;gap:14px;border-bottom:1px solid #eef1f6;">
    <div style="flex:1;min-width:110px"><div style="font-size:11px;color:{C_MUTE}">PRICE</div><div style="font-size:21px;font-weight:800">${price:,.2f}</div></div>
    <div style="flex:1;min-width:110px"><div style="font-size:11px;color:{C_MUTE}">WEIGHTED TARGET</div><div style="font-size:21px;font-weight:800;color:{C_ACCENT}">${weighted:,.0f}</div></div>
    <div style="flex:1;min-width:110px"><div style="font-size:11px;color:{C_MUTE}">UPSIDE</div><div style="font-size:21px;font-weight:800;color:{C_GOOD if upside>0 else C_BAD}">{upside:+.1f}%</div></div>
    <div style="flex:1;min-width:110px"><div style="font-size:11px;color:{C_MUTE}">RATING</div><div style="font-size:21px;font-weight:800;color:{rec_col}">{rec}</div></div>
  </div>

  <div style="padding:22px 30px;">
    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">1 · Executive summary</h2>
    <p style="line-height:1.6;font-size:14px">Fifth Third is a newly-enlarged super-regional (post-Comerica, ~$288B assets, 9th-largest US bank) built on a
    <b>consumer franchise that funds the bank and drives growth</b> (consumer interest income {cons_cagr:.0f}% CAGR, ~60% volume-driven; CSBB = ~⅔ of NII).
    We value it with <b>14 independent methods</b> spanning intrinsic, relative, structural, forecast-driven, expectations and M&amp;A lenses. They converge on a
    weighted 12-month target of <b>${weighted:,.0f}</b> ({upside:+.1f}% vs ${price:,.2f}) — a <b>{rec}</b>. EPS recovers from a merger-charge trough to
    <b>${b[0]['eps']:.2f} (26E) → ${b[2]['eps']:.2f} (28E)</b>; the Comerica deal is <b>{ad['day1_accr']*100:+.1f}% day-1 but +{ad['rows'][1]['accretion']*100:.0f}% accretive by 27E</b>
    with a ~{ad['earnback_yrs']:.1f}-yr tangible-book earnback.</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">2 · Master valuation — all methods</h2>
    {img('ms_field','master field')}
    <table style="width:100%;border-collapse:collapse;font-size:12.5px">
      <tr style="background:{C_BG}"><th style="text-align:left;padding:6px">Method</th><th style="text-align:right;padding:6px">Value</th><th style="text-align:right;padding:6px">vs Price</th><th style="text-align:right;padding:6px">Weight</th></tr>
      {method_rows}
    </table>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">3 · Warranted-multiple regression (P/TBV ~ ROTCE)</h2>
    <p style="line-height:1.6;font-size:14px">The classic bank cross-sectional test: better-returning banks earn a higher premium to tangible book. {reg_txt}</p>
    {img('ms_regression','regression')}

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">4 · Reverse DCF — what's priced in</h2>
    <p style="line-height:1.6;font-size:14px">Inverting the bank identity at today's price ({rdcf['cur_ptbv']:.2f}× tangible book, {A['rotce']*100:.1f}% ROTCE, {A['coe']*100:.1f}% cost of equity)
    implies the market is pricing <b>~{rdcf['implied_g']*100:.1f}% perpetual growth</b>. Our base case assumes {rdcf['base_g']*100:.1f}% — so at the current price the stock
    {'is discounting slightly MORE growth than our base case (modestly full)' if rdcf['implied_g']>rdcf['base_g'] else 'is discounting LESS growth than our base case (modestly cheap)'}.
    Fair value at our base-case growth would be ~<b>${rdcf['fair_at_base']:.0f}</b>.</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">5 · Capital-return yield</h2>
    <p style="line-height:1.6;font-size:14px">Total shareholder yield ≈ <b>{cry['total_yield']*100:.1f}%</b> (dividend {cry['div_yield']*100:.1f}% + buyback ~{cry['buyback_yield']*100:.1f}%),
    against an earnings yield of {cry['earnings_yield']*100:.1f}%. A healthy, well-covered capital return supports the shares while synergies build.</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">6 · Methods catalog (what's under the hood)</h2>
    <p style="line-height:1.6;font-size:13px;color:{C_MUTE}">Intrinsic: DDM (Gordon), justified P/TBV, residual income, justified P/E. Relative: peer P/E, peer P/TBV, P/TBV~ROTCE regression.
    Structural: sum-of-the-parts. Forecast: driver-based earnings build → forward P/E &amp; target P/TBV. Expectations: reverse DCF. M&amp;A: accretion/dilution &amp; TBV earnback.
    Frameworks: CAMELS, DuPont, horizontal/vertical/common-size, rate/volume variance, scenario &amp; two-way sensitivity. CAPM cost of equity.</p>

    <div style="background:{rec_col};color:#fff;padding:18px 22px;border-radius:8px;margin-top:16px">
      <div style="font-size:12px;letter-spacing:1px;opacity:.85">MASTER CONCLUSION</div>
      <div style="font-size:20px;font-weight:800">{rec} · Weighted target ${weighted:,.0f} · {upside:+.1f}% vs ${price:,.2f}</div>
      <div style="font-size:12px;opacity:.9;margin-top:6px">14 methods; the re-rating case rests on Comerica synergy delivery lifting ROTCE and EPS — exactly what the reverse-DCF and sensitivity frame.</div>
    </div>

    <p style="font-size:11px;color:{C_MUTE};margin-top:16px;line-height:1.5">Consolidates fitb_analysis / consumer_deepdive / trend_analysis / full_view_forecast / ib_deepdive. Live price via yfinance;
    fundamentals from FITB FY2025 10-K, Q4-2025 release and 2026 guidance. Several M&amp;A/SOTP/forecast inputs are documented estimates. Educational portfolio
    project — <b>not investment advice</b>.</p>
  </div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--to", type=str, default=None)
    args = ap.parse_args()

    print("=" * 70); print("  FITB — MASTER EQUITY RESEARCH REPORT (14 methods)"); print("=" * 70)
    mkt = get_live_market_data(TICKER)
    A = assemble(mkt)

    print(f"Price ${mkt['price']:.2f} | weighted target ${A['weighted']:.0f} "
          f"({(A['weighted']/mkt['price']-1)*100:+.1f}%)")
    print("\nValuation by method:")
    for k, v in A["methods"].items():
        if v and not np.isnan(v):
            print(f"   {k:28s} ${v:8.2f}  ({(v/mkt['price']-1)*100:+.1f}%)")
    print(f"\nReverse-DCF implied growth: {A['rdcf']['implied_g']*100:.1f}% (base {A['rdcf']['base_g']*100:.1f}%)")
    print(f"Capital-return yield: {A['cry']['total_yield']*100:.1f}%")

    images = {"ms_field": chart_master_field(A["methods"], mkt["price"], A["weighted"])}
    rc = chart_regression(A["reg"], mkt)
    if rc:
        images["ms_regression"] = rc

    html = build_html(mkt, A, images)
    report = OUT / "FITB_MASTER_report.html"
    report.write_text(html, encoding="utf-8")
    print(f"\nReport: {report}")

    if not args.no_email:
        send_email(html, images, to_override=args.to,
                   subject="FITB — MASTER Equity Research Report: 14 valuation methods, full fundamentals & forecast")
    else:
        print("[--no-email] skipped.")
    print("=" * 70, "\nDONE.")


if __name__ == "__main__":
    main()
