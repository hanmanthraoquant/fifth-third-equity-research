"""
================================================================================
 FITB — Investment-Bank Deep-Dive Techniques
================================================================================
Adds the four techniques an equity-research / M&A desk leans on, on top of the
intrinsic valuation and earnings forecast already in this project:

  1. COMPARABLE COMPANIES ANALYSIS ("trading comps") — peer P/E, P/TBV, dividend
     yield and ROTCE; apply peer medians to FITB for an implied value.
  2. M&A ACCRETION / DILUTION + TBV EARNBACK — the classic all-stock-deal build
     for the Comerica acquisition: standalone vs. pro-forma EPS, accretion %,
     tangible-book dilution and the earnback period (crossover method).
  3. SUM-OF-THE-PARTS (SOTP) — value each segment (Commercial, Consumer & SB,
     Wealth) on segment earnings × a segment-appropriate multiple.
  4. TWO-WAY SENSITIVITY ("data table") — target price across exit P/E × EPS.

Inputs: FITB FY2025 10-K + Q4-2025 segment tables + Comerica deal terms; live
price via yfinance. All M&A/SOTP inputs are documented and clearly labeled as
estimates where the company does not disclose them. $ in millions. NOT advice.

Run:  python ib_deepdive.py            # build + email
      python ib_deepdive.py --no-email
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
    get_live_market_data, get_peer_multiples, send_email, OUT, FUND, ASSUMPTIONS,
    C_INK, C_ACCENT, C_GOLD, C_GOOD, C_BAD, C_MUTE, C_BG,
)
from full_view_forecast import project, FORECAST

TICKER = "FITB"
PEERS = ["KEY", "HBAN", "RF", "CFG", "MTB", "PNC", "USB"]

# ── Segment earnings (annualized from Q4-2025 pretax, FTE, $M) ─────────────────
SEGMENTS = {  # name: (annualized pretax, P/E multiple, note)
    "Consumer & Small Business": (608 * 4, 12.0, "retail/deposit franchise"),
    "Commercial Banking":        (445 * 4, 11.0, "C&I + commercial payments"),
    "Wealth & Asset Mgmt":       (66 * 4, 18.0, "fee-based, capital-light"),
}
CORP_OTHER_NORMALIZED = -400.0   # ongoing corporate/other drag, ex one-off merger charges
CORP_OTHER_MULTIPLE = 6.0
TAX = 0.22

# ── Comerica M&A inputs (deal terms disclosed; CMA financials estimated) ───────
DEAL = {
    "exchange_ratio": 1.8663, "value_close_b": 12.7,
    "fitb_own": 0.73, "cma_own": 0.27,
    "synergy_pretax": 850.0,                 # run-rate, by Q4-2026
    "synergy_phase": {2026: 0.50, 2027: 0.90, 2028: 1.0},
    "cma_net_income_est": 700.0,             # CMA standalone net income (est, FY)
    "cma_tbv_b_est": 5.6,                    # CMA tangible book (est)
    "goodwill_created_b_est": 7.1,           # ~deal value − CMA tangible equity
}
# FITB standalone (pre-deal) anchor
FITB_STANDALONE = {"ni_common": 2400.0, "eps": 3.53, "tbvps_pre": 26.0}


# ─────────────────────────────────────────────────────────────────────────────
# 1) COMPARABLE COMPANIES ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def trading_comps(mkt, peers_df):
    def med(col):
        s = peers_df[col].astype(float)
        return float(np.nanmedian(s)) if s.notna().any() else float("nan")
    med_pe, med_pb, med_dy, med_roe = med("pe"), med("pb"), med("div_yield"), med("roe")
    eps = ASSUMPTIONS["normalized_eps"]["base"]
    implied_pe = med_pe * eps if not np.isnan(med_pe) else float("nan")
    implied_pb = med_pb * mkt["book_value_ps"] if not np.isnan(med_pb) else float("nan")
    return {"med_pe": med_pe, "med_pb": med_pb, "med_dy": med_dy, "med_roe": med_roe,
            "implied_pe": implied_pe, "implied_pb": implied_pb,
            "fitb_pe": mkt["pe_ttm"], "fitb_pb": mkt["pb"]}


# ─────────────────────────────────────────────────────────────────────────────
# 2) ACCRETION / DILUTION + TBV EARNBACK
# ─────────────────────────────────────────────────────────────────────────────
def accretion_dilution(mkt):
    proforma_shares = mkt["shares_m"]
    legacy_shares = proforma_shares * DEAL["fitb_own"]
    new_shares = proforma_shares - legacy_shares
    standalone_eps = FITB_STANDALONE["eps"]
    fitb_ni = legacy_shares * standalone_eps
    cma_ni = DEAL["cma_net_income_est"]
    syn_after_tax_full = DEAL["synergy_pretax"] * (1 - TAX)

    rows = []
    for yr in (2026, 2027, 2028):
        syn = syn_after_tax_full * DEAL["synergy_phase"][yr]
        pf_ni = fitb_ni + cma_ni + syn
        pf_eps = pf_ni / proforma_shares
        accretion = pf_eps / standalone_eps - 1
        rows.append({"year": yr, "syn": syn, "pf_eps": pf_eps, "accretion": accretion})

    # No-synergy (day-1) accretion/dilution
    day1_eps = (fitb_ni + cma_ni) / proforma_shares
    day1_accr = day1_eps / standalone_eps - 1

    # TBV earnback (crossover method, estimated):
    # goodwill dilutes acquirer TBVPS; EPS accretion rebuilds it. earnback = dilution / annual accretion $.
    tbv_dilution_pct = 0.06  # ~6% initial TBVPS dilution (typical all-stock, estimated)
    tbvps_pre = FITB_STANDALONE["tbvps_pre"]
    dilution_per_share = tbvps_pre * tbv_dilution_pct
    accretion_dollars = rows[1]["pf_eps"] - standalone_eps  # steady-state annual EPS uplift
    earnback_yrs = dilution_per_share / accretion_dollars if accretion_dollars > 0 else float("nan")

    return {"legacy_shares": legacy_shares, "new_shares": new_shares,
            "proforma_shares": proforma_shares, "standalone_eps": standalone_eps,
            "day1_eps": day1_eps, "day1_accr": day1_accr, "rows": rows,
            "tbv_dilution_pct": tbv_dilution_pct, "earnback_yrs": earnback_yrs}


# ─────────────────────────────────────────────────────────────────────────────
# 3) SUM-OF-THE-PARTS
# ─────────────────────────────────────────────────────────────────────────────
def sotp(mkt):
    parts = []
    total = 0.0
    for name, (pretax, mult, note) in SEGMENTS.items():
        after_tax = pretax * (1 - TAX)
        val = after_tax * mult
        total += val
        parts.append({"name": name, "after_tax": after_tax, "mult": mult, "val": val, "note": note})
    corp_val = CORP_OTHER_NORMALIZED * (1 - TAX) * CORP_OTHER_MULTIPLE
    total += corp_val
    parts.append({"name": "Corporate/Other (norm.)", "after_tax": CORP_OTHER_NORMALIZED * (1 - TAX),
                  "mult": CORP_OTHER_MULTIPLE, "val": corp_val, "note": "unallocated / funding"})
    per_share = total / mkt["shares_m"]
    return {"parts": parts, "equity_value": total, "per_share": per_share}


# ─────────────────────────────────────────────────────────────────────────────
# 4) TWO-WAY SENSITIVITY (data table): exit P/E  x  EPS scenario
# ─────────────────────────────────────────────────────────────────────────────
def sensitivity(mkt):
    pes = [11.0, 12.0, 13.0, 14.0]
    eps_by_case = {c: project(c, mkt["shares_m"])["rows"][1]["eps"] for c in ("bear", "base", "bull")}
    grid = {}
    for case, eps in eps_by_case.items():
        grid[case] = {pe: pe * eps for pe in pes}
    return {"pes": pes, "eps": eps_by_case, "grid": grid}


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25); ax.tick_params(colors=C_MUTE)


def chart_comps(mkt, peers_df, comps) -> Path:
    df = peers_df.copy()
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=130)
    tickers = list(df["ticker"]) + ["FITB"]
    pe = list(df["pe"].astype(float)) + [mkt["pe_ttm"]]
    x = np.arange(len(tickers))
    cols = [C_ACCENT] * len(df) + [C_BAD]
    ax.bar(x, pe, color=cols)
    ax.axhline(comps["med_pe"], color=C_GOLD, ls="--", lw=1.4, label=f"Peer median {comps['med_pe']:.1f}x")
    ax.set_xticks(x); ax.set_xticklabels(tickers, fontsize=8)
    ax.set_ylabel("P/E (ttm)"); ax.set_title("Trading comps — P/E vs peers", color=C_INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=8); _style(ax)
    p = OUT / "ib_comps.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


def chart_accretion(ad) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=130)
    cats = ["Standalone", "Day-1\n(no syn.)"] + [f"{r['year']}E" for r in ad["rows"]]
    eps = [ad["standalone_eps"], ad["day1_eps"]] + [r["pf_eps"] for r in ad["rows"]]
    cols = [C_MUTE, C_BAD] + [C_GOOD if e >= ad["standalone_eps"] else C_BAD for e in eps[2:]]
    ax.bar(cats, eps, color=cols)
    ax.axhline(ad["standalone_eps"], color=C_INK, ls="--", lw=1, label=f"Standalone ${ad['standalone_eps']:.2f}")
    for i, e in enumerate(eps):
        ax.text(i, e + 0.05, f"${e:.2f}", ha="center", fontsize=8, color=C_INK)
    ax.set_ylabel("Pro-forma EPS ($)")
    ax.set_title("Comerica accretion/dilution — dilutive day 1, accretive on synergies", color=C_INK, fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8); _style(ax)
    p = OUT / "ib_accretion.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


def chart_sotp(s, price) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=130)
    names = [p["name"] for p in s["parts"]]
    vals = [p["val"] / 1000 for p in s["parts"]]
    cols = [C_ACCENT if v >= 0 else C_BAD for v in vals]
    ax.bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.3 if v >= 0 else -0.6), f"${v:.1f}B", ha="center", fontsize=8, color=C_INK)
    ax.set_ylabel("Equity value ($B)")
    ax.set_title(f"Sum-of-the-parts → ${s['per_share']:.0f}/sh (vs ${price:.0f} price)", color=C_INK, fontweight="bold", fontsize=10)
    plt.xticks(rotation=15, ha="right", fontsize=7.5); _style(ax)
    p = OUT / "ib_sotp.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); return p


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────
def build_html(mkt, comps, ad, s, sens, peers_df, images):
    price = mkt["price"]; today = date.today().strftime("%d %b %Y")

    def img(k, alt):
        return (f'<img src="cid:{k}" alt="{alt}" style="width:100%;max-width:720px;'
                f'border:1px solid #e5e9f0;border-radius:8px;margin:8px 0;">') if k in images else ""

    def _cell(v, fmt):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "<td style='text-align:right'>n/a</td>"
            return f"<td style='text-align:right'>{fmt(v)}</td>"
        except Exception:
            return "<td style='text-align:right'>n/a</td>"

    comp_rows = ""
    for _, r in peers_df.iterrows():
        comp_rows += (
            f"<tr><td>{r['ticker']}</td>"
            + _cell(r["pe"], lambda x: f"{x:.1f}x")
            + _cell(r["pb"], lambda x: f"{x:.2f}x")
            + _cell(r["div_yield"], lambda x: f"{x*100:.1f}%")
            + _cell(r["roe"], lambda x: f"{x*100:.1f}%")
            + "</tr>")
    comp_rows += (f"<tr style='border-top:2px solid {C_INK};font-weight:800'><td>Peer median</td>"
                  f"<td style='text-align:right'>{comps['med_pe']:.1f}x</td>"
                  f"<td style='text-align:right'>{comps['med_pb']:.2f}x</td>"
                  f"<td style='text-align:right'>{comps['med_dy']*100:.1f}%</td>"
                  f"<td style='text-align:right'>{comps['med_roe']*100:.1f}%</td></tr>"
                  f"<tr style='font-weight:800;color:{C_BAD}'><td>FITB</td>"
                  f"<td style='text-align:right'>{comps['fitb_pe']:.1f}x</td>"
                  f"<td style='text-align:right'>{comps['fitb_pb']:.2f}x</td><td></td><td></td></tr>")

    ad_rows = "".join(
        f"<tr><td>{r['year']}E</td><td style='text-align:right'>${r['syn']:.0f}M</td>"
        f"<td style='text-align:right'>${r['pf_eps']:.2f}</td>"
        f"<td style='text-align:right;color:{C_GOOD if r['accretion']>=0 else C_BAD};font-weight:700'>{r['accretion']*100:+.1f}%</td></tr>"
        for r in ad["rows"])

    sotp_rows = "".join(
        f"<tr><td>{p['name']}</td><td style='text-align:right'>${p['after_tax']:,.0f}M</td>"
        f"<td style='text-align:right'>{p['mult']:.0f}x</td>"
        f"<td style='text-align:right;font-weight:700'>${p['val']/1000:.1f}B</td></tr>"
        for p in s["parts"])

    pes = sens["pes"]
    head = "".join(f"<th style='text-align:right;padding:5px'>{pe:.0f}x</th>" for pe in pes)
    sens_rows = ""
    for case in ("bear", "base", "bull"):
        cells = "".join(
            f"<td style='text-align:right;padding:5px;{'background:#fff7e6;font-weight:800;' if (case=='base' and abs(pe-12.0)<1e-9) else ''}'>${sens['grid'][case][pe]:.0f}</td>"
            for pe in pes)
        sens_rows += (f"<tr><td style='text-transform:capitalize'>{case} (EPS ${sens['eps'][case]:.2f})</td>{cells}</tr>")

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;background:{C_BG};font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:{C_INK};">
<div style="max-width:760px;margin:0 auto;background:#fff;">
  <div style="background:{C_INK};color:#fff;padding:26px 30px;">
    <div style="font-size:12px;letter-spacing:2px;color:{C_GOLD};font-weight:700;">INVESTMENT-BANK DEEP-DIVE TECHNIQUES</div>
    <div style="font-size:24px;font-weight:800;margin-top:4px;">Fifth Third Bancorp (NASDAQ: FITB)</div>
    <div style="font-size:13px;color:#c7d0dc;margin-top:6px;">{today} · comps · accretion/dilution · SOTP · sensitivity</div>
  </div>
  <div style="padding:22px 30px;">
    <div style="background:{C_BG};border-radius:8px;padding:12px 16px;font-size:12.5px;color:{C_MUTE};line-height:1.55;">
      <b style="color:{C_INK}">The IB toolkit, applied.</b> Four techniques an equity-research / M&amp;A desk runs on every name:
      <b>trading comps</b>, <b>accretion/dilution &amp; TBV earnback</b>, <b>sum-of-the-parts</b>, and a <b>two-way sensitivity data table</b>.
      Together they triangulate the ~$58 share price from four independent angles.
    </div>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">1 · Comparable companies (trading comps)</h2>
    {img('ib_comps','comps')}
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:{C_BG}"><th style="text-align:left;padding:6px">Ticker</th><th style="text-align:right;padding:6px">P/E</th><th style="text-align:right;padding:6px">P/TBV</th><th style="text-align:right;padding:6px">Div yld</th><th style="text-align:right;padding:6px">ROE</th></tr>
      {comp_rows}
    </table>
    <p style="line-height:1.6;font-size:13px;color:{C_MUTE};margin-top:6px">Applying the peer <b>median P/E ({comps['med_pe']:.1f}x)</b> to FITB's normalized EPS implies ~<b>${comps['implied_pe']:.0f}</b>;
    the peer <b>median P/TBV ({comps['med_pb']:.2f}x)</b> implies ~<b>${comps['implied_pb']:.0f}</b>. FITB screens roughly in line with the group.</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">2 · Comerica accretion / dilution &amp; TBV earnback</h2>
    <p style="line-height:1.6;font-size:14px">The signature all-stock-deal build. FITB issued ~<b>{ad['new_shares']:.0f}M</b> new shares ({DEAL['cma_own']*100:.0f}% of the pro-forma
    <b>{ad['proforma_shares']:.0f}M</b>). Day-1, before synergies, the deal is <b style='color:{C_BAD}'>{ad['day1_accr']*100:+.1f}%</b> to EPS (dilutive) — normal for a
    premium all-stock deal — but as the <b>${DEAL['synergy_pretax']:.0f}M</b> cost synergies phase in it turns solidly accretive:</p>
    {img('ib_accretion','accretion')}
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:{C_BG}"><th style="text-align:left;padding:6px">Year</th><th style="text-align:right;padding:6px">After-tax synergies</th><th style="text-align:right;padding:6px">Pro-forma EPS</th><th style="text-align:right;padding:6px">Accretion vs standalone</th></tr>
      <tr><td>Standalone</td><td style="text-align:right">—</td><td style="text-align:right">${ad['standalone_eps']:.2f}</td><td style="text-align:right">—</td></tr>
      <tr><td>Day-1 (no synergies)</td><td style="text-align:right">$0M</td><td style="text-align:right">${ad['day1_eps']:.2f}</td><td style="text-align:right;color:{C_BAD}">{ad['day1_accr']*100:+.1f}%</td></tr>
      {ad_rows}
    </table>
    <p style="line-height:1.6;font-size:13px;color:{C_MUTE};margin-top:6px">Estimated tangible-book dilution ~<b>{ad['tbv_dilution_pct']*100:.0f}%</b> (goodwill created ≈ ${DEAL['goodwill_created_b_est']:.1f}B),
    with a <b>~{ad['earnback_yrs']:.1f}-year TBV earnback</b> (crossover method). A &lt;3-yr earnback is generally considered attractive by the Street.</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">3 · Sum-of-the-parts (SOTP)</h2>
    <p style="line-height:1.6;font-size:14px">Value each segment on its own earnings and an appropriate multiple (fee-heavy Wealth gets a premium; spread businesses a bank multiple).</p>
    {img('ib_sotp','sotp')}
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:{C_BG}"><th style="text-align:left;padding:6px">Segment</th><th style="text-align:right;padding:6px">After-tax earnings</th><th style="text-align:right;padding:6px">Multiple</th><th style="text-align:right;padding:6px">Value</th></tr>
      {sotp_rows}
      <tr style="border-top:2px solid {C_INK};font-weight:800"><td>Total → per share</td><td style="text-align:right">${s['equity_value']/1000:.1f}B</td><td></td><td style="text-align:right">${s['per_share']:.0f}/sh</td></tr>
    </table>
    <p style="line-height:1.6;font-size:12px;color:{C_BAD};margin-top:6px">SOTP is multiple-sensitive; at these (conservative, peer-typical) multiples it sits below the market price, i.e. the market is already
    crediting synergy upside. It is a cross-check, not a point estimate.</p>

    <h2 style="border-left:4px solid {C_GOLD};padding-left:10px">4 · Two-way sensitivity (data table)</h2>
    <p style="line-height:1.6;font-size:14px">Target price = exit P/E × NTM EPS, flexed across both axes — the standard "data table" that frames the risk/reward:</p>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:{C_BG}"><th style="text-align:left;padding:5px">Scenario \\ exit P/E</th>{head}</tr>
      {sens_rows}
    </table>
    <p style="line-height:1.6;font-size:12px;color:{C_MUTE};margin-top:6px">Highlighted cell = base case (12x × base NTM EPS). At today's ${price:.0f}, the stock is discounting roughly a base-case outcome — upside needs
    either synergy delivery (higher EPS) or a sector re-rating (higher multiple).</p>

    <div style="background:{C_INK};color:#fff;padding:18px 22px;border-radius:8px;margin-top:16px">
      <div style="font-size:12px;letter-spacing:1px;color:{C_GOLD}">IB DEEP-DIVE CONCLUSION</div>
      <div style="font-size:15px;line-height:1.5;margin-top:6px">Four independent lenses converge on "fairly valued with optionality": comps ~in line, SOTP a touch below (market pricing synergies),
      and the Comerica deal <b>dilutive day-1 but ~{ad['rows'][1]['accretion']*100:.0f}% accretive by 2027E</b> with a sub-3-yr TBV earnback. The re-rating case rests on
      management <i>delivering</i> the ${DEAL['synergy_pretax']:.0f}M synergies — which is exactly what the sensitivity table prices.</div>
    </div>

    <p style="font-size:11px;color:{C_MUTE};margin-top:16px;line-height:1.5">Methods: comparable-companies analysis, accretion/dilution &amp; TBV earnback (crossover), sum-of-the-parts, two-way sensitivity.
    Deal terms from FITB/Comerica filings; CMA financials, synergy phasing, TBV dilution and segment multiples are documented estimates.
    Live price via yfinance. Educational portfolio project — <b>not investment advice</b>.</p>
  </div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--to", type=str, default=None)
    args = ap.parse_args()

    print("=" * 70); print("  FITB — INVESTMENT-BANK DEEP-DIVE TECHNIQUES"); print("=" * 70)
    mkt = get_live_market_data(TICKER)
    peers_df = get_peer_multiples(PEERS)

    comps = trading_comps(mkt, peers_df)
    ad = accretion_dilution(mkt)
    s = sotp(mkt)
    sens = sensitivity(mkt)

    print(f"Price ${mkt['price']:.2f}")
    print(f"Comps: peer median P/E {comps['med_pe']:.1f}x, P/TBV {comps['med_pb']:.2f}x -> implied ${comps['implied_pe']:.0f}/${comps['implied_pb']:.0f}")
    print(f"Accretion: day-1 {ad['day1_accr']*100:+.1f}%, 2027E {ad['rows'][1]['accretion']*100:+.1f}%, earnback ~{ad['earnback_yrs']:.1f}y")
    print(f"SOTP: ${s['per_share']:.0f}/sh")
    print(f"Sensitivity base 12x: ${sens['grid']['base'][12.0]:.0f}")

    images = {
        "ib_comps": str(chart_comps(mkt, peers_df, comps)),
        "ib_accretion": str(chart_accretion(ad)),
        "ib_sotp": str(chart_sotp(s, mkt["price"])),
    }
    html = build_html(mkt, comps, ad, s, sens, peers_df, images)
    report = OUT / "FITB_ib_deepdive.html"
    report.write_text(html, encoding="utf-8")
    print(f"\nReport: {report}")

    if not args.no_email:
        send_email(html, images, to_override=args.to,
                   subject="FITB — IB Deep-Dive: comps, Comerica accretion/dilution, SOTP & sensitivity")
    else:
        print("[--no-email] skipped.")
    print("=" * 70, "\nDONE.")


if __name__ == "__main__":
    main()
