"""
================================================================================
 Fifth Third Bancorp (NASDAQ: FITB) — Institutional Equity Research Engine
================================================================================
A hedge-fund / investment-bank style fundamental analysis and valuation model
for Fifth Third Bancorp, built as a single, reproducible Python project.

WHAT IT DOES
  1. Pulls live market data (price, shares, EPS, dividend, beta, multiples) for
     FITB and a regional-bank peer set via yfinance — with embedded fallbacks so
     it still runs offline.
  2. Combines that with a hand-curated fundamentals snapshot from FITB's 2025
     10-K / Q1-2026 earnings release (segments, ROTCE, efficiency, CET1, credit).
  3. Runs SIX bank-specific valuation methods and blends them:
       - Dividend Discount Model (Gordon growth)
       - Justified Price / Tangible Book  =  (ROTCE - g) / (COE - g)
       - Residual Income (Excess Returns) Model
       - Justified P/E  =  payout x (1+g) / (COE - g)
       - Peer relative (P/E and P/TBV)
       - Analyst consensus cross-check
  4. Adds the analytical frameworks the desk actually uses: CAMELS scorecard,
     DuPont decomposition, and bull / base / bear scenario analysis.
  5. Generates charts + a self-contained HTML research note and emails it.

USAGE
  python fitb_analysis.py                 # build report + charts, then email
  python fitb_analysis.py --no-email      # build only, open output/FITB_report.html
  python fitb_analysis.py --to you@x.com  # override email recipient

Author: (your name) — portfolio project.  NOT investment advice.
================================================================================
"""

from __future__ import annotations

import argparse
import os
import ssl
import smtplib
from datetime import datetime, date
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless — write PNGs, never open a window
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

TICKER = "FITB"
PEERS = ["KEY", "HBAN", "RF", "CFG", "MTB", "PNC", "USB"]  # super-regional peers

# Brand palette for a clean, professional report
C_INK = "#0b1f3a"; C_ACCENT = "#0072CE"; C_GOLD = "#C8A24B"
C_GOOD = "#1a7f5a"; C_BAD = "#b23b3b"; C_MUTE = "#5a6472"; C_BG = "#f6f8fb"


# ─────────────────────────────────────────────────────────────────────────────
# 1) CURATED FUNDAMENTALS SNAPSHOT
#    Sourced from FITB IR / SEC filings (FY2025 results + Q1-2026 release).
#    All $ in millions unless noted. See README for citations.
# ─────────────────────────────────────────────────────────────────────────────
FUND = {
    "company": "Fifth Third Bancorp",
    "ticker": "FITB",
    "exchange": "NASDAQ",
    "hq": "Cincinnati, Ohio",
    "as_of": "FY2025 actuals + Q1-2026 earnings (Comerica closed 1-Feb-2026)",

    # FY2025 (standalone, pre-merger full year)
    "fy2025": {
        "eps_diluted": 3.53,
        "ni_to_common_m": 2400.0,
        "rotce": 0.126,            # return on tangible common equity
        "roa": 0.0119,
        "efficiency_ratio": 0.569,
        "cet1": 0.1081,
        "nim": 0.0305,             # ~3.05% standalone
        "dividend_declared": 1.54,
        "loan_to_core_deposit": 0.72,
        "div_5y_cagr": 0.07,
    },

    # Q1-2026 (first quarter WITH Comerica — 2 months of contribution)
    "q1_2026": {
        "nii_fte_m": 1939.0, "nii_qoq": 0.26, "nii_yoy": 0.34,
        "revenue_fte_m": 2800.0, "revenue_yoy": 0.33,
        "net_income_m": 165.0, "eps_diluted": 0.15,
        "merger_expense_m": 635.0,
        "nim": 0.0330,
        "avg_earning_assets_b": 238.0, "acquired_earning_assets_b": 73.0,
        "loans_b": 178.0, "deposits_b": 234.0,
        "nco": 0.0037,             # net charge-offs 37 bps (lowest since 4Q23)
        "acl_ratio": 0.0179,       # allowance / loans
        "ci_loan_growth": 0.06, "consumer_loan_growth": 0.07,
    },

    # 2026 guidance (management)
    "guide_2026": {
        "nii_low_b": 8.7, "nii_high_b": 8.8,
        "nco_low": 0.0030, "nco_high": 0.0040,
        "cost_synergy_pretax_m": 850.0,  # target run-rate by Q4-2026
    },

    # Comerica acquisition
    "deal": {
        "target": "Comerica (CMA)", "structure": "all-stock",
        "value_announced_b": 10.9, "value_at_close_b": 12.7,
        "exchange_ratio": 1.8663, "premium": 0.20,
        "closed": "2026-02-01", "announced": "2025-10-06",
        "combined_assets_b": 288.0, "rank": "9th largest U.S. bank",
        "fitb_ownership": 0.73, "cma_ownership": 0.27,
    },

    # Reportable segments
    "segments": ["Commercial Banking",
                 "Consumer & Small Business Banking",
                 "Wealth & Asset Management"],

    # Noninterest income by product line, Q3-2025 ($M) — the fee engine
    "fee_lines_q3_2025": {
        "Wealth & Asset Mgmt": 181,
        "Commercial Payments": 157,
        "Consumer Banking": 144,
        "Capital Markets": 115,
        "Commercial Banking": 87,
        "Mortgage Banking": 58,
    },

    # Sell-side snapshot
    "analyst": {"rating": "Buy", "count": 17, "target_avg": 54.38,
                "target_low": 45.0, "target_high": 67.0},
}

# Valuation assumptions (base case, post-merger normalized). Documented so a
# reviewer can challenge every input — the whole point of a research model.
ASSUMPTIONS = {
    "risk_free": 0.042,        # ~10Y UST
    "equity_risk_premium": 0.050,
    "beta_default": 1.15,      # overridden by live beta when available
    "payout_ratio": 0.44,      # 1.54 / 3.53
    # normalized post-synergy ROTCE — mgmt targets peer-leading returns
    "rotce": {"bear": 0.120, "base": 0.140, "bull": 0.155},
    "growth_g": {"bear": 0.030, "base": 0.040, "bull": 0.050},
    # EPS trajectory: 2026 is depressed by merger charges; use normalized power
    "normalized_eps": {"bear": 3.90, "base": 4.35, "bull": 4.80},
    # tangible book per share estimate (post-merger). Refined by live book value.
    "tbvps_estimate": 30.0,
    "ri_years": 5,             # explicit residual-income horizon
}

# Fallback market snapshot if yfinance is unavailable (keeps the demo runnable)
FALLBACK_MKT = {
    "price": 46.25, "shares_m": 905.0, "eps_ttm": 2.40,
    "dividend_rate": 1.58, "beta": 1.15, "book_value_ps": 32.5,
    "pe_ttm": 19.5, "pb": 1.62, "market_cap_b": 41.9,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2) LIVE MARKET DATA (yfinance, graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
def get_live_market_data(ticker: str) -> dict:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            hist = t.history(period="5d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else None
        if not price:
            raise ValueError("no price")
        shares = info.get("sharesOutstanding")
        return {
            "price": float(price),
            "shares_m": (shares / 1e6) if shares else FALLBACK_MKT["shares_m"],
            "eps_ttm": info.get("trailingEps") or FALLBACK_MKT["eps_ttm"],
            "dividend_rate": info.get("dividendRate") or FALLBACK_MKT["dividend_rate"],
            "beta": info.get("beta") or FALLBACK_MKT["beta"],
            "book_value_ps": info.get("bookValue") or FALLBACK_MKT["book_value_ps"],
            "pe_ttm": info.get("trailingPE") or FALLBACK_MKT["pe_ttm"],
            "pb": info.get("priceToBook") or FALLBACK_MKT["pb"],
            "market_cap_b": (info.get("marketCap") / 1e9) if info.get("marketCap") else FALLBACK_MKT["market_cap_b"],
            "_live": True,
        }
    except Exception as ex:
        print(f"[warn] live data unavailable ({ex}); using embedded snapshot")
        d = dict(FALLBACK_MKT); d["_live"] = False
        return d


def get_peer_multiples(tickers: list[str]) -> pd.DataFrame:
    rows = []
    try:
        import yfinance as yf
        for tk in tickers:
            try:
                info = yf.Ticker(tk).info or {}
                rows.append({
                    "ticker": tk,
                    "pe": info.get("trailingPE"),
                    "pb": info.get("priceToBook"),
                    "div_yield": info.get("dividendYield"),
                    "roe": info.get("returnOnEquity"),
                })
            except Exception:
                rows.append({"ticker": tk, "pe": None, "pb": None, "div_yield": None, "roe": None})
    except Exception:
        pass
    if not rows:  # fallback peer medians (approx regional-bank levels)
        rows = [{"ticker": tk, "pe": 11.5, "pb": 1.25, "div_yield": 0.035, "roe": 0.12} for tk in tickers]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3) VALUATION MODELS
# ─────────────────────────────────────────────────────────────────────────────
def cost_of_equity(beta: float) -> float:
    """CAPM: r = rf + beta x ERP."""
    return ASSUMPTIONS["risk_free"] + beta * ASSUMPTIONS["equity_risk_premium"]


def ddm_gordon(d0: float, g: float, r: float) -> float:
    """Gordon growth DDM: V = D0*(1+g) / (r - g)."""
    if r <= g:
        return float("nan")
    return d0 * (1 + g) / (r - g)


def justified_ptbv(rotce: float, g: float, r: float, tbvps: float) -> float:
    """
    Warranted price via the bank identity  P/TBV = (ROTCE - g) / (COE - g).
    This is the single most important valuation relationship for banks: a bank
    is worth more than tangible book only to the extent it earns above its cost
    of equity, and that premium compounds with growth.
    """
    if r <= g:
        return float("nan")
    multiple = (rotce - g) / (r - g)
    return max(multiple, 0.0) * tbvps


def residual_income(bvps: float, rotce: float, r: float, g: float, years: int) -> float:
    """
    Excess-returns / residual-income intrinsic value:
        V = BVPS + Σ_t RI_t/(1+r)^t + TerminalRI/((r-g)(1+r)^N)
    where RI_t = (ROE - r) x BVPS_(t-1). Book compounds at retention x ROE.
    """
    retention = 1 - ASSUMPTIONS["payout_ratio"]
    bv = bvps
    pv = 0.0
    last_ri = 0.0
    for t in range(1, years + 1):
        ri = (rotce - r) * bv
        pv += ri / (1 + r) ** t
        last_ri = ri
        bv = bv * (1 + retention * rotce)  # clean-surplus book growth
    # terminal value of a growing residual income stream
    term = (last_ri * (1 + g)) / ((r - g)) if r > g else 0.0
    pv += term / (1 + r) ** years
    return bvps + pv


def justified_pe(payout: float, g: float, r: float, eps: float) -> float:
    """Leading justified P/E = payout*(1+g)/(r-g); target = P/E x EPS."""
    if r <= g:
        return float("nan")
    pe = payout * (1 + g) / (r - g)
    return pe * eps


def scenario_valuation(mkt: dict, tbvps: float, bvps: float) -> pd.DataFrame:
    """Run every method across bear/base/bull and return a tidy table."""
    d0 = mkt["dividend_rate"]
    rows = []
    for case in ("bear", "base", "bull"):
        r = cost_of_equity(mkt["beta"])
        g = ASSUMPTIONS["growth_g"][case]
        rotce = ASSUMPTIONS["rotce"][case]
        eps = ASSUMPTIONS["normalized_eps"][case]
        rows.append({
            "scenario": case,
            "COE": r, "g": g, "ROTCE": rotce, "norm_EPS": eps,
            "DDM": ddm_gordon(d0, g, r),
            "Justified P/TBV": justified_ptbv(rotce, g, r, tbvps),
            "Residual Income": residual_income(bvps, rotce, r, g, ASSUMPTIONS["ri_years"]),
            "Justified P/E": justified_pe(ASSUMPTIONS["payout_ratio"], g, r, eps),
        })
    return pd.DataFrame(rows)


def peer_relative_value(mkt: dict, peers: pd.DataFrame) -> dict:
    med_pe = float(np.nanmedian(peers["pe"].astype(float))) if peers["pe"].notna().any() else 11.5
    med_pb = float(np.nanmedian(peers["pb"].astype(float))) if peers["pb"].notna().any() else 1.25
    eps = ASSUMPTIONS["normalized_eps"]["base"]
    return {
        "peer_median_pe": med_pe,
        "peer_median_pb": med_pb,
        "value_on_pe": med_pe * eps,
        "value_on_pb": med_pb * mkt["book_value_ps"],
    }


def blended_fair_value(scen: pd.DataFrame, peer_rel: dict, analyst_avg: float) -> dict:
    base = scen[scen["scenario"] == "base"].iloc[0]
    methods = {
        "DDM (Gordon)": base["DDM"],
        "Justified P/TBV": base["Justified P/TBV"],
        "Residual Income": base["Residual Income"],
        "Justified P/E": base["Justified P/E"],
        "Peer P/E": peer_rel["value_on_pe"],
        "Peer P/TBV": peer_rel["value_on_pb"],
        "Analyst consensus": analyst_avg,
    }
    vals = [v for v in methods.values() if v and not np.isnan(v) and v > 0]
    fair = float(np.mean(vals)) if vals else float("nan")
    return {"methods": methods, "fair_value": fair}


# ─────────────────────────────────────────────────────────────────────────────
# 4) ANALYTICAL FRAMEWORKS (CAMELS + DuPont)
# ─────────────────────────────────────────────────────────────────────────────
def camels_scorecard() -> list[dict]:
    f = FUND
    return [
        {"pillar": "Capital", "metric": f"CET1 {f['fy2025']['cet1']*100:.2f}%",
         "read": "Well above the ~7% regulatory minimum + buffers — ample capacity to absorb the Comerica goodwill and keep buying back stock.", "grade": "Strong"},
        {"pillar": "Asset Quality", "metric": f"NCOs {f['q1_2026']['nco']*100:.0f} bps · ACL {f['q1_2026']['acl_ratio']*100:.2f}%",
         "read": "Q1-26 charge-offs the lowest since 4Q23; allowance a healthy 1.79% of loans. Credit is benign, not stretched.", "grade": "Strong"},
        {"pillar": "Management", "metric": f"${f['guide_2026']['cost_synergy_pretax_m']:.0f}M synergy target",
         "read": "Executing the largest deal in its history; credibility hinges on hitting the Q4-26 cost-synergy run-rate.", "grade": "Watch"},
        {"pillar": "Earnings", "metric": f"ROTCE {f['fy2025']['rotce']*100:.1f}% · Eff. {f['fy2025']['efficiency_ratio']*100:.1f}%",
         "read": "Double-digit ROTCE and a sub-57% efficiency ratio are top-quartile for regionals; merger accretion should push both higher.", "grade": "Strong"},
        {"pillar": "Liquidity", "metric": f"Loan/core deposit {f['fy2025']['loan_to_core_deposit']*100:.0f}%",
         "read": "72% loan-to-core-deposit = a deposit-funded balance sheet with room to grow loans without chasing hot money.", "grade": "Strong"},
        {"pillar": "Sensitivity", "metric": f"NIM {f['q1_2026']['nim']*100:.2f}%",
         "read": "Asset-sensitive; NIM expanding as funding costs fall. 2026 NII guide $8.7-8.8B assumes no rate cuts — upside if the Fed eases.", "grade": "Positive"},
    ]


def dupont() -> dict:
    """Simplified bank DuPont: ROE = ROA x equity multiplier (leverage)."""
    roa = FUND["fy2025"]["roa"]
    rotce = FUND["fy2025"]["rotce"]
    leverage = rotce / roa  # implied tangible equity multiplier
    return {"roa": roa, "rotce": rotce, "implied_leverage": leverage}


# ─────────────────────────────────────────────────────────────────────────────
# 5) CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(colors=C_MUTE)


def chart_price(ticker: str) -> Path | None:
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period="2y")
        if h.empty:
            return None
        h["SMA50"] = h["Close"].rolling(50).mean()
        h["SMA200"] = h["Close"].rolling(200).mean()
        fig, ax = plt.subplots(figsize=(9, 3.6), dpi=130)
        ax.plot(h.index, h["Close"], color=C_ACCENT, lw=1.6, label="Close")
        ax.plot(h.index, h["SMA50"], color=C_GOLD, lw=1.1, label="SMA50")
        ax.plot(h.index, h["SMA200"], color=C_MUTE, lw=1.1, label="SMA200")
        ax.set_title("FITB — Price with 50/200-day moving averages", color=C_INK, fontweight="bold")
        ax.legend(frameon=False, fontsize=8); _style(ax)
        p = OUT / "chart_price.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig)
        return p
    except Exception as ex:
        print(f"[warn] price chart skipped: {ex}")
        return None


def chart_fee_mix() -> Path:
    d = FUND["fee_lines_q3_2025"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=130)
    labels = list(d.keys()); vals = list(d.values())
    colors = [C_ACCENT, C_GOLD, C_GOOD, C_INK, C_MUTE, "#8aa"]
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 2, i, f"${v}M", va="center", fontsize=8, color=C_INK)
    ax.set_title("Fee income by product line (Q3-2025, $M)", color=C_INK, fontweight="bold")
    _style(ax); ax.grid(axis="x", alpha=0.2); ax.grid(axis="y", visible=False)
    p = OUT / "chart_fees.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


def chart_football_field(methods: dict, price: float, fair: float) -> Path:
    items = [(k, v) for k, v in methods.items() if v and not np.isnan(v) and v > 0]
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=130)
    names = [k for k, _ in items]; vals = [v for _, v in items]
    cols = [C_GOOD if v >= price else C_BAD for v in vals]
    ax.barh(names[::-1], vals[::-1], color=cols[::-1], alpha=0.85)
    ax.axvline(price, color=C_INK, ls="--", lw=1.4, label=f"Price ${price:.2f}")
    ax.axvline(fair, color=C_GOLD, ls="-", lw=1.6, label=f"Blended FV ${fair:.2f}")
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.3, i, f"${v:.0f}", va="center", fontsize=8, color=C_INK)
    ax.set_title("Valuation football field — target by method", color=C_INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=8); _style(ax); ax.grid(axis="x", alpha=0.2); ax.grid(axis="y", visible=False)
    p = OUT / "chart_valuation.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


def chart_peers(peers: pd.DataFrame, mkt: dict) -> Path:
    df = peers.copy()
    fitb_row = {"ticker": "FITB", "pe": mkt["pe_ttm"], "pb": mkt["pb"]}
    df = pd.concat([df, pd.DataFrame([fitb_row])], ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.8, 3.6), dpi=130)
    x = np.arange(len(df)); w = 0.4
    ax.bar(x - w/2, df["pe"].astype(float), w, color=C_ACCENT, label="P/E (ttm)")
    ax.bar(x + w/2, df["pb"].astype(float) * 5, w, color=C_GOLD, label="P/B x5 (scaled)")
    ax.set_xticks(x); ax.set_xticklabels(df["ticker"], fontsize=8)
    for i, tk in enumerate(df["ticker"]):
        if tk == "FITB":
            ax.get_xticklabels()[i].set_color(C_BAD); ax.get_xticklabels()[i].set_fontweight("bold")
    ax.set_title("Relative valuation vs super-regional peers", color=C_INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=8); _style(ax)
    p = OUT / "chart_peers.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 6) HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────
def _pct(x): return f"{x*100:.1f}%"


def build_html(mkt, scen, peer_rel, blend, camels, du, images: dict) -> str:
    f = FUND; price = mkt["price"]; fair = blend["fair_value"]
    upside = (fair / price - 1) * 100 if price else 0
    rec = "BUY" if upside > 12 else ("HOLD" if upside > -5 else "REDUCE")
    rec_col = C_GOOD if rec == "BUY" else (C_GOLD if rec == "HOLD" else C_BAD)
    coe = cost_of_equity(mkt["beta"])
    today = date.today().strftime("%d %b %Y")

    def img(key, alt):
        return f'<img src="cid:{key}" alt="{alt}" style="width:100%;max-width:720px;border:1px solid #e5e9f0;border-radius:8px;margin:8px 0;">' if key in images else ""

    fee_rows = "".join(
        f"<tr><td>{k}</td><td style='text-align:right'>${v}</td></tr>"
        for k, v in f["fee_lines_q3_2025"].items())

    method_rows = "".join(
        f"<tr><td>{k}</td><td style='text-align:right'>${v:,.2f}</td>"
        f"<td style='text-align:right;color:{C_GOOD if v>=price else C_BAD}'>{(v/price-1)*100:+.1f}%</td></tr>"
        for k, v in blend["methods"].items() if v and not np.isnan(v) and v > 0)

    scen_rows = "".join(
        f"<tr><td style='text-transform:capitalize'>{r['scenario']}</td>"
        f"<td style='text-align:right'>{_pct(r['COE'])}</td>"
        f"<td style='text-align:right'>{_pct(r['g'])}</td>"
        f"<td style='text-align:right'>{_pct(r['ROTCE'])}</td>"
        f"<td style='text-align:right'>${r['DDM']:.0f}</td>"
        f"<td style='text-align:right'>${r['Justified P/TBV']:.0f}</td>"
        f"<td style='text-align:right'>${r['Residual Income']:.0f}</td>"
        f"<td style='text-align:right'>${r['Justified P/E']:.0f}</td></tr>"
        for _, r in scen.iterrows())

    _camels_parts = []
    for c in camels:
        good = c["grade"] in ("Strong", "Positive")
        gcol = C_GOOD if good else C_GOLD
        _camels_parts.append(
            f"<tr><td><b>{c['pillar']}</b></td><td>{c['metric']}</td>"
            f"<td><span style='color:{gcol}'>{c['grade']}</span></td>"
            f"<td style='color:{C_MUTE}'>{c['read']}</td></tr>")
    camels_rows = "".join(_camels_parts)

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;background:{C_BG};font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:{C_INK};">
<div style="max-width:760px;margin:0 auto;background:#fff;">

  <div style="background:{C_INK};color:#fff;padding:26px 30px;">
    <div style="font-size:12px;letter-spacing:2px;color:{C_GOLD};font-weight:700;">EQUITY RESEARCH · U.S. REGIONAL BANKS</div>
    <div style="font-size:26px;font-weight:800;margin-top:4px;">Fifth Third Bancorp <span style="color:{C_GOLD}">(NASDAQ: FITB)</span></div>
    <div style="font-size:13px;color:#c7d0dc;margin-top:6px;">{f['hq']} · {today} · {f['as_of']}</div>
  </div>

  <div style="display:flex;flex-wrap:wrap;padding:18px 30px;gap:14px;border-bottom:1px solid #eef1f6;">
    <div style="flex:1;min-width:120px;"><div style="font-size:11px;color:{C_MUTE}">PRICE</div><div style="font-size:22px;font-weight:800">${price:,.2f}</div></div>
    <div style="flex:1;min-width:120px;"><div style="font-size:11px;color:{C_MUTE}">BLENDED FAIR VALUE</div><div style="font-size:22px;font-weight:800;color:{C_ACCENT}">${fair:,.2f}</div></div>
    <div style="flex:1;min-width:120px;"><div style="font-size:11px;color:{C_MUTE}">UPSIDE</div><div style="font-size:22px;font-weight:800;color:{C_GOOD if upside>0 else C_BAD}">{upside:+.1f}%</div></div>
    <div style="flex:1;min-width:120px;"><div style="font-size:11px;color:{C_MUTE}">RATING</div><div style="font-size:22px;font-weight:800;color:{rec_col}">{rec}</div></div>
  </div>

  <div style="padding:22px 30px;">
    <h2 style="color:{C_INK};border-left:4px solid {C_GOLD};padding-left:10px;">1 · Investment Thesis</h2>
    <p style="line-height:1.6;font-size:14px;">Fifth Third has just transformed itself: the all-stock acquisition of <b>Comerica</b>
    (${f['deal']['value_at_close_b']}B, closed {f['deal']['closed']}) makes it the <b>{f['deal']['rank']}</b> with roughly
    <b>${f['deal']['combined_assets_b']:.0f}B</b> of assets. The market is looking through a Q1-26 EPS trough of $0.15 — depressed by
    <b>${f['q1_2026']['merger_expense_m']:.0f}M</b> of one-off merger charges — toward a franchise with a sub-57% efficiency ratio,
    double-digit ROTCE, a low-{int(f['fy2025']['loan_to_core_deposit']*100)}% loan-to-core-deposit ratio, and <b>${f['guide_2026']['cost_synergy_pretax_m']:.0f}M</b>
    of targeted cost synergies. The core question for the stock: does management capture those synergies and lift combined ROTCE toward
    ~14%+ without a credit or integration accident? If yes, the shares are worth meaningfully more than tangible book. Our blended
    fair value of <b>${fair:,.2f}</b> implies <b>{upside:+.1f}%</b> vs the current ${price:,.2f}.</p>

    <h2 style="color:{C_INK};border-left:4px solid {C_GOLD};padding-left:10px;">2 · How Fifth Third makes money</h2>
    <p style="line-height:1.6;font-size:14px;">The bank reports three segments — <b>{', '.join(f['segments'])}</b>. Roughly two-thirds of
    revenue is <b>net interest income</b> (spread on ${f['q1_2026']['loans_b']:.0f}B of loans funded by ${f['q1_2026']['deposits_b']:.0f}B of
    deposits; Q1-26 NIM {f['q1_2026']['nim']*100:.2f}%), and the rest is <b>fee income</b> — a genuinely diversified, capital-light engine:</p>
    {img('fees','fee mix')}
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:{C_BG};"><th style="text-align:left;padding:6px">Fee line (Q3-25)</th><th style="text-align:right;padding:6px">$M</th></tr>
      {fee_rows}
    </table>
    <p style="line-height:1.6;font-size:13px;color:{C_MUTE};margin-top:6px;">Wealth &amp; Asset Management and Commercial Payments are the standouts —
    recurring, fee-based, low-capital revenue that the Street rewards with a higher multiple than pure spread income.</p>

    <h2 style="color:{C_INK};border-left:4px solid {C_GOLD};padding-left:10px;">3 · CAMELS scorecard</h2>
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:{C_BG};"><th style="text-align:left;padding:6px">Pillar</th><th style="text-align:left;padding:6px">Metric</th><th style="text-align:left;padding:6px">Grade</th><th style="text-align:left;padding:6px">Read</th></tr>
      {camels_rows}
    </table>

    <h2 style="color:{C_INK};border-left:4px solid {C_GOLD};padding-left:10px;">4 · Profitability (DuPont)</h2>
    <p style="line-height:1.6;font-size:14px;">ROA <b>{_pct(du['roa'])}</b> × implied tangible leverage <b>{du['implied_leverage']:.1f}×</b> ≈ ROTCE <b>{_pct(du['rotce'])}</b>.
    A 1.19% ROA is strong for a regional; the returns are earned on a conservatively-levered balance sheet rather than by stacking risk.</p>

    <h2 style="color:{C_INK};border-left:4px solid {C_GOLD};padding-left:10px;">5 · Valuation — six methods, blended</h2>
    <p style="line-height:1.6;font-size:14px;">Cost of equity (CAPM) = {ASSUMPTIONS['risk_free']*100:.1f}% + {mkt['beta']:.2f}β × {ASSUMPTIONS['equity_risk_premium']*100:.1f}% =
    <b>{_pct(coe)}</b>. The anchor for any bank is the identity <b>P/TBV = (ROTCE − g)/(COE − g)</b> — a bank only deserves a premium to
    tangible book when it out-earns its cost of equity.</p>
    {img('valuation','valuation football field')}
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:{C_BG};"><th style="text-align:left;padding:6px">Method</th><th style="text-align:right;padding:6px">Value</th><th style="text-align:right;padding:6px">vs Price</th></tr>
      {method_rows}
      <tr style="border-top:2px solid {C_INK};font-weight:800;"><td style="padding:6px">Blended fair value</td><td style="text-align:right">${fair:,.2f}</td><td style="text-align:right;color:{C_GOOD if upside>0 else C_BAD}">{upside:+.1f}%</td></tr>
    </table>

    <h3 style="color:{C_INK};margin-top:18px;">Scenario analysis (bear / base / bull)</h3>
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
      <tr style="background:{C_BG};"><th style="padding:5px;text-align:left">Case</th><th style="padding:5px;text-align:right">COE</th><th style="padding:5px;text-align:right">g</th><th style="padding:5px;text-align:right">ROTCE</th><th style="padding:5px;text-align:right">DDM</th><th style="padding:5px;text-align:right">P/TBV</th><th style="padding:5px;text-align:right">Resid.Inc</th><th style="padding:5px;text-align:right">P/E</th></tr>
      {scen_rows}
    </table>

    <h2 style="color:{C_INK};border-left:4px solid {C_GOLD};padding-left:10px;">6 · Relative valuation</h2>
    <p style="line-height:1.6;font-size:14px;">Peer median P/E <b>{peer_rel['peer_median_pe']:.1f}×</b>, P/TBV <b>{peer_rel['peer_median_pb']:.2f}×</b>
    (peers: {', '.join(PEERS)}).</p>
    {img('peers','peer comparison')}

    <h2 style="color:{C_INK};border-left:4px solid {C_GOLD};padding-left:10px;">7 · Price &amp; trend</h2>
    {img('price','price chart')}

    <h2 style="color:{C_INK};border-left:4px solid {C_GOLD};padding-left:10px;">8 · Risks &amp; catalysts</h2>
    <p style="line-height:1.6;font-size:14px;"><b>Catalysts:</b> hitting the ${f['guide_2026']['cost_synergy_pretax_m']:.0f}M synergy run-rate by Q4-26;
    NIM expansion if the Fed cuts (guide assumes none); buybacks resuming once the deal's capital is digested; WAM/payments fee momentum.
    <br><b>Risks:</b> integration missteps on the largest deal in company history; commercial-real-estate / C&amp;I credit if the economy rolls over
    (NCOs are low now at {f['q1_2026']['nco']*100:.0f}bps but that is cyclical); deposit competition compressing NIM; regulatory review of a top-10 bank;
    goodwill from the deal makes tangible book optically thinner.</p>

    <div style="background:{rec_col};color:#fff;padding:18px 22px;border-radius:8px;margin-top:16px;">
      <div style="font-size:12px;letter-spacing:1px;opacity:.85">RECOMMENDATION</div>
      <div style="font-size:22px;font-weight:800;">{rec} · Fair value ${fair:,.2f} · {upside:+.1f}% vs ${price:,.2f}</div>
      <div style="font-size:12px;opacity:.9;margin-top:6px;">Sell-side consensus: {f['analyst']['rating']} ({f['analyst']['count']} analysts), avg target ${f['analyst']['target_avg']:.2f} (range ${f['analyst']['target_low']:.0f}–${f['analyst']['target_high']:.0f}).</div>
    </div>

    <p style="font-size:11px;color:{C_MUTE};margin-top:18px;line-height:1.5;">
    Methodology: DDM (Gordon), justified P/TBV = (ROTCE−g)/(COE−g), residual-income (excess returns), justified P/E, peer-relative, and
    analyst consensus, blended equal-weight. Market data via yfinance ({'live' if mkt.get('_live') else 'embedded snapshot'});
    fundamentals from FITB IR / SEC filings (FY2025 + Q1-2026). Educational portfolio project — <b>NOT investment advice</b>.</p>
  </div>
</div></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# 7) EMAIL
# ─────────────────────────────────────────────────────────────────────────────
def _load_email_creds():
    """
    Read email credentials from (in order): environment variables, a project-local
    .env file, or a path given by the FITB_ENV_PATH env var. Never hard-code
    secrets or personal paths — keeps the repo safe to publish.
    """
    user = os.environ.get("EMAIL_USER"); pw = os.environ.get("EMAIL_APP_PASSWORD")
    to = os.environ.get("EMAIL_TO")
    candidates = [BASE / ".env"]
    if os.environ.get("FITB_ENV_PATH"):
        candidates.append(Path(os.environ["FITB_ENV_PATH"]))
    for envfile in candidates:
        if not envfile.exists():
            continue
        for line in envfile.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("EMAIL_USER") and not user:
                user = line.split("=", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("EMAIL_APP_PASSWORD") and not pw:
                pw = line.split("=", 1)[1].strip().strip('"').strip("'").replace(" ", "")
            elif line.startswith("EMAIL_TO") and not to:
                to = line.split("=", 1)[1].strip().strip('"').strip("'")
    return user, pw, to


def send_email(html: str, images: dict, to_override: str | None = None,
               subject: str | None = None) -> bool:
    user, pw, to = _load_email_creds()
    to = to_override or to
    if not (user and pw and to):
        print("[email] creds/recipient missing — skipped. Set EMAIL_USER/EMAIL_APP_PASSWORD/EMAIL_TO.")
        return False
    msg = EmailMessage()
    msg["From"] = f"Equity Research <{user}>"
    msg["To"] = to
    msg["Subject"] = subject or "Equity Research — Fifth Third Bancorp (FITB): Institutional Fundamental Analysis & Valuation"
    msg.set_content("Your email client does not support HTML. See the attached HTML report.")
    msg.add_alternative(html, subtype="html")
    # embed images as inline cid attachments on the HTML part
    html_part = msg.get_payload()[-1]
    for cid, path in images.items():
        try:
            data = Path(path).read_bytes()
            html_part.add_related(data, maintype="image", subtype="png", cid=f"<{cid}>", filename=f"{cid}.png")
        except Exception as ex:
            print(f"[email] could not embed {cid}: {ex}")
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls(context=ctx); s.login(user, pw); s.send_message(msg)
        print(f"[email] sent to {to}")
        return True
    except Exception as ex:
        print(f"[email] failed: {ex}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 8) MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Fifth Third Bancorp equity research engine")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--to", type=str, default=None, help="override email recipient")
    args = ap.parse_args()

    print("=" * 70)
    print("  FIFTH THIRD BANCORP (FITB) — INSTITUTIONAL EQUITY RESEARCH ENGINE")
    print("=" * 70)

    mkt = get_live_market_data(TICKER)
    print(f"Price ${mkt['price']:.2f} | shares {mkt['shares_m']:.0f}M | beta {mkt['beta']:.2f} | "
          f"P/E {mkt['pe_ttm']:.1f} | P/B {mkt['pb']:.2f} | data={'LIVE' if mkt['_live'] else 'fallback'}")

    # Tangible book estimate: blend live book value with our post-merger estimate
    tbvps = round(0.5 * mkt["book_value_ps"] + 0.5 * ASSUMPTIONS["tbvps_estimate"], 2)
    bvps = mkt["book_value_ps"]

    peers = get_peer_multiples(PEERS)
    scen = scenario_valuation(mkt, tbvps, bvps)
    peer_rel = peer_relative_value(mkt, peers)
    blend = blended_fair_value(scen, peer_rel, FUND["analyst"]["target_avg"])
    camels = camels_scorecard()
    du = dupont()

    print(f"\nCost of equity (CAPM): {cost_of_equity(mkt['beta'])*100:.2f}%")
    print("Fair value by method:")
    for k, v in blend["methods"].items():
        if v and not np.isnan(v):
            print(f"   {k:20s} ${v:8.2f}   ({(v/mkt['price']-1)*100:+.1f}%)")
    print(f"   {'BLENDED FAIR VALUE':20s} ${blend['fair_value']:8.2f}   "
          f"({(blend['fair_value']/mkt['price']-1)*100:+.1f}%)")

    # charts
    images = {}
    for key, fn in (("price", lambda: chart_price(TICKER)),
                    ("fees", chart_fee_mix),
                    ("valuation", lambda: chart_football_field(blend["methods"], mkt["price"], blend["fair_value"])),
                    ("peers", lambda: chart_peers(peers, mkt))):
        p = fn()
        if p:
            images[key] = str(p)

    html = build_html(mkt, scen, peer_rel, blend, camels, du, images)
    report = OUT / "FITB_report.html"
    report.write_text(html, encoding="utf-8")
    print(f"\nReport written: {report}")

    if not args.no_email:
        send_email(html, images, to_override=args.to)
    else:
        print("[--no-email] skipped sending.")

    print("=" * 70)
    print("DONE.")


if __name__ == "__main__":
    main()
