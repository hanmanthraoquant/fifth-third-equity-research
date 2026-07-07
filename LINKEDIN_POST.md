# LinkedIn post — draft

> Copy-paste this into LinkedIn. Replace `LINK` with the GitHub repo URL, and the
> screenshot line with 1–2 charts from `output/` (the valuation football field and
> the EPS forecast look best). Feel free to trim — LinkedIn shows ~3 lines before
> "…see more", so the first two lines matter most.

---

I spent this week building an equity research project on Fifth Third Bancorp ($FITB) — and tried to value it the way an actual investment bank would. 🏦📊

I picked FITB because it just did something huge: it acquired Comerica in a ~$12.7B all-stock deal (closed Feb 2026), making it the 9th-largest US bank (~$288B in assets). The first quarter after the deal, reported EPS dropped to $0.15 — but that was because of ~$635M of one-time merger costs. The interesting question for a student like me: what is the business actually worth once you look through that noise?

So I built it out in Python, end to end. What's inside:

🔹 6 valuation methods → Dividend Discount Model, Residual Income (Excess Returns), Justified P/TBV = (ROTCE − g)/(COE − g), Justified P/E, peer comps, and analyst consensus
🔹 A consumer-banking deep dive → the loan book by product (auto, mortgage, home equity, credit card, solar), the yields/finance charges, interchange + deposit fees, and the segment P&L
🔹 Standard trend analysis → horizontal analysis, vertical/common-size, and a rate/volume variance decomposition (the same table banks disclose in their 10-K)
🔹 A driver-based earnings forecast → EPS ~$3.68 (2026E) → ~$4.80 (2028E)
🔹 Investment-bank techniques → trading comps, M&A accretion/dilution + tangible-book earnback on the Comerica deal, Sum-of-the-Parts, and a two-way sensitivity table
🔹 Two techniques I was proud to add → a P/TBV-vs-ROTCE peer regression (the "warranted multiple") and a reverse DCF that backs out the growth the market is already pricing in

Biggest lesson: the methods disagree, and that's the point. A DDM lowballs a bank that retains most of its earnings, while residual income and P/TBV capture the compounding. When I weighted all 14 techniques, I landed on a ~$58 target vs a ~$58 price — basically fairly valued. The reverse DCF said the market is pricing ~3.9% long-term growth, almost exactly my base-case assumption. 🎯

A few things I learned:
✅ For banks, tangible book value and ROTCE matter more than revenue multiples
✅ Fifth Third is a deposit/debit-led bank, NOT a credit-card lender (card book is only ~1.4% of loans)
✅ An all-stock deal is dilutive on day one but can be accretive once cost synergies land (+13.5% by 2027E here)

All the code is open-source and reproducible (pulls live data with yfinance, has offline fallbacks): LINK

⚠️ This is a learning/portfolio project — NOT investment advice.

Would love feedback from anyone in equity research or banking. What method would you add? 🙏

#EquityResearch #Valuation #InvestmentBanking #FinTech #Python #Banking #FinancialModeling #FifthThird #DCF #StockAnalysis
