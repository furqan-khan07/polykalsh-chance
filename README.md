# polykalsh-chance

Get a better idea about your chances of winning your [Polymarket](https://polymarket.com) and [Kalshi](https://kalshi.com) bets.

Estimates the probability of any stock or crypto asset finishing above or below a strike price using Geometric Brownian Motion — 10,000 simulated paths, live prices via Yahoo Finance, Wilson score confidence intervals.

---

## Live Web App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://polykalsh-chance-dzg5kmwehddz2tcapku7iv.streamlit.app)

---

## How it works

1. Fetches the last 90 days of daily close prices for the asset
2. Computes annualized drift (μ) and volatility (σ) from log returns
3. Simulates 10,000 terminal prices using exact single-step GBM:

   ```
   S_T = S₀ · exp((μ − σ²/2)·T + σ·√T·Z),   Z ~ N(0,1)
   ```

4. Counts what fraction of paths finish above / below the strike
5. Reports probabilities with 95% Wilson score confidence intervals

Stocks use a 252-day trading year; crypto uses a 365-day calendar year.

---

## Run locally

```bash
# 1. Clone
git clone https://github.com/furqan-khan07/polykalsh-chance.git
cd polykalsh-chance

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Web app (Streamlit)
streamlit run app.py

# 3b. CLI version
python polykalsh_chance.py
```

---

## Deploy to Streamlit Cloud (free)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Pick your fork, branch `main`, main file `app.py`
4. Click **Deploy** — done

---

## Example

| Input | Value |
|---|---|
| Asset | BTC |
| Strike | $85,000 |
| Time horizon | 4h |

```
  ABOVE $85,000: 34.2%  [████████░░░░░░░░░░░░░░░░]
  BELOW $85,000: 65.8%  [████████████████░░░░░░░░]

  Confidence (95%):
    Above: 33.3% – 35.1%
    Below: 64.9% – 66.7%

  Annualized Volatility: 68.4%
  Drift (annual): +12.3%
```

---

## Supported assets

**Crypto** (auto-appends `-USD`): BTC, ETH, SOL, DOGE, XRP, ADA, AVAX, MATIC, LTC, LINK

**Stocks / ETFs**: any valid Yahoo Finance ticker — AAPL, TSLA, SPY, QQQ, etc.

---

## Disclaimer

This tool is for informational purposes only and does not constitute financial advice.
