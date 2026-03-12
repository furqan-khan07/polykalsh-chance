"""
polykalsh_chance.py — Monte Carlo price probability predictor
Uses Geometric Brownian Motion to simulate asset price paths and estimate
the probability of finishing above/below a strike price at a given time horizon.
"""

import re
import sys
import numpy as np
import yfinance as yf
from scipy.stats import norm

# ── Constants ─────────────────────────────────────────────────────────────────
CRYPTO_TICKERS = {"BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "AVAX", "MATIC", "LTC", "LINK"}
TRADING_HOURS_PER_YEAR = 252 * 6.5   # 1638.0  — stocks
CALENDAR_HOURS_PER_YEAR = 365 * 24   # 8760    — crypto
N_SIMULATIONS = 10_000
LOOKBACK_DAYS = 90
CONFIDENCE_LEVEL = 0.95
BAR_WIDTH = 24


def normalize_ticker(ticker: str) -> str:
    """Uppercase + strip; append -USD for crypto if not already present."""
    t = ticker.strip().upper()
    base = t.split("-")[0]
    if base in CRYPTO_TICKERS and not t.endswith("-USD"):
        return f"{base}-USD"
    return t


def is_crypto(ticker: str) -> bool:
    """Return True if ticker is a crypto asset."""
    return ticker.endswith("-USD") and ticker.split("-")[0] in CRYPTO_TICKERS


def parse_time(time_str: str) -> float:
    """Parse time string like '4h', '2d', '1w', '3m' into hours."""
    unit_map = {"h": 1, "d": 24, "w": 168, "m": 720}
    m = re.match(r'^(\d+\.?\d*)([hdwm])$', time_str.strip().lower())
    if not m:
        raise ValueError(
            f"Invalid time format '{time_str}'. Use e.g. 4h, 2d, 1w, 3m "
            f"(h=hours, d=days, w=weeks, m=months)"
        )
    value, unit = float(m.group(1)), m.group(2)
    return value * unit_map[unit]


def fetch_data(ticker: str, lookback_days: int = LOOKBACK_DAYS):
    """Fetch historical price data and current price for ticker.

    Returns:
        (current_price: float, close_prices: np.ndarray)
    """
    print(f"  Fetching data for {ticker}...")
    t = yf.Ticker(ticker)
    hist = t.history(period=f"{lookback_days}d")

    if hist.empty or len(hist) < 10:
        raise ValueError(
            f"Insufficient data for '{ticker}'. "
            "Check the ticker symbol and try again."
        )

    try:
        current_price = t.fast_info["last_price"]
        if current_price is None or current_price <= 0:
            raise KeyError
    except (KeyError, AttributeError):
        current_price = float(hist["Close"].iloc[-1])

    return float(current_price), hist["Close"].values.astype(float)


def calc_gbm_params(prices: np.ndarray, periods_per_year: int):
    """Compute annualized drift (mu) and volatility (sigma) from price history.

    Returns:
        (mu_annual: float, sigma_annual: float)
    """
    log_returns = np.diff(np.log(prices))
    mu_annual = log_returns.mean() * periods_per_year
    sigma_annual = log_returns.std(ddof=1) * np.sqrt(periods_per_year)

    if sigma_annual == 0:
        raise ValueError("Computed volatility is zero — insufficient price variation in data.")

    return mu_annual, sigma_annual


def run_monte_carlo(S0: float, mu: float, sigma: float, T_years: float, n_sims: int) -> np.ndarray:
    """Run vectorized single-step GBM simulation.

    Uses the exact terminal distribution — no discretization error.
    Returns array of n_sims final prices.
    """
    Z = np.random.standard_normal(n_sims)
    return S0 * np.exp((mu - 0.5 * sigma ** 2) * T_years + sigma * np.sqrt(T_years) * Z)


def compute_stats(final_prices: np.ndarray, strike: float) -> dict:
    """Compute probabilities and Wilson score confidence intervals."""
    n = len(final_prices)
    p_above = float(np.sum(final_prices > strike)) / n
    p_below = 1.0 - p_above

    z = norm.ppf(0.975)  # 1.96 for 95% CI

    def wilson_ci(p, n, z):
        denom = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denom
        margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    ci_above = wilson_ci(p_above, n, z)
    ci_below = wilson_ci(p_below, n, z)

    return {
        "p_above": p_above,
        "p_below": p_below,
        "ci_above": ci_above,
        "ci_below": ci_below,
    }


def print_results(
    ticker: str,
    current_price: float,
    strike: float,
    time_hours: float,
    stats: dict,
    mu: float,
    sigma: float,
    n_sims: int,
) -> None:
    """Print formatted simulation results."""
    # Format time for display
    if time_hours < 24:
        time_label = f"{time_hours:g} hour{'s' if time_hours != 1 else ''}"
    elif time_hours < 168:
        days = time_hours / 24
        time_label = f"{days:g} day{'s' if days != 1 else ''}"
    elif time_hours < 720:
        weeks = time_hours / 168
        time_label = f"{weeks:g} week{'s' if weeks != 1 else ''}"
    else:
        months = time_hours / 720
        time_label = f"{months:g} month{'s' if months != 1 else ''}"

    def bar(prob: float) -> str:
        filled = round(prob * BAR_WIDTH)
        return "█" * filled + "░" * (BAR_WIDTH - filled)

    border = "━" * 42
    p_above = stats["p_above"]
    p_below = stats["p_below"]
    ci_a_lo, ci_a_hi = stats["ci_above"]
    ci_b_lo, ci_b_hi = stats["ci_below"]

    print(f"\n  {border}")
    print(f"    Asset: {ticker:<12}  Live Price: ${current_price:,.2f}")
    print(f"    Strike: ${strike:,.2f}{'':>3}  Time: {time_label}")
    print(f"  {border}")
    print(f"    Simulations: {n_sims:,}\n")
    print(f"    ABOVE ${strike:,.2f}: {p_above*100:.1f}%  [{bar(p_above)}]")
    print(f"    BELOW ${strike:,.2f}: {p_below*100:.1f}%  [{bar(p_below)}]")
    print(f"\n    Confidence (95%):")
    print(f"      Above: {ci_a_lo*100:.1f}% – {ci_a_hi*100:.1f}%")
    print(f"      Below: {ci_b_lo*100:.1f}% – {ci_b_hi*100:.1f}%")
    print(f"\n    Annualized Volatility: {sigma*100:.1f}%")
    sign = "+" if mu >= 0 else ""
    print(f"    Drift (annual): {sign}{mu*100:.1f}%")
    print(f"  {border}\n")


def main() -> None:
    print("\n  ╔══════════════════════════════════════╗")
    print("  ║   Monte Carlo Price Predictor  v1.0  ║")
    print("  ║   Polymarket / Kalshi Probability    ║")
    print("  ╚══════════════════════════════════════╝\n")

    try:
        # ── Step 1: Ticker ──────────────────────────────────────────────────
        while True:
            try:
                raw = input("  Asset ticker (e.g. BTC, AAPL, ETH): ").strip()
                if not raw:
                    print("  Please enter a ticker symbol.")
                    continue
                ticker = normalize_ticker(raw)
                break
            except ValueError as e:
                print(f"  Error: {e}")

        # ── Step 2: Strike price ────────────────────────────────────────────
        while True:
            try:
                raw = input("  Strike price (e.g. 85000): ").strip()
                strike = float(raw.replace(",", ""))
                if strike <= 0:
                    raise ValueError("Strike price must be positive.")
                break
            except ValueError as e:
                print(f"  Error: {e}")

        # ── Step 3: Time horizon ────────────────────────────────────────────
        while True:
            try:
                raw = input("  Time horizon (e.g. 4h, 2d, 1w, 3m): ").strip()
                T_hours = parse_time(raw)
                break
            except ValueError as e:
                print(f"  Error: {e}")

        # ── Step 4: Fetch + compute ─────────────────────────────────────────
        while True:
            try:
                current_price, prices = fetch_data(ticker)
                break
            except ValueError as e:
                print(f"\n  Error: {e}")
                raw = input("  Try a different ticker (or press Enter to quit): ").strip()
                if not raw:
                    print("\n  Aborted.")
                    return
                ticker = normalize_ticker(raw)

        crypto = is_crypto(ticker)
        periods_per_year = 365 if crypto else 252
        hours_per_year = CALENDAR_HOURS_PER_YEAR if crypto else TRADING_HOURS_PER_YEAR

        mu, sigma = calc_gbm_params(prices, periods_per_year)
        T_years = T_hours / hours_per_year

        print(f"  Running {N_SIMULATIONS:,} simulations...")
        final_prices = run_monte_carlo(current_price, mu, sigma, T_years, N_SIMULATIONS)
        stats = compute_stats(final_prices, strike)

        print_results(ticker, current_price, strike, T_hours, stats, mu, sigma, N_SIMULATIONS)

    except (KeyboardInterrupt, EOFError):
        print("\n  Aborted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
