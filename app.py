"""
app.py — Streamlit web interface for Monte Carlo price predictor
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from polykalsh_chance import (
    normalize_ticker,
    is_crypto,
    parse_time,
    fetch_data,
    calc_gbm_params,
    run_monte_carlo,
    compute_stats,
    N_SIMULATIONS,
    CALENDAR_HOURS_PER_YEAR,
    TRADING_HOURS_PER_YEAR,
)

st.set_page_config(
    page_title="polykalsh-chance",
    page_icon=None,
    layout="centered",
)

st.markdown("""
<style>
  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Base */
  html, body, [data-testid="stAppViewContainer"] {
    background: #0f0f0f;
    color: #e0e0e0;
  }
  [data-testid="stMain"] { background: #0f0f0f; }

  /* Typography */
  h1 { font-size: 1.1rem !important; font-weight: 600 !important;
       letter-spacing: 0.04em; color: #ffffff; margin-bottom: 0.15rem !important; }
  .subtitle { font-size: 0.78rem; color: #555; margin-bottom: 2rem; }

  /* Inputs */
  [data-testid="stTextInput"] input,
  [data-testid="stNumberInput"] input {
    background: #181818 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #e0e0e0 !important;
    font-size: 0.85rem !important;
  }
  [data-testid="stTextInput"] input:focus,
  [data-testid="stNumberInput"] input:focus {
    border-color: #444 !important;
    box-shadow: none !important;
  }
  label { font-size: 0.72rem !important; color: #666 !important;
          text-transform: uppercase; letter-spacing: 0.06em; }

  /* Button */
  [data-testid="stButton"] button {
    background: #1a1a1a !important;
    border: 1px solid #2e2e2e !important;
    color: #ccc !important;
    border-radius: 4px !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em;
    transition: border-color 0.15s, color 0.15s;
  }
  [data-testid="stButton"] button:hover {
    border-color: #555 !important;
    color: #fff !important;
  }

  /* Metrics */
  [data-testid="stMetric"] {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 4px;
    padding: 0.8rem 1rem;
  }
  [data-testid="stMetricLabel"] { font-size: 0.68rem !important; color: #555 !important;
                                   text-transform: uppercase; letter-spacing: 0.06em; }
  [data-testid="stMetricValue"] { font-size: 1.1rem !important; color: #e0e0e0 !important;
                                   font-weight: 500 !important; }

  /* Prob bars */
  .prob-block { margin-bottom: 1rem; }
  .prob-label { font-size: 0.68rem; color: #555; text-transform: uppercase;
                letter-spacing: 0.06em; margin-bottom: 0.4rem; }
  .prob-value { font-size: 2rem; font-weight: 600; color: #e0e0e0;
                line-height: 1; margin-bottom: 0.5rem; }
  .prob-bar-track { height: 3px; background: #1e1e1e; border-radius: 2px; }
  .prob-bar-fill  { height: 3px; border-radius: 2px; }
  .prob-ci { font-size: 0.7rem; color: #444; margin-top: 0.35rem; }

  /* Section label */
  .section-label { font-size: 0.68rem; color: #444; text-transform: uppercase;
                   letter-spacing: 0.06em; margin: 1.8rem 0 0.8rem; }

  /* Caption */
  .footer-note { font-size: 0.65rem; color: #333; margin-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<h1>polykalsh-chance</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Monte Carlo price probability — GBM, 10k paths, live prices</p>',
    unsafe_allow_html=True,
)

# ── Inputs ─────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1.4, 1])

with col1:
    ticker_input = st.text_input("Ticker", value="BTC", placeholder="BTC, AAPL…")

with col2:
    strike_input = st.number_input(
        "Strike price ($)", min_value=0.0001, value=85000.0, step=100.0, format="%.2f"
    )

with col3:
    time_input = st.text_input("Time horizon", value="4h", placeholder="4h, 2d, 1w…")

run = st.button("Run", use_container_width=True)

# ── Simulation ─────────────────────────────────────────────────────────────────
if run:
    errors = []
    ticker = normalize_ticker(ticker_input) if ticker_input.strip() else ""
    if not ticker:
        errors.append("Enter a ticker symbol.")

    strike = float(strike_input)
    T_hours = None
    try:
        T_hours = parse_time(time_input)
    except ValueError as e:
        errors.append(str(e))

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    with st.spinner(""):
        try:
            current_price, prices = fetch_data(ticker)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        crypto = is_crypto(ticker)
        periods_per_year = 365 if crypto else 252
        hours_per_year = CALENDAR_HOURS_PER_YEAR if crypto else TRADING_HOURS_PER_YEAR

        try:
            mu, sigma = calc_gbm_params(prices, periods_per_year)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        T_years = T_hours / hours_per_year
        final_prices = run_monte_carlo(current_price, mu, sigma, T_years, N_SIMULATIONS)
        stats = compute_stats(final_prices, strike)

    # Time label
    if T_hours < 24:
        time_label = f"{T_hours:g}h"
    elif T_hours < 168:
        time_label = f"{T_hours/24:g}d"
    elif T_hours < 720:
        time_label = f"{T_hours/168:g}w"
    else:
        time_label = f"{T_hours/720:g}mo"

    # ── Live stats ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Market</p>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Price", f"${current_price:,.2f}")
    m2.metric("Strike", f"${strike:,.2f}")
    m3.metric("Horizon", time_label)

    # ── Probability display ─────────────────────────────────────────────────────
    p_above = stats["p_above"]
    p_below = stats["p_below"]
    ci_a_lo, ci_a_hi = stats["ci_above"]
    ci_b_lo, ci_b_hi = stats["ci_below"]

    st.markdown('<p class="section-label">Probability</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        <div class="prob-block">
          <div class="prob-label">Above ${strike:,.0f}</div>
          <div class="prob-value">{p_above*100:.1f}%</div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width:{p_above*100:.1f}%;background:#4a9eff;"></div>
          </div>
          <div class="prob-ci">95% CI &nbsp;{ci_a_lo*100:.1f}% – {ci_a_hi*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="prob-block">
          <div class="prob-label">Below ${strike:,.0f}</div>
          <div class="prob-value">{p_below*100:.1f}%</div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width:{p_below*100:.1f}%;background:#e05c5c;"></div>
          </div>
          <div class="prob-ci">95% CI &nbsp;{ci_b_lo*100:.1f}% – {ci_b_hi*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Model params ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Model</p>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    p1.metric("Volatility (ann.)", f"{sigma*100:.1f}%")
    sign = "+" if mu >= 0 else ""
    p2.metric("Drift (ann.)", f"{sign}{mu*100:.1f}%")

    # ── Histogram ───────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Distribution</p>', unsafe_allow_html=True)

    clip_hi = np.percentile(final_prices, 99.5)
    clip_lo = np.percentile(final_prices, 0.5)
    display_prices = final_prices[(final_prices >= clip_lo) & (final_prices <= clip_hi)]
    above_mask = display_prices > strike

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=display_prices[~above_mask], nbinsx=80,
        name=f"Below", marker_color="rgba(224,92,92,0.55)", showlegend=True,
    ))
    fig.add_trace(go.Histogram(
        x=display_prices[above_mask], nbinsx=80,
        name=f"Above", marker_color="rgba(74,158,255,0.55)", showlegend=True,
    ))
    fig.add_vline(
        x=strike, line_dash="dash", line_color="rgba(255,255,255,0.25)",
        annotation_text=f"strike", annotation_position="top right",
        annotation_font_color="rgba(255,255,255,0.3)", annotation_font_size=11,
    )
    fig.add_vline(
        x=current_price, line_dash="dot", line_color="rgba(255,255,255,0.15)",
        annotation_text=f"now", annotation_position="top left",
        annotation_font_color="rgba(255,255,255,0.2)", annotation_font_size=11,
    )
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Final price ($)",
        yaxis_title="Paths",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11, color="#555")),
        margin=dict(t=30, b=40, l=0, r=0),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111111",
        font=dict(color="#555", size=11),
        xaxis=dict(gridcolor="#1a1a1a", linecolor="#1a1a1a", tickfont=dict(color="#444")),
        yaxis=dict(gridcolor="#1a1a1a", linecolor="#1a1a1a", tickfont=dict(color="#444")),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Footer ──────────────────────────────────────────────────────────────────
    asset_type = "crypto · 365-day year" if crypto else "equity · 252-day year"
    st.markdown(
        f'<p class="footer-note">{N_SIMULATIONS:,} simulations · {asset_type} · '
        f'90-day lookback · not financial advice</p>',
        unsafe_allow_html=True,
    )
