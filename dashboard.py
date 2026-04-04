from __future__ import annotations
"""
TrustTrade AI — Streamlit Dashboard
Interactive visual interface for the multi-agent trading system.

Run: streamlit run dashboard.py
"""

import sys
import time
from dataclasses import asdict

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import settings
from agents.signal_agent import SignalAgent
from agents.risk_agent import RiskAgent
from agents.portfolio_agent import PortfolioAgent
from agents.execution_agent import ExecutionAgent
from agents.manager_agent import ManagerAgent
from data.kraken_feed import kraken_feed
from data.mock_data import get_sample_btc_prices, generate_price_series
from portfolio.tracker import PortfolioTracker
from portfolio.risk_metrics import compute_risk_metrics
from execution.kraken_executor import KrakenExecutor
from strategy.trend import compute_ma
from strategy.momentum import compute_rsi

# --- Page Config ---
st.set_page_config(
    page_title="TrustTrade AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
if "portfolio" not in st.session_state:
    st.session_state.portfolio = PortfolioTracker()
    st.session_state.signal_agent = SignalAgent()
    st.session_state.risk_agent = RiskAgent()
    st.session_state.manager = ManagerAgent(st.session_state.portfolio)
    st.session_state.executor = ExecutionAgent(st.session_state.portfolio)
    st.session_state.round_history = []
    st.session_state.price_history = []
    st.session_state.running = False


def reset_system():
    st.session_state.portfolio = PortfolioTracker(initial_balance=settings.portfolio.initial_balance)
    st.session_state.manager = ManagerAgent(st.session_state.portfolio)
    st.session_state.executor = ExecutionAgent(st.session_state.portfolio)
    st.session_state.round_history = []
    st.session_state.price_history = []
    st.session_state.running = False


# --- Sidebar ---
with st.sidebar:
    st.title("TrustTrade AI")
    st.caption("Multi-Agent Trading System")

    st.divider()

    # Mode selection
    mode = st.selectbox("Trading Mode", ["demo", "paper", "backtest"], index=0)
    settings.mode = mode

    # Trading pair
    pair = st.selectbox("Trading Pair", ["BTC/USD", "ETH/USD"], index=0)
    settings.trading_pair = pair

    # Balance
    balance = st.number_input("Initial Balance ($)", value=10000.0, min_value=100.0, step=1000.0)
    settings.portfolio.initial_balance = balance

    st.divider()
    st.subheader("Strategy Parameters")

    settings.strategy.short_ma_period = st.slider("Short MA Period", 3, 20, 5)
    settings.strategy.long_ma_period = st.slider("Long MA Period", 10, 50, 20)
    settings.strategy.rsi_period = st.slider("RSI Period", 5, 30, 14)

    st.divider()
    st.subheader("Risk Parameters")

    settings.risk.max_position_pct = st.slider("Max Position %", 0.01, 0.30, 0.10)
    settings.risk.max_drawdown_pct = st.slider("Max Drawdown %", 0.05, 0.30, 0.15)
    settings.risk.daily_loss_limit_pct = st.slider("Daily Loss Limit %", 0.01, 0.10, 0.05)

    st.divider()

    # Kraken CLI status
    kraken = KrakenExecutor()
    cli_available = kraken.is_available()
    if cli_available:
        st.success("Kraken CLI: Connected")
    else:
        st.warning("Kraken CLI: Not found")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("Run", type="primary", use_container_width=True)
    with col2:
        reset_btn = st.button("Reset", use_container_width=True)

    if reset_btn:
        reset_system()
        st.rerun()


# --- Helper: Run one trading round ---
def run_round(prices: list[float]) -> dict:
    portfolio = st.session_state.portfolio
    signal_agent = st.session_state.signal_agent
    risk_agent = st.session_state.risk_agent
    manager = st.session_state.manager
    executor = st.session_state.executor

    current_price = prices[-1]

    signal = signal_agent.analyze(prices)
    risk = risk_agent.evaluate(prices, signal.direction, portfolio, current_price)
    decision = manager.decide(signal, risk, prices, current_price)
    result = executor.execute(decision.trade_intent)

    portfolio_state = portfolio.get_state(current_price)

    round_data = {
        "round": len(st.session_state.round_history) + 1,
        "price": current_price,
        "signal_direction": signal.direction,
        "signal_score": signal.combined_signal,
        "signal_confidence": signal.confidence,
        "trend_signal": signal.strategy_signals.get("trend", 0),
        "momentum_signal": signal.strategy_signals.get("momentum", 0),
        "risk_action": risk.action,
        "risk_score": risk.risk_score,
        "final_action": decision.trade_intent.direction,
        "quantity": decision.trade_intent.quantity,
        "execution": result.message,
        "execution_source": result.execution_source,
        "balance": portfolio_state["balance"],
        "total_value": portfolio_state["total_value"],
        "realized_pnl": portfolio_state["realized_pnl"],
        "drawdown": portfolio_state["drawdown"],
        "artifact_hash": decision.validation_artifact.get("hash", "")[:16],
    }

    st.session_state.round_history.append(round_data)
    return round_data


# --- Main Content ---
st.title("TrustTrade AI Dashboard")

# Run logic
if run_btn:
    reset_system()

    if mode == "demo":
        prices_full = get_sample_btc_prices()
    elif mode == "backtest":
        prices_full = generate_price_series(base_price=65000, num_points=200, volatility=0.02, seed=42)
    else:  # paper — try Kraken CLI
        prices_full = kraken_feed.get_price_series(pair)

    st.session_state.price_history = prices_full
    window = 20

    progress = st.progress(0, text="Running agent loop...")
    total_rounds = len(prices_full) - window

    for i in range(window, len(prices_full)):
        price_window = prices_full[max(0, i - window): i + 1]
        run_round(price_window)
        progress.progress((i - window + 1) / total_rounds, text=f"Round {i - window + 1}/{total_rounds}")

    progress.empty()
    st.success(f"Completed {total_rounds} rounds!")

# --- Display Results ---
if st.session_state.round_history:
    df = pd.DataFrame(st.session_state.round_history)
    prices = st.session_state.price_history

    # --- Row 1: Key Metrics ---
    metrics = compute_risk_metrics(st.session_state.portfolio)
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    final_value = df["total_value"].iloc[-1]
    total_return = ((final_value - settings.portfolio.initial_balance) / settings.portfolio.initial_balance) * 100

    col1.metric("Portfolio Value", f"${final_value:,.2f}", f"{total_return:+.2f}%")
    col2.metric("Realized PnL", f"${metrics.total_pnl:,.2f}")
    col3.metric("Win Rate", f"{metrics.win_rate * 100:.1f}%")
    col4.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    col5.metric("Max Drawdown", f"{metrics.max_drawdown * 100:.1f}%")
    col6.metric("Total Trades", f"{metrics.total_trades}")

    st.divider()

    # --- Row 2: Price Chart + Signals ---
    st.subheader("Price Chart & Trading Signals")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Price & Moving Averages", "Signal Score", "Portfolio Value"),
    )

    # Price line
    fig.add_trace(
        go.Scatter(x=list(range(len(prices))), y=prices, name="Price",
                   line=dict(color="#2196F3", width=2)),
        row=1, col=1,
    )

    # Moving averages
    if len(prices) >= settings.strategy.long_ma_period:
        short_mas = [compute_ma(prices[:i+1], settings.strategy.short_ma_period)
                     for i in range(len(prices))]
        long_mas = [compute_ma(prices[:i+1], settings.strategy.long_ma_period)
                    for i in range(len(prices))]
        fig.add_trace(
            go.Scatter(x=list(range(len(prices))), y=short_mas,
                       name=f"MA{settings.strategy.short_ma_period}",
                       line=dict(color="#FF9800", width=1, dash="dash")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=list(range(len(prices))), y=long_mas,
                       name=f"MA{settings.strategy.long_ma_period}",
                       line=dict(color="#9C27B0", width=1, dash="dash")),
            row=1, col=1,
        )

    # Buy/Sell markers
    buy_rounds = df[df["final_action"] == "BUY"]
    sell_rounds = df[df["final_action"].isin(["SELL", "CLOSE"])]

    window_offset = 20  # Account for the sliding window start

    if not buy_rounds.empty:
        buy_indices = (buy_rounds["round"] - 1 + window_offset).tolist()
        buy_prices_list = buy_rounds["price"].tolist()
        fig.add_trace(
            go.Scatter(x=buy_indices, y=buy_prices_list, mode="markers",
                       name="BUY", marker=dict(color="#4CAF50", size=12, symbol="triangle-up")),
            row=1, col=1,
        )

    if not sell_rounds.empty:
        sell_indices = (sell_rounds["round"] - 1 + window_offset).tolist()
        sell_prices_list = sell_rounds["price"].tolist()
        fig.add_trace(
            go.Scatter(x=sell_indices, y=sell_prices_list, mode="markers",
                       name="SELL/CLOSE", marker=dict(color="#F44336", size=12, symbol="triangle-down")),
            row=1, col=1,
        )

    # Signal score
    fig.add_trace(
        go.Bar(x=df["round"].tolist(), y=df["signal_score"].tolist(), name="Signal Score",
               marker_color=["#4CAF50" if s > 0 else "#F44336" for s in df["signal_score"]]),
        row=2, col=1,
    )
    fig.add_hline(y=0.15, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=-0.15, line_dash="dash", line_color="gray", row=2, col=1)

    # Portfolio value
    fig.add_trace(
        go.Scatter(x=df["round"].tolist(), y=df["total_value"].tolist(), name="Portfolio Value",
                   line=dict(color="#4CAF50", width=2), fill="tozeroy",
                   fillcolor="rgba(76, 175, 80, 0.1)"),
        row=3, col=1,
    )
    fig.add_hline(y=settings.portfolio.initial_balance, line_dash="dash",
                  line_color="gray", row=3, col=1)

    fig.update_layout(height=700, showlegend=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Row 3: Agent Decision Panel ---
    st.subheader("Agent Decisions")

    col_left, col_right = st.columns(2)

    with col_left:
        # Action breakdown
        action_counts = df["final_action"].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=action_counts.index.tolist(),
            values=action_counts.values.tolist(),
            marker_colors=["#4CAF50" if a == "BUY" else "#F44336" if a in ("SELL", "CLOSE") else "#9E9E9E"
                           for a in action_counts.index],
        )])
        fig_pie.update_layout(title="Action Distribution", height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        # Strategy signals comparison
        fig_signals = go.Figure()
        fig_signals.add_trace(go.Scatter(
            x=df["round"].tolist(), y=df["trend_signal"].tolist(),
            name="Trend", line=dict(color="#FF9800"),
        ))
        fig_signals.add_trace(go.Scatter(
            x=df["round"].tolist(), y=df["momentum_signal"].tolist(),
            name="Momentum", line=dict(color="#2196F3"),
        ))
        fig_signals.update_layout(title="Strategy Signals", height=300, template="plotly_white")
        st.plotly_chart(fig_signals, use_container_width=True)

    st.divider()

    # --- Row 4: Trade Log ---
    st.subheader("Trade Log")

    trades_only = df[df["final_action"] != "HOLD"][
        ["round", "price", "final_action", "quantity", "signal_score",
         "risk_score", "execution_source", "execution", "total_value"]
    ].copy()

    if not trades_only.empty:
        trades_only.columns = ["Round", "Price", "Action", "Qty", "Signal",
                               "Risk Score", "Source", "Result", "Portfolio"]

        def color_action(val):
            if val == "BUY":
                return "background-color: #E8F5E9"
            elif val in ("SELL", "CLOSE"):
                return "background-color: #FFEBEE"
            return ""

        st.dataframe(
            trades_only.style.applymap(color_action, subset=["Action"]).format({
                "Price": "${:,.2f}",
                "Qty": "{:.6f}",
                "Signal": "{:+.4f}",
                "Risk Score": "{:.4f}",
                "Portfolio": "${:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No trades executed yet.")

    st.divider()

    # --- Row 5: Risk Metrics ---
    st.subheader("Risk Report")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Profit Factor", f"{metrics.profit_factor:.2f}")
    col2.metric("Avg Win", f"${metrics.avg_win:,.2f}")
    col3.metric("Avg Loss", f"${metrics.avg_loss:,.2f}")
    col4.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")

    st.divider()

    # --- Row 6: ERC-8004 Validation Log ---
    st.subheader("ERC-8004 Validation Artifacts")

    artifacts_df = df[df["artifact_hash"] != ""][["round", "final_action", "price", "artifact_hash"]].copy()
    if not artifacts_df.empty:
        artifacts_df.columns = ["Round", "Action", "Price", "Artifact Hash"]
        st.dataframe(
            artifacts_df.style.format({"Price": "${:,.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No validation artifacts generated.")

    # --- Row 7: Full Decision Log (expandable) ---
    with st.expander("Full Decision Log (all rounds)"):
        display_df = df[["round", "price", "signal_direction", "signal_score",
                         "risk_action", "risk_score", "final_action", "total_value", "drawdown"]].copy()
        display_df.columns = ["Round", "Price", "Signal", "Score", "Risk", "Risk Score",
                              "Action", "Portfolio", "Drawdown"]
        st.dataframe(
            display_df.style.format({
                "Price": "${:,.2f}",
                "Score": "{:+.4f}",
                "Risk Score": "{:.4f}",
                "Portfolio": "${:,.2f}",
                "Drawdown": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

else:
    # Empty state
    st.info("Click **Run** in the sidebar to start the trading agent.")

    st.markdown("""
    ### How it works

    1. **Signal Agent** analyzes market data using trend (MA crossover) and momentum (RSI + MACD) strategies
    2. **Risk Agent** checks drawdown, daily loss limits, volatility, and consecutive losses
    3. **Portfolio Agent** calculates position size with stop-loss and take-profit levels
    4. **Manager Agent** makes the final decision and generates ERC-8004 validation artifacts
    5. **Execution Agent** routes orders to Kraken CLI (paper/live) or internal simulation

    Choose a mode in the sidebar and click **Run** to begin.
    """)
