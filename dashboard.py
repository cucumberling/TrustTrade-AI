from __future__ import annotations
"""
TrustTrade AI — Streamlit Dashboard
Interactive visual interface for the multi-agent trading system.
Supports Chinese / English language switching.

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

# ============================================================
# Language definitions
# ============================================================
LANG = {
    "en": {
        "page_title": "TrustTrade AI",
        "subtitle": "Multi-Agent Trading System",
        "language": "Language",
        "trading_mode": "Trading Mode",
        "trading_pair": "Trading Pair",
        "initial_balance": "Initial Balance ($)",
        "strategy_params": "Strategy Parameters",
        "short_ma": "Short MA Period",
        "long_ma": "Long MA Period",
        "rsi_period": "RSI Period",
        "risk_params": "Risk Parameters",
        "max_position": "Max Position %",
        "max_drawdown": "Max Drawdown %",
        "daily_loss": "Daily Loss Limit %",
        "kraken_connected": "Kraken CLI: Connected",
        "kraken_missing": "Kraken CLI: Not found",
        "run": "Run",
        "reset": "Reset",
        "running": "Running agent loop...",
        "round_progress": "Round {current}/{total}",
        "completed": "Completed {n} rounds!",
        "portfolio_value": "Portfolio Value",
        "realized_pnl": "Realized PnL",
        "win_rate": "Win Rate",
        "sharpe_ratio": "Sharpe Ratio",
        "max_drawdown_label": "Max Drawdown",
        "total_trades": "Total Trades",
        "price_chart": "Price Chart & Trading Signals",
        "price_ma": "Price & Moving Averages",
        "signal_score": "Signal Score",
        "portfolio_value_chart": "Portfolio Value",
        "action_dist": "Action Distribution",
        "strategy_signals": "Strategy Signals",
        "agent_decisions": "Agent Decisions",
        "trade_log": "Trade Log",
        "no_trades": "No trades executed yet.",
        "risk_report": "Risk Report",
        "profit_factor": "Profit Factor",
        "avg_win": "Avg Win",
        "avg_loss": "Avg Loss",
        "sortino_ratio": "Sortino Ratio",
        "erc8004": "ERC-8004 Validation Artifacts",
        "no_artifacts": "No validation artifacts generated.",
        "full_log": "Full Decision Log (all rounds)",
        "empty_state": "Click **Run** in the sidebar to start the trading agent.",
        "how_it_works": "How it works",
        "how_desc": """
1. **Signal Agent** analyzes market data using trend (MA crossover) and momentum (RSI + MACD) strategies
2. **Risk Agent** checks drawdown, daily loss limits, volatility, and consecutive losses
3. **Portfolio Agent** calculates position size with stop-loss and take-profit levels
4. **Manager Agent** makes the final decision and generates ERC-8004 validation artifacts
5. **Execution Agent** routes orders to Kraken CLI (paper/live) or internal simulation

Choose a mode in the sidebar and click **Run** to begin.
""",
        # Agent decision process section
        "agent_process": "Agent Decision Process",
        "select_round": "Select round to inspect",
        "signal_agent": "Signal Agent (Analyst)",
        "risk_agent": "Risk Agent (Risk Manager)",
        "portfolio_agent": "Portfolio Agent (Position Sizing)",
        "manager_agent": "Manager Agent (Final Decision)",
        "execution_agent": "Execution Agent (Trader)",
        "direction": "Direction",
        "combined_signal": "Combined Signal",
        "confidence": "Confidence",
        "reasons": "Reasoning",
        "risk_action": "Risk Decision",
        "risk_score_label": "Risk Score",
        "position_size": "Position Size",
        "stop_loss": "Stop Loss",
        "take_profit": "Take Profit",
        "portfolio_reasoning": "Reasoning",
        "final_decision": "Final Decision",
        "final_reasoning": "Reasoning",
        "artifact_hash": "Artifact Hash",
        "exec_source": "Execution Source",
        "exec_result": "Result",
        "approved": "APPROVED",
        "rejected": "REJECTED",
        "reduced": "REDUCE SIZE",
        "no_data": "No data",
    },
    "zh": {
        "page_title": "TrustTrade AI",
        "subtitle": "多 Agent 自主交易系统",
        "language": "语言",
        "trading_mode": "交易模式",
        "trading_pair": "交易对",
        "initial_balance": "初始资金 ($)",
        "strategy_params": "策略参数",
        "short_ma": "短期均线周期",
        "long_ma": "长期均线周期",
        "rsi_period": "RSI 周期",
        "risk_params": "风控参数",
        "max_position": "最大仓位 %",
        "max_drawdown": "最大回撤 %",
        "daily_loss": "每日亏损限制 %",
        "kraken_connected": "Kraken CLI: 已连接",
        "kraken_missing": "Kraken CLI: 未找到",
        "run": "运行",
        "reset": "重置",
        "running": "正在运行 Agent 循环...",
        "round_progress": "第 {current}/{total} 轮",
        "completed": "完成 {n} 轮交易!",
        "portfolio_value": "投资组合价值",
        "realized_pnl": "已实现盈亏",
        "win_rate": "胜率",
        "sharpe_ratio": "夏普比率",
        "max_drawdown_label": "最大回撤",
        "total_trades": "总交易数",
        "price_chart": "价格图表与交易信号",
        "price_ma": "价格与均线",
        "signal_score": "信号评分",
        "portfolio_value_chart": "投资组合价值",
        "action_dist": "操作分布",
        "strategy_signals": "策略信号对比",
        "agent_decisions": "Agent 决策",
        "trade_log": "交易记录",
        "no_trades": "尚未执行任何交易。",
        "risk_report": "风险报告",
        "profit_factor": "盈利因子",
        "avg_win": "平均盈利",
        "avg_loss": "平均亏损",
        "sortino_ratio": "索提诺比率",
        "erc8004": "ERC-8004 链上验证记录",
        "no_artifacts": "暂无验证记录。",
        "full_log": "完整决策日志（所有轮次）",
        "empty_state": "点击侧边栏的 **运行** 按钮启动交易 Agent。",
        "how_it_works": "工作原理",
        "how_desc": """
1. **Signal Agent（分析师）** 用趋势（均线交叉）和动量（RSI + MACD）策略分析市场数据
2. **Risk Agent（风控经理）** 检查回撤、每日亏损限制、波动率、连续亏损次数
3. **Portfolio Agent（资金管理）** 计算仓位大小，设置止损和止盈价格
4. **Manager Agent（决策者）** 综合所有 Agent 意见做最终决定，生成 ERC-8004 验证产物
5. **Execution Agent（交易员）** 通过 Kraken CLI（模拟/实盘）或内部模拟执行订单

在侧边栏选择模式，点击 **运行** 开始。
""",
        # Agent decision process section
        "agent_process": "Agent 决策过程",
        "select_round": "选择要查看的轮次",
        "signal_agent": "Signal Agent（分析师）",
        "risk_agent": "Risk Agent（风控经理）",
        "portfolio_agent": "Portfolio Agent（资金管理）",
        "manager_agent": "Manager Agent（决策者）",
        "execution_agent": "Execution Agent（交易员）",
        "direction": "方向",
        "combined_signal": "综合信号",
        "confidence": "置信度",
        "reasons": "分析理由",
        "risk_action": "风控决定",
        "risk_score_label": "风险评分",
        "position_size": "仓位比例",
        "stop_loss": "止损价",
        "take_profit": "止盈价",
        "portfolio_reasoning": "分析理由",
        "final_decision": "最终决定",
        "final_reasoning": "决策理由",
        "artifact_hash": "验证哈希",
        "exec_source": "执行来源",
        "exec_result": "执行结果",
        "approved": "通过",
        "rejected": "拒绝",
        "reduced": "减仓",
        "no_data": "无数据",
    },
}


def t(key: str) -> str:
    """Get translated string for current language."""
    lang = st.session_state.get("lang", "zh")
    return LANG[lang].get(key, key)


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

if "lang" not in st.session_state:
    st.session_state.lang = "zh"


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
    st.caption(t("subtitle"))

    # Language switcher
    lang_options = {"中文": "zh", "English": "en"}
    selected_lang = st.selectbox(
        t("language"),
        list(lang_options.keys()),
        index=0 if st.session_state.lang == "zh" else 1,
    )
    new_lang = lang_options[selected_lang]
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.divider()

    # Mode selection
    mode = st.selectbox(t("trading_mode"), ["demo", "paper", "backtest"], index=0, key="k_mode")
    settings.mode = mode

    # Trading pair
    pair = st.selectbox(t("trading_pair"), ["BTC/USD", "ETH/USD"], index=0, key="k_pair")
    settings.trading_pair = pair

    # Balance
    balance = st.number_input(t("initial_balance"), value=10000.0, min_value=100.0, step=1000.0, key="k_balance")
    settings.portfolio.initial_balance = balance

    st.divider()
    st.subheader(t("strategy_params"))

    short_ma = st.slider(t("short_ma"), 3, 20, 5, key="k_short_ma")
    long_ma = st.slider(t("long_ma"), 10, 50, 20, key="k_long_ma")
    rsi_period = st.slider(t("rsi_period"), 5, 30, 14, key="k_rsi")
    settings.strategy.short_ma_period = short_ma
    settings.strategy.long_ma_period = long_ma
    settings.strategy.rsi_period = rsi_period

    st.divider()
    st.subheader(t("risk_params"))

    max_pos = st.slider(t("max_position"), 0.01, 0.30, 0.10, key="k_max_pos")
    max_dd = st.slider(t("max_drawdown"), 0.05, 0.30, 0.15, key="k_max_dd")
    daily_loss = st.slider(t("daily_loss"), 0.01, 0.10, 0.05, key="k_daily_loss")
    settings.risk.max_position_pct = max_pos
    settings.risk.max_drawdown_pct = max_dd
    settings.risk.daily_loss_limit_pct = daily_loss

    st.divider()

    # Kraken CLI status
    kraken = KrakenExecutor()
    cli_available = kraken.is_available()
    if cli_available:
        st.success(t("kraken_connected"))
    else:
        st.warning(t("kraken_missing"))

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button(t("run"), type="primary", use_container_width=True)
    with col2:
        reset_btn = st.button(t("reset"), use_container_width=True)

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

    # Each agent produces reasoning we capture
    signal = signal_agent.analyze(prices)
    risk = risk_agent.evaluate(prices, signal.direction, portfolio, current_price)
    decision = manager.decide(signal, risk, prices, current_price)
    result = executor.execute(decision.trade_intent)

    portfolio_state = portfolio.get_state(current_price)

    round_data = {
        "round": len(st.session_state.round_history) + 1,
        "price": current_price,
        # Signal Agent
        "signal_direction": signal.direction,
        "signal_score": signal.combined_signal,
        "signal_confidence": signal.confidence,
        "trend_signal": signal.strategy_signals.get("trend", 0),
        "momentum_signal": signal.strategy_signals.get("momentum", 0),
        "signal_reasons": signal.reasons,
        # Risk Agent
        "risk_action": risk.action,
        "risk_score": risk.risk_score,
        "risk_reasons": risk.reasons,
        # Portfolio Agent (via manager's decision)
        "stop_loss": decision.trade_intent.stop_loss,
        "take_profit": decision.trade_intent.take_profit,
        "position_pct": decision.trade_intent.position_pct,
        "portfolio_reasoning": decision.trade_intent.reasoning,
        # Manager Agent
        "final_action": decision.trade_intent.direction,
        "final_reasoning": decision.final_reasoning,
        "quantity": decision.trade_intent.quantity,
        # Execution Agent
        "execution": result.message,
        "execution_source": result.execution_source,
        # Portfolio state
        "balance": portfolio_state["balance"],
        "total_value": portfolio_state["total_value"],
        "realized_pnl": portfolio_state["realized_pnl"],
        "drawdown": portfolio_state["drawdown"],
        # ERC-8004
        "artifact_hash": decision.validation_artifact.get("hash", "")[:16],
    }

    st.session_state.round_history.append(round_data)
    return round_data


# --- Collect current params snapshot (string for reliable comparison) ---
current_params = f"{mode}|{pair}|{balance}|{short_ma}|{long_ma}|{rsi_period}|{max_pos:.4f}|{max_dd:.4f}|{daily_loss:.4f}"

# --- Main Content ---
st.title(t("page_title"))


def execute_run():
    """Run the full agent loop with current settings."""
    reset_system()

    if mode == "demo":
        prices_full = get_sample_btc_prices()
    elif mode == "backtest":
        prices_full = generate_price_series(base_price=65000, num_points=200, volatility=0.02, seed=42)
    else:  # paper — try Kraken CLI
        prices_full = kraken_feed.get_price_series(pair)

    st.session_state.price_history = prices_full
    window = 20

    progress = st.progress(0, text=t("running"))
    total_rounds = len(prices_full) - window

    for i in range(window, len(prices_full)):
        price_window = prices_full[max(0, i - window): i + 1]
        run_round(price_window)
        progress.progress(
            (i - window + 1) / total_rounds,
            text=t("round_progress").format(current=i - window + 1, total=total_rounds),
        )

    progress.empty()
    st.session_state.last_params = current_params


# Determine whether to run: button click OR parameter change after first run
params_changed = (
    "last_params" in st.session_state
    and st.session_state.last_params != current_params
    and len(st.session_state.round_history) > 0
)

if run_btn or params_changed:
    execute_run()
    st.success(t("completed").format(n=len(st.session_state.round_history)))

# --- Display Results ---
if st.session_state.round_history:
    df = pd.DataFrame(st.session_state.round_history)
    prices = st.session_state.price_history

    # --- Row 1: Key Metrics ---
    metrics = compute_risk_metrics(st.session_state.portfolio)
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    final_value = df["total_value"].iloc[-1]
    total_return = ((final_value - settings.portfolio.initial_balance) / settings.portfolio.initial_balance) * 100

    col1.metric(t("portfolio_value"), f"${final_value:,.2f}", f"{total_return:+.2f}%")
    col2.metric(t("realized_pnl"), f"${metrics.total_pnl:,.2f}")
    col3.metric(t("win_rate"), f"{metrics.win_rate * 100:.1f}%")
    col4.metric(t("sharpe_ratio"), f"{metrics.sharpe_ratio:.2f}")
    col5.metric(t("max_drawdown_label"), f"{metrics.max_drawdown * 100:.1f}%")
    col6.metric(t("total_trades"), f"{metrics.total_trades}")

    st.divider()

    # ===========================================================
    # AGENT DECISION PROCESS — step-by-step reasoning pipeline
    # ===========================================================
    st.subheader(t("agent_process"))

    selected_round = st.slider(
        t("select_round"), 1, len(st.session_state.round_history),
        value=len(st.session_state.round_history),
    )
    rd = st.session_state.round_history[selected_round - 1]

    # Visual pipeline: 5 agents in a row
    agent_cols = st.columns(5)

    # --- 1. Signal Agent ---
    with agent_cols[0]:
        direction_color = (
            "#4CAF50" if rd["signal_direction"] == "BUY"
            else "#F44336" if rd["signal_direction"] == "SELL"
            else "#9E9E9E"
        )
        st.markdown(f"### {t('signal_agent')}")
        st.markdown(f"**{t('direction')}:** :{('green' if rd['signal_direction'] == 'BUY' else 'red' if rd['signal_direction'] == 'SELL' else 'gray')}[**{rd['signal_direction']}**]")
        st.markdown(f"**{t('combined_signal')}:** `{rd['signal_score']:+.4f}`")
        st.markdown(f"**{t('confidence')}:** `{rd['signal_confidence']:.1%}`")
        st.markdown(f"**{t('reasons')}:**")
        for reason in rd.get("signal_reasons", []):
            st.caption(f"- {reason}")

    # --- 2. Risk Agent ---
    with agent_cols[1]:
        risk_action = rd["risk_action"]
        risk_label = t("approved") if risk_action == "APPROVE" else t("rejected") if risk_action == "REJECT" else t("reduced")
        risk_color = "green" if risk_action == "APPROVE" else "red" if risk_action == "REJECT" else "orange"
        st.markdown(f"### {t('risk_agent')}")
        st.markdown(f"**{t('risk_action')}:** :{risk_color}[**{risk_label}**]")
        st.markdown(f"**{t('risk_score_label')}:** `{rd['risk_score']:.4f}`")
        st.markdown(f"**{t('reasons')}:**")
        for reason in rd.get("risk_reasons", []):
            st.caption(f"- {reason}")

    # --- 3. Portfolio Agent ---
    with agent_cols[2]:
        st.markdown(f"### {t('portfolio_agent')}")
        pct = rd.get("position_pct", 0)
        sl = rd.get("stop_loss")
        tp = rd.get("take_profit")
        st.markdown(f"**{t('position_size')}:** `{pct:.1%}`")
        st.markdown(f"**{t('stop_loss')}:** `${sl:,.2f}`" if sl else f"**{t('stop_loss')}:** —")
        st.markdown(f"**{t('take_profit')}:** `${tp:,.2f}`" if tp else f"**{t('take_profit')}:** —")
        st.markdown(f"**{t('portfolio_reasoning')}:**")
        st.caption(rd.get("portfolio_reasoning", "—"))

    # --- 4. Manager Agent ---
    with agent_cols[3]:
        action = rd["final_action"]
        action_color = "green" if action == "BUY" else "red" if action in ("SELL", "CLOSE") else "gray"
        st.markdown(f"### {t('manager_agent')}")
        st.markdown(f"**{t('final_decision')}:** :{action_color}[**{action}**]")
        st.markdown(f"**{t('final_reasoning')}:**")
        st.caption(rd.get("final_reasoning", "—"))
        ah = rd.get("artifact_hash", "")
        if ah:
            st.markdown(f"**{t('artifact_hash')}:** `{ah}`")

    # --- 5. Execution Agent ---
    with agent_cols[4]:
        st.markdown(f"### {t('execution_agent')}")
        st.markdown(f"**{t('exec_source')}:** `{rd.get('execution_source', '—')}`")
        st.markdown(f"**{t('exec_result')}:**")
        st.caption(rd.get("execution", "—"))

    # Arrow flow visualization
    st.markdown(
        "<div style='text-align:center; font-size:1.2em; color:#888; padding:4px 0;'>"
        "Signal Agent &rarr; Risk Agent &rarr; Portfolio Agent &rarr; Manager Agent &rarr; Execution Agent"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # --- Row 2: Price Chart + Signals ---
    st.subheader(t("price_chart"))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(t("price_ma"), t("signal_score"), t("portfolio_value_chart")),
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

    window_offset = 20

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
        go.Bar(x=df["round"].tolist(), y=df["signal_score"].tolist(), name=t("signal_score"),
               marker_color=["#4CAF50" if s > 0 else "#F44336" for s in df["signal_score"]]),
        row=2, col=1,
    )
    fig.add_hline(y=0.15, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=-0.15, line_dash="dash", line_color="gray", row=2, col=1)

    # Portfolio value
    fig.add_trace(
        go.Scatter(x=df["round"].tolist(), y=df["total_value"].tolist(), name=t("portfolio_value"),
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
    st.subheader(t("agent_decisions"))

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
        fig_pie.update_layout(title=t("action_dist"), height=300)
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
        fig_signals.update_layout(title=t("strategy_signals"), height=300, template="plotly_white")
        st.plotly_chart(fig_signals, use_container_width=True)

    st.divider()

    # --- Row 4: Trade Log ---
    st.subheader(t("trade_log"))

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
            trades_only.style.map(color_action, subset=["Action"]).format({
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
        st.info(t("no_trades"))

    st.divider()

    # --- Row 5: Risk Metrics ---
    st.subheader(t("risk_report"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("profit_factor"), f"{metrics.profit_factor:.2f}")
    col2.metric(t("avg_win"), f"${metrics.avg_win:,.2f}")
    col3.metric(t("avg_loss"), f"${metrics.avg_loss:,.2f}")
    col4.metric(t("sortino_ratio"), f"{metrics.sortino_ratio:.2f}")

    st.divider()

    # --- Row 6: ERC-8004 Validation Log ---
    st.subheader(t("erc8004"))

    artifacts_df = df[df["artifact_hash"] != ""][["round", "final_action", "price", "artifact_hash"]].copy()
    if not artifacts_df.empty:
        artifacts_df.columns = ["Round", "Action", "Price", "Artifact Hash"]
        st.dataframe(
            artifacts_df.style.format({"Price": "${:,.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(t("no_artifacts"))

    # --- Row 7: Full Decision Log (expandable) ---
    with st.expander(t("full_log")):
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
    st.info(t("empty_state"))
    st.markdown(f"### {t('how_it_works')}")
    st.markdown(t("how_desc"))
