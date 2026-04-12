from __future__ import annotations

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
from onchain.identity import IdentityManager, create_agent_registration_json
from onchain.reputation import ReputationManager
from onchain.validation import ValidationManager

# ============================================================
# Language definitions
# ============================================================
LANG = {
    "en": {
        "page_title": "TrustTrade AI",
        "subtitle": "Multi-Agent Trading System",
        "language": "Language",
        "trading_mode": "Trading Mode",
        "style": "Style",
        "trading_pair": "Trading Pair",
        "initial_balance": "Initial Balance ($)",
        "strategy_params": "Strategy Parameters",
        "short_ma": "Short MA Period",
        "long_ma": "Long MA Period",
        "rsi_period": "RSI Period",
        "risk_params": "Risk Parameters",
        "max_position": "Max Margin %",
        "max_drawdown": "Max Drawdown %",
        "daily_loss": "Daily Loss Limit %",
        "leverage": "Leverage",
        "leverage_help": "Kraken Margin: 1x = spot, up to 5x on BTC/USD",
        "kraken_connected": "Kraken CLI: Connected",
        "kraken_missing": "Kraken CLI: Not found",
        "run": "Run",
        "reset": "Reset",
        "running": "Running agent loop...",
        "round_progress": "Round {current}/{total}",
        "completed": "Completed {n} rounds!",
        "portfolio_value": "Portfolio Value",
        "realized_pnl": "Realized PnL",
        "total_pnl": "Total PnL",
        "unrealized": "Unrealized",
        "fees": "Fees",
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
        "empty_state": "Click **Run** above to start the trading agent.",
        "how_it_works": "How it works",
        "how_desc": """
1. **Signal Agent** analyzes market data using trend (MA crossover) and momentum (RSI + MACD) strategies
2. **Risk Agent** checks drawdown, daily loss limits, volatility, and consecutive losses
3. **Portfolio Agent** calculates position size with stop-loss and take-profit levels
4. **Manager Agent** makes the final decision and generates ERC-8004 validation artifacts
5. **Execution Agent** routes orders to Kraken CLI (paper/live) or internal simulation

Choose a mode and click **Run** to begin.
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
        # Fundamental agents
        "fundamentals_section": "Fundamental Context (live snapshot)",
        "use_fundamentals": "Fundamental agents",
        "use_llm_manager": "LLM Manager (Claude)",
        "funding_agent": "Funding Rate Agent",
        "sentiment_agent": "Sentiment Agent",
        "onchain_agent": "On-Chain Agent",
        "funding_w": "Funding weight",
        "sentiment_w": "Sentiment weight",
        "onchain_w": "On-chain weight",
        "trend_w": "Trend weight",
        "momentum_w": "Momentum weight",
        "hold_thr": "HOLD threshold",
        "round_breakdown": "Round Breakdown",
        "risk_rejected": "Risk Rejected",
        "executed": "Executed",
        "full_cycles": "Full Cycles",
        "open_close": "open→close",
        "risk_per_trade": "Risk / trade",
    },
    "zh": {
        "page_title": "TrustTrade AI",
        "subtitle": "多 Agent 自主交易系统",
        "language": "语言",
        "trading_mode": "交易模式",
        "style": "风格",
        "trading_pair": "交易对",
        "initial_balance": "初始资金 ($)",
        "strategy_params": "策略参数",
        "short_ma": "短期均线周期",
        "long_ma": "长期均线周期",
        "rsi_period": "RSI 周期",
        "risk_params": "风控参数",
        "max_position": "最大保证金 %",
        "max_drawdown": "最大回撤 %",
        "daily_loss": "每日亏损限制 %",
        "leverage": "杠杆倍数",
        "leverage_help": "Kraken 保证金交易：1x=现货，BTC/USD 最高 5x",
        "kraken_connected": "Kraken CLI: 已连接",
        "kraken_missing": "Kraken CLI: 未找到",
        "run": "运行",
        "reset": "重置",
        "running": "正在运行 Agent 循环...",
        "round_progress": "第 {current}/{total} 轮",
        "completed": "完成 {n} 轮交易!",
        "portfolio_value": "投资组合价值",
        "realized_pnl": "已实现盈亏",
        "total_pnl": "总盈亏",
        "unrealized": "未实现",
        "fees": "手续费",
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
        "empty_state": "点击上方 **运行** 按钮启动交易 Agent。",
        "how_it_works": "工作原理",
        "how_desc": """
1. **Signal Agent（分析师）** 用趋势（均线交叉）和动量（RSI + MACD）策略分析市场数据
2. **Risk Agent（风控经理）** 检查回撤、每日亏损限制、波动率、连续亏损次数
3. **Portfolio Agent（资金管理）** 计算仓位大小，设置止损和止盈价格
4. **Manager Agent（决策者）** 综合所有 Agent 意见做最终决定，生成 ERC-8004 验证产物
5. **Execution Agent（交易员）** 通过 Kraken CLI（模拟/实盘）或内部模拟执行订单

选择模式，点击 **运行** 开始。
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
        # Fundamental agents
        "fundamentals_section": "基本面 Agent 实时快照",
        "use_fundamentals": "启用基本面 Agent",
        "use_llm_manager": "LLM 决策官 (Claude)",
        "funding_agent": "资金费率 Agent",
        "sentiment_agent": "情绪 Agent",
        "onchain_agent": "链上数据 Agent",
        "funding_w": "资金费率权重",
        "sentiment_w": "情绪权重",
        "onchain_w": "链上权重",
        "trend_w": "趋势权重",
        "momentum_w": "动量权重",
        "hold_thr": "HOLD 阈值",
        "round_breakdown": "轮决策拆解",
        "risk_rejected": "风控拒绝",
        "executed": "实际成交",
        "full_cycles": "完整周期",
        "open_close": "开仓→平仓",
        "risk_per_trade": "单笔风险",
    },
}


def t(key: str) -> str:
    """Get translated string for current language."""
    lang = st.session_state.get("lang", "en")
    return LANG[lang].get(key, key)


# --- Page Config ---
st.set_page_config(
    page_title="TrustTrade AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Session State Initialization ---
if "portfolio" not in st.session_state:
    st.session_state.portfolio = PortfolioTracker()
    st.session_state.signal_agent = SignalAgent()
    st.session_state.risk_agent = RiskAgent()
    st.session_state.manager = ManagerAgent(st.session_state.portfolio)
    st.session_state.executor = ExecutionAgent(st.session_state.portfolio)
    # ERC-8004 managers
    st.session_state.identity = IdentityManager()
    st.session_state.validation = ValidationManager()
    st.session_state.reputation = ReputationManager()
    st.session_state.agent_id = None
    st.session_state.round_history = []
    st.session_state.price_history = []
    st.session_state.running = False

if "lang" not in st.session_state:
    st.session_state.lang = "en"


def reset_system():
    st.session_state.portfolio = PortfolioTracker()
    st.session_state.signal_agent = SignalAgent()
    st.session_state.risk_agent = RiskAgent()
    st.session_state.manager = ManagerAgent(st.session_state.portfolio)
    st.session_state.executor = ExecutionAgent(st.session_state.portfolio)
    st.session_state.identity = IdentityManager()
    st.session_state.validation = ValidationManager()
    st.session_state.reputation = ReputationManager()
    st.session_state.agent_id = None
    st.session_state.round_history = []
    st.session_state.price_history = []
    st.session_state.running = False


# ============================================================
# TOP BAR — Title + Core Controls (one line)
# ============================================================
top_title, top_mode, top_style, top_pair, top_balance, top_run, top_reset, top_lang = st.columns(
    [1.8, 1.2, 1.2, 1.0, 1.2, 0.7, 0.7, 0.7]
)

with top_title:
    st.markdown("### TrustTrade AI")

with top_mode:
    mode = st.selectbox(t("trading_mode"), ["paper (Kraken)"], index=0, key="k_mode", label_visibility="collapsed")
    mode = "paper"
    settings.mode = mode

with top_style:
    style_labels = {
        "en": ["balanced", "aggressive", "conservative", "trend", "reversal"],
        "zh": ["均衡", "激进", "保守", "趋势跟踪", "均值回归"],
    }
    style = st.selectbox(
        t("style"), style_labels[st.session_state.lang],
        index=0, key="k_style", label_visibility="collapsed",
    )
    # Map localized label back to key
    style_key = style_labels["en"][style_labels[st.session_state.lang].index(style)]

with top_pair:
    pair = st.selectbox(t("trading_pair"), ["BTC/USD", "ETH/USD"], index=0, key="k_pair", label_visibility="collapsed")
    settings.trading_pair = pair

with top_balance:
    balance = st.number_input(t("initial_balance"), value=10000.0, min_value=100.0, step=1000.0, key="k_balance", label_visibility="collapsed")
    settings.portfolio.initial_balance = balance

with top_run:
    run_btn = st.button(t("run"), type="primary", use_container_width=True)

with top_reset:
    reset_btn = st.button(t("reset"), use_container_width=True)
    if reset_btn:
        reset_system()
        st.rerun()

with top_lang:
    lang_options = {"EN": "en", "中文": "zh"}
    selected_lang = st.selectbox(
        t("language"),
        list(lang_options.keys()),
        index=0 if st.session_state.lang == "en" else 1,
        label_visibility="collapsed",
    )
    new_lang = lang_options[selected_lang]
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

# ============================================================
# Style presets — apply before widgets render
# ============================================================
STYLE_PRESETS = {
    # margin = % of equity used as collateral. With leverage L, notional = margin * L.
    # Industry-standard discretionary risk: 1-2% per trade. We use 10-20% margin × leverage
    # so the engine reflects realistic crypto-margin trading on Kraken (max 5x BTC/USD).
    # NOTE: all presets require trend+momentum agreement (the filter in SignalAgent).
    # Shorter MAs need higher hold_thr to avoid whipsaw in ranging markets.
    "balanced":     {"short_ma": 20, "long_ma": 50, "rsi": 14, "max_pos": 50, "max_dd": 25, "daily_loss": 10,
                     "trend_w": 0.6, "momentum_w": 0.4, "hold_thr": 0.06, "risk_per_trade": 0.10, "leverage": 3.0},
    "aggressive":   {"short_ma": 12, "long_ma": 30, "rsi": 10, "max_pos": 60, "max_dd": 35, "daily_loss": 20,
                     "trend_w": 0.5, "momentum_w": 0.5, "hold_thr": 0.06, "risk_per_trade": 0.12, "leverage": 5.0},
    "conservative": {"short_ma": 20, "long_ma": 50, "rsi": 14, "max_pos": 40, "max_dd": 15, "daily_loss": 8,
                     "trend_w": 0.6, "momentum_w": 0.4, "hold_thr": 0.08, "risk_per_trade": 0.05, "leverage": 1.0},
    "trend":        {"short_ma": 15, "long_ma": 40, "rsi": 14, "max_pos": 50, "max_dd": 25, "daily_loss": 12,
                     "trend_w": 0.7, "momentum_w": 0.3, "hold_thr": 0.06, "risk_per_trade": 0.10, "leverage": 3.0},
    "reversal":     {"short_ma": 20, "long_ma": 50, "rsi": 10, "max_pos": 40, "max_dd": 20, "daily_loss": 10,
                     "trend_w": 0.3, "momentum_w": 0.7, "hold_thr": 0.06, "risk_per_trade": 0.08, "leverage": 2.0},
}

# Apply preset — always sync widget keys to selected style on change
if "prev_style" not in st.session_state:
    st.session_state.prev_style = ""
if style_key != st.session_state.prev_style:
    p = STYLE_PRESETS[style_key]
    # Force-update all widget session_state keys
    for k, v in [
        ("k_short_ma", p["short_ma"]), ("k_long_ma", p["long_ma"]),
        ("k_rsi", p["rsi"]), ("k_max_pos", p["max_pos"]),
        ("k_max_dd", p["max_dd"]), ("k_daily_loss", p["daily_loss"]),
        ("k_leverage", float(p["leverage"])),
    ]:
        st.session_state[k] = v
    st.session_state.prev_style = style_key
    st.rerun()

# Always apply non-widget params from current style
_p = STYLE_PRESETS[style_key]
settings.strategy.signal_weights = {"trend": _p["trend_w"], "momentum": _p["momentum_w"]}
settings.portfolio.fixed_fraction = _p["risk_per_trade"]
settings.portfolio.leverage = _p["leverage"]

# ============================================================
# SECOND ROW — Strategy & Risk params (expandable)
# ============================================================
with st.expander(f"⚙ {t('strategy_params')} / {t('risk_params')}", expanded=False):
    p_col1, p_col2, p_col3, p_col4, p_col5, p_col6, p_col7, p_col8 = st.columns(8)

    with p_col1:
        short_ma = st.number_input(t("short_ma"), min_value=3, max_value=30, value=12, key="k_short_ma")
        settings.strategy.short_ma_period = short_ma

    with p_col2:
        long_ma = st.number_input(t("long_ma"), min_value=10, max_value=100, value=26, key="k_long_ma")
        settings.strategy.long_ma_period = long_ma

    with p_col3:
        rsi_period = st.number_input(t("rsi_period"), min_value=5, max_value=30, value=14, key="k_rsi")
        settings.strategy.rsi_period = rsi_period

    with p_col4:
        max_pos = st.number_input(t("max_position"), min_value=10, max_value=90, value=50, step=10, key="k_max_pos")
        settings.risk.max_position_pct = max_pos / 100.0

    with p_col5:
        max_dd = st.number_input(t("max_drawdown"), min_value=5, max_value=50, value=25, step=5, key="k_max_dd")
        settings.risk.max_drawdown_pct = max_dd / 100.0

    with p_col6:
        daily_loss = st.number_input(t("daily_loss"), min_value=1, max_value=20, value=10, step=1, key="k_daily_loss")
        settings.risk.daily_loss_limit_pct = daily_loss / 100.0

    with p_col7:
        leverage = st.number_input(
            t("leverage"), min_value=1.0, max_value=5.0, value=float(_p["leverage"]),
            step=1.0, key="k_leverage", help=t("leverage_help"),
        )
        settings.portfolio.leverage = leverage

    with p_col8:
        kraken = KrakenExecutor()
        cli_available = kraken.is_available()
        if cli_available:
            st.success(t("kraken_connected"))
        else:
            st.warning(t("kraken_missing"))


# ============================================================
# THIRD ROW — Fundamental agents & LLM Manager
# ============================================================
with st.expander(f"🌐 {t('fundamentals_section')}", expanded=False):
    f_col1, f_col2, f_col3, f_col4, f_col5, f_col6, f_col7 = st.columns(7)

    with f_col1:
        use_fund = st.toggle(t("use_fundamentals"), value=True, key="k_use_fund")
        settings.strategy.use_fundamentals = use_fund

    with f_col2:
        use_llm_mgr = st.toggle(
            t("use_llm_manager"),
            value=False,
            key="k_use_llm_mgr",
            help="Calls Claude on every BUY/SELL to confirm/veto/reduce. Requires ANTHROPIC_API_KEY.",
        )
        settings.strategy.use_llm_manager = use_llm_mgr

    with f_col3:
        w_trend = st.number_input(t("trend_w"), value=float(_p["trend_w"]) * 0.7, min_value=0.0, max_value=1.0, step=0.05, key="k_w_trend", format="%.2f")
    with f_col4:
        w_mom = st.number_input(t("momentum_w"), value=float(_p["momentum_w"]) * 0.7, min_value=0.0, max_value=1.0, step=0.05, key="k_w_mom", format="%.2f")
    with f_col5:
        w_fund = st.number_input(t("funding_w"), value=0.10, min_value=0.0, max_value=1.0, step=0.05, key="k_w_fund", format="%.2f")
    with f_col6:
        w_sent = st.number_input(t("sentiment_w"), value=0.10, min_value=0.0, max_value=1.0, step=0.05, key="k_w_sent", format="%.2f")
    with f_col7:
        w_chain = st.number_input(t("onchain_w"), value=0.10, min_value=0.0, max_value=1.0, step=0.05, key="k_w_chain", format="%.2f")

    # Apply custom weights — overrides preset
    settings.strategy.signal_weights = {
        "trend": w_trend,
        "momentum": w_mom,
        "funding": w_fund,
        "sentiment": w_sent,
        "onchain": w_chain,
    }


# --- Helper: Run one trading round ---
def run_round(prices: list[float]) -> dict:
    portfolio = st.session_state.portfolio
    signal_agent = st.session_state.signal_agent
    risk_agent = st.session_state.risk_agent
    manager = st.session_state.manager
    executor = st.session_state.executor

    # LOOKAHEAD-BIAS FIX:
    # Signals are generated from CLOSED bars only (prices[:-1]).
    # Execution then happens at prices[-1], which represents the *next* bar's
    # available price (the open of the bar after the signal closes).
    # This prevents the unrealistic "decide and fill at the same close" pattern.
    if len(prices) < 2:
        return {}
    signal_prices = prices[:-1]
    current_price = prices[-1]

    signal = signal_agent.analyze(signal_prices)
    risk = risk_agent.evaluate(signal_prices, signal.direction, portfolio, current_price)
    decision = manager.decide(signal, risk, signal_prices, current_price)
    result = executor.execute(decision.trade_intent)

    portfolio_state = portfolio.get_state(current_price)

    # ── ERC-8004: Create validation artifact for every non-HOLD trade ──
    validation_artifact = None
    if decision.trade_intent.direction != "HOLD":
        validation_artifact = st.session_state.validation.create_trade_intent_artifact(
            pair=decision.trade_intent.pair,
            direction=decision.trade_intent.direction,
            quantity=decision.trade_intent.quantity,
            price=current_price,
            agent_id=st.session_state.agent_id or 0,
            risk_score=risk.risk_score,
            decision_data={
                "signal_score": signal.combined_signal,
                "risk_action": risk.action,
                "portfolio_state": portfolio_state,
            },
        )
        st.session_state.validation.submit_validation_request(
            agent_id=st.session_state.agent_id or 0,
            artifact=validation_artifact,
        )

    # ── ERC-8004: Submit reputation feedback when a trade closes with PnL ──
    if decision.trade_intent.direction == "CLOSE" and result.success:
        closed = [t for t in portfolio.trade_history if t.action == "CLOSE"]
        if closed:
            last_pnl = closed[-1].pnl
            yield_pct = (last_pnl / portfolio.initial_balance) * 100
            st.session_state.reputation.submit_trading_yield(
                agent_id=st.session_state.agent_id or 0,
                yield_pct=yield_pct,
                period="trade",
            )

    round_data = {
        "round": len(st.session_state.round_history) + 1,
        "price": current_price,
        # Signal Agent
        "signal_direction": signal.direction,
        "signal_score": signal.combined_signal,
        "signal_confidence": signal.confidence,
        "trend_signal": signal.strategy_signals.get("trend", 0),
        "momentum_signal": signal.strategy_signals.get("momentum", 0),
        "funding_signal": signal.strategy_signals.get("funding", 0),
        "sentiment_signal": signal.strategy_signals.get("sentiment", 0),
        "onchain_signal": signal.strategy_signals.get("onchain", 0),
        "fundamentals": signal.fundamentals,
        "signal_reasons": signal.reasons,
        # LLM Manager
        "llm_used": getattr(decision, "llm_used", False),
        "llm_verdict": getattr(decision, "llm_verdict", None),
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


def execute_run():
    """Run the full agent loop with current settings."""
    # Recreate all agents with fresh state
    st.session_state.portfolio = PortfolioTracker(initial_balance=balance)
    _sa = SignalAgent()
    _sa.hold_threshold = _p["hold_thr"]
    # Snapshot fundamental agents ONCE per run (slow-moving signals)
    if settings.strategy.use_fundamentals:
        with st.spinner("Fetching funding / sentiment / on-chain snapshot..."):
            _sa.refresh_fundamentals(pair)
    st.session_state.signal_agent = _sa
    st.session_state.risk_agent = RiskAgent()
    st.session_state.manager = ManagerAgent(st.session_state.portfolio)
    st.session_state.executor = ExecutionAgent(st.session_state.portfolio)
    # Reset ERC-8004 session state
    st.session_state.identity = IdentityManager()
    st.session_state.validation = ValidationManager()
    st.session_state.reputation = ReputationManager()
    st.session_state.round_history = []
    st.session_state.price_history = []
    st.session_state.fundamentals_snapshot = {
        "funding": _sa._fund_snapshot,
        "sentiment": _sa._sent_snapshot,
        "onchain": _sa._chain_snapshot,
    }

    # ── ERC-8004: Register agent identity before trading starts ──
    with st.spinner("Registering agent on ERC-8004 Identity Registry..."):
        registration_json = create_agent_registration_json()
        st.session_state.agent_id = st.session_state.identity.register_agent(
            agent_uri="https://agent.trusttrade.ai/registration.json"
        )
        st.session_state.agent_registration = registration_json

    prices_full = kraken_feed.get_price_series(pair, count=720)

    st.session_state.price_history = prices_full
    # Window must be large enough to feed the longest MA period meaningfully.
    # +10 buffer ensures stdev/returns calc has enough samples.
    window = max(60, settings.strategy.long_ma_period + 10)

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

    # ── ERC-8004: Submit session-level reputation feedback (success rate + total yield) ──
    portfolio = st.session_state.portfolio
    final_price = prices_full[-1]
    metrics = compute_risk_metrics(portfolio)
    total_yield_pct = ((portfolio.total_value(final_price) - balance) / balance) * 100
    st.session_state.reputation.submit_trading_yield(
        agent_id=st.session_state.agent_id or 0,
        yield_pct=total_yield_pct,
        period="session",
    )
    st.session_state.reputation.submit_success_rate(
        agent_id=st.session_state.agent_id or 0,
        win_rate=metrics.win_rate,
        period="session",
    )


if run_btn:
    execute_run()
    st.success(t("completed").format(n=len(st.session_state.round_history)))

# --- Display Results ---
if st.session_state.round_history:
    df = pd.DataFrame(st.session_state.round_history)
    prices = st.session_state.price_history

    # --- Row 1: Key Metrics (4 core) ---
    metrics = compute_risk_metrics(st.session_state.portfolio)

    final_value = df["total_value"].iloc[-1]
    total_return = ((final_value - balance) / balance) * 100
    pnl_color = "#4CAF50" if metrics.total_pnl >= 0 else "#F44336"

    # True total PnL = current value - initial capital (includes all fees)
    total_pnl = final_value - balance
    fees_paid = st.session_state.portfolio.total_fees

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(t("portfolio_value"), f"${final_value:,.2f}", f"{total_return:+.2f}%")
    col2.metric(t("total_pnl"), f"${total_pnl:,.2f}",
                f"{t('fees')}: ${fees_paid:,.2f}" if fees_paid > 0 else None)
    col3.metric(t("realized_pnl"), f"${metrics.total_pnl:,.2f}")
    col4.metric(t("win_rate"), f"{metrics.win_rate * 100:.1f}%")
    col5.metric(t("total_trades"), f"{metrics.total_trades}")

    # ===========================================================
    # ROUND BREAKDOWN — explain where the 600+ rounds went
    # ===========================================================
    total_rounds = len(df)
    n_buy   = int((df["final_action"] == "BUY").sum())
    n_sell  = int((df["final_action"] == "SELL").sum())
    n_close = int((df["final_action"] == "CLOSE").sum())
    n_hold  = int((df["final_action"] == "HOLD").sum())
    n_risk_reject = int((df["risk_action"] == "REJECT").sum())

    n_exec = n_buy + n_sell + n_close
    st.caption(
        f"📊 **{total_rounds} {t('round_breakdown')}** | "
        f"BUY: **{n_buy}** · SELL: **{n_sell}** · CLOSE: **{n_close}** · "
        f"HOLD: **{n_hold}** ({n_hold/total_rounds*100:.0f}%)  ·  "
        f"{t('risk_rejected')}: **{n_risk_reject}**  ·  "
        f"{t('executed')}: **{n_exec}**  ·  "
        f"{t('full_cycles')}: **{n_close}** ({t('open_close')})"
    )

    st.divider()

    # ===========================================================
    # FUNDAMENTAL CONTEXT PANEL — funding / sentiment / on-chain
    # ===========================================================
    fund_snap = st.session_state.get("fundamentals_snapshot", {})
    if fund_snap and any(fund_snap.values()):
        st.subheader(f"🌐 {t('fundamentals_section')}")
        fc1, fc2, fc3 = st.columns(3)

        # Funding
        with fc1:
            f = fund_snap.get("funding", {}) or {}
            sig = f.get("signal", 0.0)
            color = "#00C853" if sig > 0.05 else "#FF1744" if sig < -0.05 else "#9E9E9E"
            st.markdown(f"#### 💰 {t('funding_agent')}")
            st.markdown(
                f"<div style='font-size:28px; font-weight:bold; color:{color};'>{sig:+.3f}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"**Regime:** {f.get('regime', '—')}")
            st.caption(f"**24h avg:** {f.get('avg_24h_pct', 0):.4f}% / 8h")
            st.caption(f.get("reason", ""))

        # Sentiment
        with fc2:
            s = fund_snap.get("sentiment", {}) or {}
            sig = s.get("signal", 0.0)
            color = "#00C853" if sig > 0.05 else "#FF1744" if sig < -0.05 else "#9E9E9E"
            st.markdown(f"#### 📰 {t('sentiment_agent')}")
            st.markdown(
                f"<div style='font-size:28px; font-weight:bold; color:{color};'>{sig:+.3f}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"**Fear & Greed:** {s.get('fear_greed', '—')}")
            st.caption(f"**Headlines:** {s.get('headline_count', 0)} ({'LLM' if s.get('used_llm') else 'keyword'})")
            st.caption(s.get("reason", ""))

        # On-chain
        with fc3:
            c = fund_snap.get("onchain", {}) or {}
            sig = c.get("signal", 0.0)
            color = "#00C853" if sig > 0.05 else "#FF1744" if sig < -0.05 else "#9E9E9E"
            st.markdown(f"#### ⛓ {t('onchain_agent')}")
            st.markdown(
                f"<div style='font-size:28px; font-weight:bold; color:{color};'>{sig:+.3f}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"**Mempool:** {c.get('mempool_tx', 0):,} tx")
            st.caption(f"**Fee:** {c.get('fee_sat_vb', 0)} sat/vB · **Hashrate Δ:** {c.get('hashrate_pct', 0):+.1f}%")
            st.caption(c.get("reason", ""))

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
        if rd.get("llm_used"):
            st.markdown("**🤖 Claude verdict:**")
            st.caption(rd.get("llm_verdict", "—") or "—")
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
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(t("price_ma"), t("portfolio_value_chart")),
    )

    # Price line with gradient fill (use tonexty with a baseline trace for proper auto-range)
    price_min = min(prices)
    price_max = max(prices)
    price_pad = max((price_max - price_min) * 0.15, price_max * 0.002)
    # Invisible baseline so fill doesn't go to zero
    fig.add_trace(
        go.Scatter(x=list(range(len(prices))), y=[price_min - price_pad]*len(prices),
                   mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(prices))), y=prices, name="Price",
                   line=dict(color="#2962FF", width=2.5),
                   fill="tonexty", fillcolor="rgba(41,98,255,0.06)"),
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
                       line=dict(color="#FF6D00", width=1.5, dash="dash")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=list(range(len(prices))), y=long_mas,
                       name=f"MA{settings.strategy.long_ma_period}",
                       line=dict(color="#AA00FF", width=1.5, dash="dash")),
            row=1, col=1,
        )

    # Buy/Sell markers
    buy_rounds = df[df["final_action"] == "BUY"]
    sell_rounds = df[df["final_action"].isin(["SELL", "CLOSE"])]

    # The trading window used during execution; round 1 starts at prices[window]
    window_offset = max(60, settings.strategy.long_ma_period + 10)

    if not buy_rounds.empty:
        buy_indices = (buy_rounds["round"] - 1 + window_offset).tolist()
        # Use the actual price from the prices array to ensure markers sit exactly on the line
        buy_prices_list = [prices[int(idx)] if int(idx) < len(prices) else buy_rounds.iloc[i]["price"]
                           for i, idx in enumerate(buy_indices)]
        fig.add_trace(
            go.Scatter(x=buy_indices, y=buy_prices_list, mode="markers",
                       name="BUY", marker=dict(color="#00C853", size=14, symbol="triangle-up",
                                               line=dict(width=1, color="#fff"))),
            row=1, col=1,
        )

    if not sell_rounds.empty:
        sell_indices = (sell_rounds["round"] - 1 + window_offset).tolist()
        sell_prices_list = [prices[int(idx)] if int(idx) < len(prices) else sell_rounds.iloc[i]["price"]
                            for i, idx in enumerate(sell_indices)]
        fig.add_trace(
            go.Scatter(x=sell_indices, y=sell_prices_list, mode="markers",
                       name="SELL/CLOSE", marker=dict(color="#FF1744", size=14, symbol="triangle-down",
                                                      line=dict(width=1, color="#fff"))),
            row=1, col=1,
        )

    # Portfolio value with color based on profit/loss
    total_values = df["total_value"].tolist()
    pv_color = "#00C853" if total_values[-1] >= balance else "#FF1744"
    pv_fill_rgb = "0,200,83" if total_values[-1] >= balance else "255,23,68"
    # Invisible baseline for portfolio value (avoid fill to zero)
    pv_base = min(total_values) - max((max(total_values) - min(total_values)) * 0.3, balance * 0.005)
    fig.add_trace(
        go.Scatter(x=df["round"].tolist(), y=[pv_base]*len(total_values),
                   mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["round"].tolist(), y=total_values, name=t("portfolio_value"),
                   line=dict(color=pv_color, width=2.5),
                   fill="tonexty", fillcolor=f"rgba({pv_fill_rgb},0.06)"),
        row=2, col=1,
    )
    fig.add_hline(y=balance, line_dash="dash", line_color="rgba(150,150,150,0.5)", row=2, col=1,
                  annotation_text=f"Initial ${balance:,.0f}", annotation_position="bottom right")
    # Auto-range price chart (row 1)
    fig.update_yaxes(range=[price_min - price_pad, price_max + price_pad], row=1, col=1)

    # Auto-range portfolio value (row 2)
    val_min = min(total_values)
    val_max = max(total_values)
    val_pad = max((val_max - val_min) * 0.3, balance * 0.005)
    fig.update_yaxes(range=[val_min - val_pad, val_max + val_pad], row=2, col=1)

    fig.update_layout(
        height=550, showlegend=True, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=40, b=20),
        font=dict(size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Row 3: Trade Log with $ amounts and PnL ---
    st.subheader(t("trade_log"))

    trades_only = df[df["final_action"] != "HOLD"][
        ["round", "price", "final_action", "quantity", "total_value", "realized_pnl"]
    ].copy()

    if not trades_only.empty:
        # Calculate trade amount (quantity * price)
        trades_only["amount"] = trades_only["quantity"] * trades_only["price"]

        # Calculate per-trade PnL: difference in realized_pnl between consecutive trades
        trades_only["trade_pnl"] = trades_only["realized_pnl"].diff().fillna(0.0)
        # First trade (BUY) has no PnL yet
        for idx in trades_only.index:
            if trades_only.loc[idx, "final_action"] == "BUY":
                trades_only.loc[idx, "trade_pnl"] = 0.0

        # Cumulative PnL
        trades_only["cum_pnl"] = trades_only["realized_pnl"]

        display_trades = trades_only[["round", "final_action", "price", "quantity", "amount",
                                      "trade_pnl", "cum_pnl", "total_value"]].copy()
        display_trades.columns = ["Round", "Action", "Price", "Qty", "Amount",
                                  "PnL", "Cumulative PnL", "Portfolio"]

        def color_action(val):
            if val == "BUY":
                return "background-color: rgba(0,200,83,0.15); color: #00C853; font-weight: bold"
            elif val in ("SELL", "CLOSE"):
                return "background-color: rgba(255,23,68,0.15); color: #FF1744; font-weight: bold"
            return ""

        def color_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "color: #00C853; font-weight: bold"
                elif val < 0:
                    return "color: #FF1744; font-weight: bold"
            return ""

        st.dataframe(
            display_trades.style
                .map(color_action, subset=["Action"])
                .map(color_pnl, subset=["PnL", "Cumulative PnL"])
                .format({
                    "Price": "${:,.2f}",
                    "Qty": "{:.6f}",
                    "Amount": "${:,.2f}",
                    "PnL": "${:+,.2f}",
                    "Cumulative PnL": "${:+,.2f}",
                    "Portfolio": "${:,.2f}",
                }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(t("no_trades"))

    # ========================================================
    # ERC-8004 on-chain status panel
    # ========================================================
    st.divider()
    st.subheader("🔗 ERC-8004 Trustless Agent Registry")

    erc_col1, erc_col2, erc_col3 = st.columns(3)

    agent_id = st.session_state.get("agent_id")
    v_artifacts = st.session_state.validation.get_all_artifacts()
    rep_feedback = st.session_state.reputation.get_feedback_history()

    # --- Identity Registry ---
    with erc_col1:
        st.markdown("#### 🆔 Identity Registry")
        if agent_id is not None:
            st.markdown(f"**Agent ID:** `#{agent_id}`")
            st.caption(f"Chain: Base Sepolia ({settings.onchain.chain_id})")
            reg_addr = settings.onchain.identity_registry_address
            st.caption(f"Contract: `{reg_addr[:10]}...{reg_addr[-6:]}`")
            st.success("Registered")
        else:
            st.warning("Not registered")

    # --- Validation Registry ---
    with erc_col2:
        st.markdown("#### ✅ Validation Registry")
        st.metric("Signed trade intents", len(v_artifacts))
        if v_artifacts:
            signed = sum(1 for a in v_artifacts if a.get("signature"))
            st.caption(f"EIP-712 signed: {signed} / {len(v_artifacts)}")
            last = v_artifacts[-1]
            st.caption(f"Latest: `{last['hash'][:16]}…`")

    # --- Reputation Registry ---
    with erc_col3:
        st.markdown("#### ⭐ Reputation Registry")
        st.metric("Feedback records", len(rep_feedback))
        if rep_feedback:
            trade_yields = [f for f in rep_feedback if f.get("tag1") == "tradingYield"]
            success = [f for f in rep_feedback if f.get("tag1") == "successRate"]
            if trade_yields:
                st.caption(f"Trade yields: {len(trade_yields)}")
            if success:
                st.caption(f"Session success: {success[-1]['value']:.1%}")

    # Full artifact table (expander)
    if v_artifacts:
        with st.expander(f"📋 All {len(v_artifacts)} ERC-8004 Validation Artifacts"):
            artifact_rows = [
                {
                    "Type": a["type"],
                    "Pair": a["data"]["pair"],
                    "Direction": a["data"]["direction"],
                    "Price": f"${int(a['data']['price']) / 100:,.2f}",
                    "Hash": a["hash"][:24] + "…",
                    "Signed": "Yes" if a.get("signature") else "No",
                }
                for a in v_artifacts
            ]
            st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)

else:
    # Empty state
    st.info(t("empty_state"))
    st.markdown(f"### {t('how_it_works')}")
    st.markdown(t("how_desc"))
