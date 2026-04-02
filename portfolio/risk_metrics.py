from __future__ import annotations
"""
Portfolio risk metrics: Sharpe ratio, max drawdown, win rate, etc.
"""

import math
from dataclasses import dataclass

from portfolio.tracker import PortfolioTracker


@dataclass
class RiskReport:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float


def compute_risk_metrics(tracker: PortfolioTracker) -> RiskReport:
    """Compute comprehensive risk metrics from trade history."""
    closed_trades = [t for t in tracker.trade_history if t.action == "CLOSE"]
    pnls = [t.pnl for t in closed_trades]

    total_trades = len(pnls)
    if total_trades == 0:
        return RiskReport(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, total_pnl=0.0, avg_win=0.0, avg_loss=0.0,
            profit_factor=0.0, max_drawdown=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
        )

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    total_pnl = sum(pnls)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown from trade-by-trade equity curve
    equity = [tracker.initial_balance]
    for pnl in pnls:
        equity.append(equity[-1] + pnl)

    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (annualized, assuming ~365 trades/year as proxy)
    mean_return = sum(pnls) / len(pnls) if pnls else 0
    std_return = math.sqrt(sum((p - mean_return) ** 2 for p in pnls) / len(pnls)) if len(pnls) > 1 else 0
    sharpe = (mean_return / std_return) * math.sqrt(365) if std_return > 0 else 0

    # Sortino ratio (only downside deviation)
    downside = [p for p in pnls if p < 0]
    downside_std = math.sqrt(sum(p ** 2 for p in downside) / len(pnls)) if downside else 0
    sortino = (mean_return / downside_std) * math.sqrt(365) if downside_std > 0 else 0

    return RiskReport(
        total_trades=total_trades,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=round(win_rate, 4),
        total_pnl=round(total_pnl, 2),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        profit_factor=round(profit_factor, 4),
        max_drawdown=round(max_dd, 4),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
    )


def format_risk_report(report: RiskReport) -> str:
    return (
        f"=== Risk Report ===\n"
        f"Total Trades:     {report.total_trades}\n"
        f"Win Rate:         {report.win_rate * 100:.1f}%\n"
        f"Total PnL:        ${report.total_pnl:,.2f}\n"
        f"Avg Win:          ${report.avg_win:,.2f}\n"
        f"Avg Loss:         ${report.avg_loss:,.2f}\n"
        f"Profit Factor:    {report.profit_factor:.2f}\n"
        f"Max Drawdown:     {report.max_drawdown * 100:.1f}%\n"
        f"Sharpe Ratio:     {report.sharpe_ratio:.2f}\n"
        f"Sortino Ratio:    {report.sortino_ratio:.2f}\n"
    )
