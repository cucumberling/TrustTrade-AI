from __future__ import annotations
"""
AI Trading Agent — Main Orchestration
Multi-agent autonomous trading system with ERC-8004 + Kraken CLI integration.

Usage:
    python main.py                  # Run with mock data (default)
    python main.py --mode paper     # Paper trading via Kraken CLI
    python main.py --mode live      # Live trading (requires API keys)
    python main.py --mode backtest  # Backtest with historical data
"""

import argparse
import sys
import time
from dataclasses import asdict

from config.settings import settings
from agents.signal_agent import SignalAgent
from agents.risk_agent import RiskAgent
from agents.portfolio_agent import PortfolioAgent
from agents.execution_agent import ExecutionAgent
from agents.manager_agent import ManagerAgent
from data.kraken_feed import kraken_feed
from data.mock_data import get_sample_btc_prices, generate_price_series
from portfolio.tracker import PortfolioTracker
from portfolio.risk_metrics import compute_risk_metrics, format_risk_report
from onchain.identity import IdentityManager, create_agent_registration_json
from onchain.reputation import ReputationManager
from onchain.validation import ValidationManager
from llm.explain import explainer
from utils.logger import logger


class TradingSystem:
    def __init__(self):
        self.portfolio = PortfolioTracker()
        self.signal_agent = SignalAgent()
        self.risk_agent = RiskAgent()
        self.portfolio_agent = PortfolioAgent()
        self.manager = ManagerAgent(self.portfolio)
        self.executor = ExecutionAgent(self.portfolio)

        # ERC-8004 modules
        self.identity = IdentityManager()
        self.reputation = ReputationManager()
        self.validation = ValidationManager()

        # State
        self.agent_id = None
        self.round_count = 0

    def initialize(self):
        """Initialize the trading system: register agent identity on-chain."""
        logger.info("=" * 60)
        logger.info("  AI TRADING AGENT — INITIALIZING")
        logger.info("=" * 60)
        logger.info(f"Mode: {settings.mode}")
        logger.info(f"Trading Pair: {settings.trading_pair}")
        logger.info(f"Initial Balance: ${settings.portfolio.initial_balance:,.2f}")

        # Register agent identity (ERC-8004)
        registration = create_agent_registration_json()
        self.agent_id = self.identity.register_agent(
            agent_uri="https://agent.example.com/registration.json"
        )
        logger.info(f"Agent ID: {self.agent_id}")
        logger.info("Initialization complete.\n")

    def run_single_round(self, prices: list[float]) -> dict:
        """Execute one complete trading round through the agent pipeline."""
        self.round_count += 1
        # Lookahead-bias fix: signals computed on closed bars only,
        # execution price = current bar's open (proxied by prices[-1]).
        if len(prices) < 2:
            return {}
        signal_prices = prices[:-1]
        current_price = prices[-1]

        logger.info(f"\n{'='*40} Round {self.round_count} {'='*40}")
        logger.info(f"Current Price: ${current_price:,.2f} | Prices: {len(prices)} data points")

        # Step 1: Signal Agent — analyze market
        signal = self.signal_agent.analyze(signal_prices)

        # Step 2: Risk Agent — evaluate risk
        risk = self.risk_agent.evaluate(
            prices=signal_prices,
            proposed_direction=signal.direction,
            portfolio=self.portfolio,
            current_price=current_price,
        )

        # Step 3: Manager Agent — final decision
        decision = self.manager.decide(
            signal=signal,
            risk=risk,
            prices=signal_prices,
            current_price=current_price,
        )

        # Step 4: Execution Agent — execute trade
        result = self.executor.execute(decision.trade_intent)

        # Step 5: Create validation artifact (ERC-8004)
        if decision.trade_intent.direction != "HOLD":
            artifact = self.validation.create_trade_intent_artifact(
                pair=decision.trade_intent.pair,
                direction=decision.trade_intent.direction,
                quantity=decision.trade_intent.quantity,
                price=current_price,
                agent_id=self.agent_id or 0,
                risk_score=risk.risk_score,
                decision_data=decision.validation_artifact,
            )

            # Submit validation request
            self.validation.submit_validation_request(
                agent_id=self.agent_id or 0,
                artifact=artifact,
            )

        # Step 6: Submit reputation feedback if trade closed
        if decision.trade_intent.direction == "CLOSE" and result.success:
            closed_trades = [t for t in self.portfolio.trade_history if t.action == "CLOSE"]
            if closed_trades:
                last_pnl = closed_trades[-1].pnl
                yield_pct = (last_pnl / self.portfolio.initial_balance) * 100
                self.reputation.submit_trading_yield(
                    agent_id=self.agent_id or 0,
                    yield_pct=yield_pct,
                    period="trade",
                )

        # Step 7: Generate explanation
        portfolio_state = self.portfolio.get_state(current_price)
        explanation = explainer.explain_decision(
            signal_summary=decision.signal_summary,
            risk_summary=decision.risk_summary,
            trade_direction=decision.trade_intent.direction,
            portfolio_state=portfolio_state,
            final_reasoning=decision.final_reasoning,
        )

        # Log the decision
        logger.log_decision(
            signals=decision.signal_summary,
            risk_decision=risk.action,
            final_action=decision.trade_intent.direction,
            portfolio_state=portfolio_state,
            explanation=explanation,
        )

        print(f"\n{explanation}\n")

        return {
            "round": self.round_count,
            "price": current_price,
            "signal": signal.direction,
            "risk": risk.action,
            "action": decision.trade_intent.direction,
            "execution": result.message,
            "portfolio": portfolio_state,
        }

    def run_backtest(self, prices: list[float], window: int = 30):
        """Run backtest over a price series using a sliding window."""
        logger.info(f"\n{'='*60}")
        logger.info(f"  BACKTEST MODE — {len(prices)} price points, window={window}")
        logger.info(f"{'='*60}\n")

        for i in range(window, len(prices)):
            price_window = prices[max(0, i - window): i + 1]
            self.run_single_round(price_window)

        self._print_final_report(prices[-1])

    def run_live_loop(self, rounds: int = 0):
        """Run live/paper trading loop. rounds=0 means infinite."""
        logger.info(f"\n{'='*60}")
        logger.info(f"  LIVE/PAPER TRADING MODE")
        logger.info(f"  Interval: {settings.loop_interval_seconds}s")
        logger.info(f"{'='*60}\n")

        count = 0
        import datetime
        last_reset_date = datetime.date.today()
        try:
            while rounds == 0 or count < rounds:
                # Reset daily PnL at day boundary
                today = datetime.date.today()
                if today != last_reset_date:
                    self.portfolio.reset_daily_pnl()
                    logger.info(f"Daily PnL reset for {today}")
                    last_reset_date = today

                prices = kraken_feed.get_price_series()
                if prices:
                    self.run_single_round(prices)
                else:
                    logger.warning("No price data available")

                count += 1
                if rounds == 0 or count < rounds:
                    logger.info(f"Sleeping {settings.loop_interval_seconds}s...")
                    time.sleep(settings.loop_interval_seconds)
        except KeyboardInterrupt:
            logger.info("\nTrading loop interrupted by user.")

        if prices:
            self._print_final_report(prices[-1])

    def run_demo(self):
        """Run a demo with mock data to showcase the full pipeline."""
        logger.info(f"\n{'='*60}")
        logger.info(f"  DEMO MODE — Using mock data")
        logger.info(f"{'='*60}\n")

        # Generate multiple rounds of mock data with different market conditions
        base_prices = get_sample_btc_prices()

        # Run through the price series with a sliding window
        window = 20
        for i in range(window, len(base_prices)):
            price_window = base_prices[max(0, i - window): i + 1]
            self.run_single_round(price_window)

        self._print_final_report(base_prices[-1])

    def _print_final_report(self, final_price: float):
        """Print the final session report."""
        risk_report = compute_risk_metrics(self.portfolio)
        portfolio_state = self.portfolio.get_state(final_price)
        decision_history = self.manager.get_decision_summary()

        # Risk report
        print("\n" + format_risk_report(risk_report))

        # Session report
        report = explainer.generate_session_report(
            decision_history=decision_history,
            risk_report=asdict(risk_report),
            portfolio_state=portfolio_state,
        )
        print(report)

        # Validation artifacts summary
        artifacts = self.validation.get_all_artifacts()
        if artifacts:
            print(f"\n[ERC-8004 Validation Artifacts: {len(artifacts)}]")
            for a in artifacts[-3:]:
                print(f"  - {a['type']}: {a['data']['direction']} {a['data']['pair']} | hash={a['hash'][:16]}...")

        # Reputation feedback summary
        feedback = self.reputation.get_feedback_history()
        if feedback:
            print(f"\n[ERC-8004 Reputation Feedback: {len(feedback)}]")
            for f in feedback[-3:]:
                print(f"  - {f['tag1']}: {f['value']}")


def parse_args():
    parser = argparse.ArgumentParser(description="AI Trading Agent")
    parser.add_argument(
        "--mode", choices=["demo", "paper", "live", "backtest"],
        default="demo",
        help="Trading mode (default: demo)",
    )
    parser.add_argument(
        "--pair", default="BTC/USD",
        help="Trading pair (default: BTC/USD)",
    )
    parser.add_argument(
        "--rounds", type=int, default=0,
        help="Number of rounds (0=infinite for live/paper)",
    )
    parser.add_argument(
        "--balance", type=float, default=10000.0,
        help="Initial balance (default: 10000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply CLI args to settings
    settings.mode = args.mode
    settings.trading_pair = args.pair
    settings.portfolio.initial_balance = args.balance

    # Initialize system
    system = TradingSystem()
    system.initialize()

    # Run based on mode
    if args.mode == "demo":
        system.run_demo()
    elif args.mode == "backtest":
        prices = generate_price_series(
            base_price=65000, num_points=200,
            volatility=0.02, seed=42,
        )
        system.run_backtest(prices, window=30)
    elif args.mode in ("paper", "live"):
        settings.mode = args.mode
        system.run_live_loop(rounds=args.rounds)
    else:
        system.run_demo()


if __name__ == "__main__":
    main()
