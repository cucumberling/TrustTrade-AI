# TrustTrade AI

**Multi-agent autonomous trading system with ERC-8004 on-chain trust and Kraken CLI integration.**

Built for the [LabLab AI Trading Agents Hackathon](https://lablab.ai) (Mar 30 – Apr 12, 2026).

---

## Architecture

```
Market Data ──► Signal Agent ──► Risk Agent ──► Portfolio Agent
                (Analyst)       (Risk Mgr)     (Position Sizing)
                                                      │
                                               Manager Agent
                                              (Final Decision)
                                                      │
                                             Execution Agent
                                               (Trader)
                                                      │
                                    ┌─────────────────┼─────────────────┐
                              Kraken CLI        Internal Sim       On-Chain
                            (Paper/Live)        (Fallback)      (ERC-8004)
```

**5 specialized AI agents** collaborate like a professional trading team:

| Agent | Role | What It Does |
|-------|------|-------------|
| **Signal Agent** | Analyst | Combines trend (MA crossover) and momentum (RSI + MACD) strategies |
| **Risk Agent** | Risk Manager | 6-point risk check: drawdown, daily loss, volatility, consecutive losses, position conflicts |
| **Portfolio Agent** | Position Sizing | Calculates quantity, stop-loss, and take-profit using ATR |
| **Manager Agent** | Decision Maker | Final BUY/SELL/HOLD decision + ERC-8004 validation artifacts |
| **Execution Agent** | Trader | Routes orders to Kraken CLI, live API, or internal simulation |

---

## Features

- **Multi-strategy signals** — Trend following (MA crossover) + Momentum (RSI + MACD), weighted combination
- **6-point risk management** — Max position %, max drawdown, daily loss limit, volatility threshold, consecutive loss halt, duplicate position check
- **Kraken CLI integration** — Paper trading via `kraken paper buy/sell`, real-time prices via `kraken ticker`, 3-tier fallback (CLI → REST API → mock data)
- **ERC-8004 on-chain trust** — Agent identity (ERC-721 NFT), reputation feedback (tradingYield), EIP-712 signed validation artifacts on Base Sepolia
- **Interactive dashboard** — Streamlit + Plotly with bilingual UI (English/Chinese), agent decision process visualization, live metrics
- **Position sizing** — Fixed fraction or Kelly criterion, ATR-based stop-loss/take-profit

---

## Quick Start

### 1. Install dependencies

```bash
cd multi_agent_trading
pip install -r requirements.txt
```

### 2. Run demo (no API keys needed)

```bash
python main.py --mode demo
```

### 3. Launch dashboard

```bash
streamlit run dashboard.py
# Open http://localhost:8501
```

### 4. Paper trading with Kraken CLI

```bash
# Install Kraken CLI: https://github.com/krakenfx/kraken-cli
kraken paper init
python main.py --mode paper --pair BTC/USD
```

---

## CLI Options

```
python main.py [OPTIONS]

  --mode {demo,paper,live,backtest}   Trading mode (default: demo)
  --pair PAIR                         Trading pair (default: BTC/USD)
  --rounds N                          Number of rounds, 0=infinite (default: 0)
  --balance AMOUNT                    Initial balance in USD (default: 10000)
```

---

## ERC-8004 Integration

The system integrates with the [ERC-8004 Trustless Agents](https://eips.ethereum.org/EIPS/eip-8004) standard on **Base Sepolia**:

| Registry | Contract Address | Purpose |
|----------|-----------------|---------|
| **Identity** | `0x8004A818BFB912233c491871b3d84c89A494BD9e` | Agent NFT registration with capabilities URI |
| **Reputation** | `0x8004B663056A597Dffe9eCcC1965A193B7388713` | Trading yield feedback (tradingYield, successRate) |
| **Validation** | Local EIP-712 signing | Trade intent artifacts with SHA-256 hash |

### On-chain setup

```bash
# Create .env file with your Base Sepolia wallet
echo "AGENT_PRIVATE_KEY=your_private_key_here" > .env

# Get test ETH from a faucet:
# https://faucets.chain.link/base-sepolia
```

The system gracefully falls back to local simulation when on-chain is unavailable.

---

## Kraken CLI Integration

Requires [Kraken CLI](https://github.com/krakenfx/kraken-cli) v0.3.0+.

```bash
kraken paper init          # Initialize $10,000 paper account
kraken paper buy BTC/USD 0.001 -o json
kraken paper sell BTC/USD 0.001 -o json
kraken paper status -o json
kraken ticker BTC/USD      # Real-time price
```

Data fallback chain: **Kraken CLI → Kraken REST API → Mock data** — the system runs in any environment.

---

## Dashboard

Interactive Streamlit dashboard with:

- **Bilingual UI** — Chinese / English toggle
- **Agent decision process** — Step-by-step reasoning for each round (Signal → Risk → Portfolio → Manager → Execution)
- **Price charts** — Candlesticks with MA overlays and BUY/SELL markers
- **Risk metrics** — Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor
- **Trade log** — Every trade with execution source and PnL
- **ERC-8004 artifacts** — Validation hashes and on-chain records

---

## Project Structure

```
multi_agent_trading/
├── agents/              # 5 AI agents (signal, risk, portfolio, manager, execution)
├── config/              # Settings with .env support
├── data/                # Kraken feed + mock data
├── execution/           # Kraken CLI executor
├── onchain/             # ERC-8004 identity, reputation, validation
├── portfolio/           # Position tracker + risk metrics
├── strategy/            # Trend (MA) + Momentum (RSI/MACD) strategies
├── llm/                 # Claude API integration for explanations
├── utils/               # Structured logging
├── dashboard.py         # Streamlit interactive dashboard
├── main.py              # CLI entry point
└── requirements.txt     # Python dependencies
```

---

## Tech Stack

- **Python 3.9+** — Core language
- **Kraken CLI** — Paper/live trading execution
- **Web3.py + eth-account** — ERC-8004 on-chain integration (Base Sepolia)
- **Streamlit + Plotly** — Interactive dashboard
- **Claude API** — LLM-powered trade explanations (optional)

---

## License

[MIT](LICENSE) — TrustTrade AI, 2026
