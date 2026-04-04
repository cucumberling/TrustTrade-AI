#!/bin/bash
cd /Users/cucumberling/Desktop/multi_agent_trading
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
exec streamlit run dashboard.py --server.headless true --server.port 8501
