"""Wrapper to launch Streamlit dashboard with correct working directory."""
import os
import sys

# Fix working directory before streamlit imports
os.chdir("/Users/cucumberling/Desktop/multi_agent_trading")
sys.path.insert(0, "/Users/cucumberling/Desktop/multi_agent_trading")

# Set port from environment if provided
port = os.environ.get("PORT", "8501")

from streamlit.web.cli import main
sys.argv = ["streamlit", "run", "/Users/cucumberling/Desktop/multi_agent_trading/dashboard.py",
            "--server.headless", "true", "--server.port", port]
main()
