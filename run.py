#!/usr/bin/env python
"""
Launch script for BTC Active Trading Lab.
"""
import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    # Get the dashboard app path
    project_root = Path(__file__).parent
    app_path = project_root / "dashboard" / "app.py"
    
    # Run Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    
    print("🚀 Starting BTC Active Trading Lab...")
    print(f"   Dashboard: http://localhost:8501")
    print()
    
    subprocess.run(cmd, cwd=str(project_root))


if __name__ == "__main__":
    main()
