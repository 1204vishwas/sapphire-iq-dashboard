"""
Convenient One-Click Launcher for the Streamlit Dashboard
Run: python run_dashboard.py
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    print("=" * 60)
    print("Launching Amazon India Executive BI Dashboard...")
    print(f"Target App: {app_path}")
    print("=" * 60)
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
