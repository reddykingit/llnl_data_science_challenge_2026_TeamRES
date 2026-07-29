from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
DASHBOARD_SCRIPT = ROOT / "src" / "dashboard.py"
if not DASHBOARD_SCRIPT.exists():
    raise FileNotFoundError(f"Could not find dashboard script at {DASHBOARD_SCRIPT}")

sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(DASHBOARD_SCRIPT), run_name="__main__")
