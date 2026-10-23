"""
================================================================================
    DROPLET-AI: PRIMARY REPOSITORY ENTRYPOINT
================================================================================
Launches the autonomous desktop copilot, transparent HUD, and spatial reasoning engine.

Usage:
    python main.py
    python droplet_ai.py
    python live_copilot.py
================================================================================
"""
import sys
from pathlib import Path

# Ensure repository root is in sys.path
ROOT_DIR = str(Path(__file__).resolve().parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if __name__ == "__main__":
    import droplet_ai
    # Run the main entrypoint loop
    if hasattr(droplet_ai, "main"):
        droplet_ai.main()
    else:
        # Fallback to direct execution
        import runpy
        runpy.run_path(str(Path(__file__).parent / "droplet_ai.py"), run_name="__main__")
