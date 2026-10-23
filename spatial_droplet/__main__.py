"""
Package entry point for `python -m spatial_droplet`.
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from spatial_droplet.app import main

if __name__ == "__main__":
    main()

