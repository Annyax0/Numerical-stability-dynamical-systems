"""Allows pytest to run from an uninstalled checkout or an editable install."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
