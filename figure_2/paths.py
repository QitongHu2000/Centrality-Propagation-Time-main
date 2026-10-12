"""Locations local to this figure; no parent-project or archive lookup."""

from pathlib import Path

FIGURE = "figure_2"
FIGURE_MODEL = None
CODE = Path(__file__).resolve().parent
PROJECT_ROOT = CODE
DATA = CODE / "data"
NETWORKS = DATA / "networks"
RESULTS = CODE / "results"
REPORTS = CODE / "reports"
ASSETS = CODE / "assets"


def figure_path(name=None, suffix=".pdf"):
    return CODE / (FIGURE + suffix)
