"""Configuration for the variable-length circuit-infidelity campaign."""

from pathlib import Path

from experiments.circuit_benchmarks.config import CASE_KEYS, CHI_MAIN
from experiments.circuit_benchmarks.config import DT as DT

CAMPAIGN_ID = "circuit-infidelity-until-saturation-v2"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

CASE_ORDER = tuple(CASE_KEYS)
CHI_CAP = CHI_MAIN

# This prospective criterion chooses only the displayed endpoint. Once a
# method is outside the reliable regime, its infidelity is effectively flat
# when its range over the trailing window spans at most this many decades.
# A 0.05-decade range corresponds to a maximum/minimum ratio of about 1.12.
SATURATION_LOG_RANGE_DECADES = 0.05
SATURATION_WINDOW_STEPS = 10
MAX_STEPS = 120

DISPLAY_FLOOR = 1e-13
FIGURE_WIDTH_MM = 178.0
FIGURE_HEIGHT_MM = 52.0
DPI = 600
