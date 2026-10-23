from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_DIR / "data"
EVENTLOGS_DIR = DATA_DIR / "eventlogs"
BASELINES_FILE = DATA_DIR / "baseline.csv"
RUNNABLE_CASES = DATA_DIR / "runnable_cases.csv"

# Results paths
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

# Create results directories if they don't exist
for directory in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)