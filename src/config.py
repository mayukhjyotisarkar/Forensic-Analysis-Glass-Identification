"""Central configuration: paths, column names, class labels, constants.

Every other module imports from here, so a change (e.g. a new data path or
random seed) only ever needs to be made in one place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "glass.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "glass_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Tried in order if data/glass.csv is missing. Both are header-less copies of
# the UCI Glass Identification dataset (the UCI one has an extra Id column).
DATA_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv",
]

# ------------------------------------------------------------ dataset schema
FEATURES = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
TARGET = "Type"

# Note: type 4 (vehicle windows, non-float) exists in the documentation
# but has NO samples in the dataset.
CLASS_NAMES = {
    1: "Building windows (float)",
    2: "Building windows (non-float)",
    3: "Vehicle windows (float)",
    5: "Containers",
    6: "Tableware",
    7: "Headlamps",
}

# --------------------------------------------------------------- experiment
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_REPEATS = 3
