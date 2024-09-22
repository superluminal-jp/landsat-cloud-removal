import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Base paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"

# Data processing settings
NUM_PROCESSES: int = os.cpu_count() or 1
LOCAL_STORAGE_PATH: Path = DATA_DIR / "landsat_dataset"
GEOMETRY_FOLDER_PATH: Path = DATA_DIR / "geometry"
ASSETS: List[str] = ["blue", "green", "red", "qa_pixel"]
PREFECTURE: str = os.getenv("PREFECTURE", "岡山県")
YEAR: int = int(os.getenv("YEAR", "2023"))
TILE_SIZE: int = int(os.getenv("TILE_SIZE", "256"))
N_SAMPLE: int = int(os.getenv("N_SAMPLE", "1000"))

# Model settings
EPOCHS: int = int(os.getenv("EPOCHS", "1000"))
PATIENCE: int = int(EPOCHS ** 0.5)
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))
DEPTH: int = int(os.getenv("DEPTH", "3"))

# Experiment settings
DATE: str = datetime.now().strftime("%y%m%d")
EXPERIMENT: str = f"ConvTransformer_epoch{EPOCHS}_batch{BATCH_SIZE}_sample{N_SAMPLE}_depth{DEPTH}"
EXPERIMENT_NAME: str = f"{DATE}_{EXPERIMENT}"

# Paths
BASE_PATH: Path = MODELS_DIR / EXPERIMENT_NAME
TRAIN_VALID_PATH: Path = DATA_DIR / "train_valid"
LOG_FILE: Path = BASE_PATH / f"process_{EXPERIMENT_NAME}.log"
MODEL_PATH: Path = BASE_PATH / f"{EXPERIMENT_NAME}.h5"
WEIGHT_PATH: Path = BASE_PATH / f"{EXPERIMENT_NAME}.weights.h5"

# AWS settings
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# Landsat API settings
LANDSAT_API_URL: str = "https://landsatlook.usgs.gov/stac-server"
LANDSAT_SEARCH_URL: str = f"{LANDSAT_API_URL}/search"
LANDSAT_SEARCH_PARAMS: Dict[str, Any] = {
    "collections": ["landsat-c2l2-sr"],
    "limit": int(os.getenv("LANDSAT_SEARCH_LIMIT", "4000"))
}

# Logging settings
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Create necessary directories
for path in [BASE_PATH, LOCAL_STORAGE_PATH, GEOMETRY_FOLDER_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Additional settings can be added here as needed