import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Subdirectories
DATASET_DIR = os.path.join(PROJECT_ROOT, "data")
REGISTRY_DIR = os.path.join(PROJECT_ROOT, "model_registry")