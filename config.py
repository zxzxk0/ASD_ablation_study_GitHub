"""
config.py — Global settings
===========================
API credentials must be supplied through environment variables.
Do not commit API keys to Git.
"""

import os
from pathlib import Path

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model identifiers
MODELS = {
    "gpt4v": "gpt-4o",
    "gemini": "gemini-2.5-flash",
}

# Data paths
# Override these with environment variables if your files live elsewhere.
TRAIN_JSONL = os.getenv(
    "ASD_TRAIN_JSONL",
    str(Path("data") / "train_scanpath_absolute.jsonl"),
)
TEST_JSONL = os.getenv(
    "ASD_TEST_JSONL",
    str(Path("data") / "test_scanpath_absolute.jsonl"),
)

# Optional root remapping for image paths stored inside JSONL.
IMAGE_ROOT_REMAP = os.getenv("ASD_IMAGE_ROOT_REMAP") or None

# CARS range
CARS_MIN = 15.0
CARS_MAX = 60.0

# Inference settings
MAX_TOKENS = 512
TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_DELAY_SEC = 5
REQUEST_DELAY_SEC = 1.0

# Output
RESULTS_DIR = os.getenv("ASD_RESULTS_DIR", "./results")
