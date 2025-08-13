from pathlib import Path
import numpy as np
import tensorflow as tf

# Defaults (override via env vars in app if needed)
DATA_PATH     = Path("data/MMM_data.csv")
ARTIFACT_PATH = Path("classes")
WINDOW        = 45
TEST_RATIO    = 0.15
SEED          = 42
TARGET        = "demand"

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
