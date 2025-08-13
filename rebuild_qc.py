# scripts/rebuild_qc.py
from pathlib import Path
import joblib
import pandas as pd

from core.data import DataLoader, make_splits
from core.features import QuartileCluster
from core.config import DATA_PATH

# Load and engineer data using the same pipeline as the app
df_full = DataLoader.load_csv(DATA_PATH)
df_train, df_val, df_test = make_splits(df_full, ratios=(0.70, 0.15, 0.15))
#print(df_train.columns)

qc = QuartileCluster().fit(df_train)
# qc.fit(df_train)

Path("classes").mkdir(exist_ok=True)
joblib.dump(qc, "classes/qc.pkl")
print("Saved classes/qc.pkl with core.features.QuartileCluster")
