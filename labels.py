"""
fix_labels_paths.py

Run this ONCE after moving frames_enhanced to the internal SSD.
Updates all flow_dir paths in labels_two_stream.csv to the new location.

Usage:
    python fix_labels_paths.py
"""

import pandas as pd

LABELS_CSV = "data/processed/labels_two_stream_val.csv"

# ── Set these to match your actual paths ─────────────────────────────────────
OLD_PREFIX = "L:\Accident Prediction\data\optical_flow_png"        # where it WAS (external SSD)
NEW_PREFIX = "D:\College\Accident Prediction\data\optical_flow_png"  # where it IS NOW
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(LABELS_CSV)

before = df["flow_dir"].iloc[0]

df["flow_dir"] = df["flow_dir"].str.replace(
    OLD_PREFIX, NEW_PREFIX, regex=False
)

after = df["flow_dir"].iloc[0]

df.to_csv(LABELS_CSV, index=False)

print(f"✅ Updated {len(df)} rows in {LABELS_CSV}")
print(f"   Before : {before}")
print(f"   After  : {after}")