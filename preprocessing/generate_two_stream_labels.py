"""
generate_two_stream_labels.py

Generates labels_two_stream.csv without creating any NPZ clips.
Each row points directly to:
    - A video's frame directory (RGB)
    - A video's flow directory  (PNG flow)
    - The clip start/end frame indices
    - The soft label

This replaces generate_two_stream_clips_png.py entirely.
No large NPZ files are written — frames are loaded on-the-fly during training.
"""

import os
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENH_CRASH_ROOT  = os.path.join(ROOT, "data", "frames_enhanced", "crash")
ENH_NORMAL_ROOT = os.path.join(ROOT, "data", "frames_enhanced", "normal")

FLOW_CRASH_ROOT  = os.path.join(ROOT, "data", "optical_flow_png", "crash")
FLOW_NORMAL_ROOT = os.path.join(ROOT, "data", "optical_flow_png", "normal")

ANNOTATION_FILE = os.path.join(ROOT, "data", "excels", "final_accident_frames.csv")

OUTPUT_ROOT = os.path.join(ROOT, "data", "processed")
LABEL_FILE  = os.path.join(OUTPUT_ROOT, "labels_two_stream.csv")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
FPS         = 10
CLIP_LENGTH = 16
STRIDE      = 4        # increased from 2 → fewer clips, faster training
FRAME_EXT   = (".jpg", ".png", ".jpeg")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def assign_label(crash_frame, clip_end):
    delta = (crash_frame - clip_end) / FPS
    if delta <= 1.0:   return 1.0
    elif delta <= 1.5: return 0.8
    elif delta <= 2.0: return 0.6
    else:              return 0.0


def extract_video_id(folder_path):
    return folder_path.replace("\\", "/").strip("/").split("/")[-1]


def get_sorted_frames(frame_dir):
    return sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith(FRAME_EXT)
    ])


# ─────────────────────────────────────────────
# Process crash videos
# ─────────────────────────────────────────────
df = pd.read_csv(ANNOTATION_FILE)
df = df[df["Validity"] != "Not Present"]

labels = []

print("\nIndexing CRASH videos...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    video_id    = extract_video_id(row["folder"])
    crash_frame = int(row["accident_frame"])

    rgb_dir  = os.path.join(ENH_CRASH_ROOT, video_id)
    flow_dir = os.path.join(FLOW_CRASH_ROOT, video_id)

    if not os.path.exists(rgb_dir) or not os.path.exists(flow_dir):
        continue

    frame_files = get_sorted_frames(rgb_dir)
    if len(frame_files) < CLIP_LENGTH:
        continue

    for start in range(0, len(frame_files) - CLIP_LENGTH + 1, STRIDE):
        end = start + CLIP_LENGTH - 1

        if end >= crash_frame:
            continue

        label = assign_label(crash_frame, end)

        labels.append({
            "rgb_dir"  : rgb_dir,
            "flow_dir" : flow_dir,
            "start"    : start,
            "end"      : end,
            "label"    : label,
            "video"    : video_id,
            "type"     : "crash"
        })

# ─────────────────────────────────────────────
# Process normal videos
# ─────────────────────────────────────────────
print("\nIndexing NORMAL videos...")

for video_id in tqdm(os.listdir(ENH_NORMAL_ROOT)):
    rgb_dir  = os.path.join(ENH_NORMAL_ROOT, video_id)
    flow_dir = os.path.join(FLOW_NORMAL_ROOT, video_id)

    if not os.path.isdir(rgb_dir) or not os.path.exists(flow_dir):
        continue

    frame_files = get_sorted_frames(rgb_dir)
    if len(frame_files) < CLIP_LENGTH:
        continue

    for start in range(0, len(frame_files) - CLIP_LENGTH + 1, STRIDE):
        end = start + CLIP_LENGTH - 1

        labels.append({
            "rgb_dir"  : rgb_dir,
            "flow_dir" : flow_dir,
            "start"    : start,
            "end"      : end,
            "label"    : 0.0,
            "video"    : video_id,
            "type"     : "normal"
        })

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
pd.DataFrame(labels).to_csv(LABEL_FILE, index=False)

print(f"\n✅ Label generation complete.")
print(f"   Total clips indexed : {len(labels)}")
print(f"   Labels CSV          : {LABEL_FILE}")
print(f"   No NPZ files written — frames loaded on-the-fly during training.")
