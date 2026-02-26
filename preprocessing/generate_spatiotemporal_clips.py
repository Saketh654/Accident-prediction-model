import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Enhanced frame roots
ENH_CRASH_ROOT = os.path.join(ROOT, "data", "frames_enhanced", "crash")
ENH_NORMAL_ROOT = os.path.join(ROOT, "data", "frames_enhanced", "normal")

# Accident annotation CSV
ANNOTATION_FILE = os.path.join(
    ROOT, "data", "excels", "final_accident_frames.csv"
)

# Output
OUTPUT_ROOT = os.path.join(ROOT, "data", "processed")
CLIP_ROOT = os.path.join(OUTPUT_ROOT, "clips_enhanced")
LABEL_FILE = os.path.join(OUTPUT_ROOT, "labels_enhanced.csv")

os.makedirs(CLIP_ROOT, exist_ok=True)

# Configuration
FPS = 10
CLIP_LENGTH = 16
STRIDE = 2
FRAME_SIZE = (224, 224)
FRAME_EXT = (".jpg", ".png", ".jpeg")

df = pd.read_csv(ANNOTATION_FILE)
df = df[df["Validity"] != "Not Present"]

labels = []


# Utility functions
def load_frame(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read frame: {path}")
    img = cv2.resize(img, FRAME_SIZE)
    img = img.astype(np.float32) / 255.0
    return img


def assign_label(crash_frame, clip_end):
    delta = (crash_frame - clip_end) / FPS
    if delta <= 1.0:
        return 1.0
    elif delta <= 1.5:
        return 0.8
    elif delta <= 2.0:
        return 0.6
    else:
        return 0.0


def extract_video_id(folder_path):
    """
    From: data\\frames\\crash\\000001
    To:   000001
    """
    folder_path = folder_path.replace("\\", "/")
    return folder_path.strip("/").split("/")[-1]


# PROCESS ENHANCED CRASH VIDEOS
crash_out = os.path.join(CLIP_ROOT, "crash")
os.makedirs(crash_out, exist_ok=True)

print("\nProcessing ENHANCED CRASH videos...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    video_id = extract_video_id(row["folder"])
    crash_frame = int(row["accident_frame"])

    frame_dir = os.path.join(ENH_CRASH_ROOT, video_id)

    if not os.path.exists(frame_dir):
        continue

    frame_files = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith(FRAME_EXT)
    ])

    if len(frame_files) < CLIP_LENGTH:
        continue

    for start in range(0, len(frame_files) - CLIP_LENGTH + 1, STRIDE):
        end = start + CLIP_LENGTH - 1

        if end >= crash_frame:
            continue

        clip = []
        for i in range(start, start + CLIP_LENGTH):
            clip.append(
                load_frame(os.path.join(frame_dir, frame_files[i]))
            )

        clip = np.stack(clip)

        label = assign_label(crash_frame, end)

        clip_name = f"{video_id}_s{start}_e{end}.npy"
        clip_path = os.path.join(crash_out, clip_name)
        np.save(clip_path, clip)

        labels.append({
            "clip_path": clip_path,
            "label": label,
            "video": video_id,
            "clip_end": end,
            "type": "crash"
        })

# PROCESS ENHANCED NORMAL VIDEOS
normal_out = os.path.join(CLIP_ROOT, "normal")
os.makedirs(normal_out, exist_ok=True)

print("\nProcessing ENHANCED NORMAL videos...")

for video_id in tqdm(os.listdir(ENH_NORMAL_ROOT)):
    frame_dir = os.path.join(ENH_NORMAL_ROOT, video_id)

    frame_files = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith(FRAME_EXT)
    ])

    if len(frame_files) < CLIP_LENGTH:
        continue

    for start in range(0, len(frame_files) - CLIP_LENGTH + 1, STRIDE):
        end = start + CLIP_LENGTH - 1

        clip = []
        for i in range(start, start + CLIP_LENGTH):
            clip.append(
                load_frame(os.path.join(frame_dir, frame_files[i]))
            )

        clip = np.stack(clip)

        clip_name = f"{video_id}_s{start}_e{end}.npy"
        clip_path = os.path.join(normal_out, clip_name)
        np.save(clip_path, clip)

        labels.append({
            "clip_path": clip_path,
            "label": 0.0,
            "video": video_id,
            "clip_end": end,
            "type": "normal"
        })

# SAVE LABEL FILE
os.makedirs(OUTPUT_ROOT, exist_ok=True)
pd.DataFrame(labels).to_csv(LABEL_FILE, index=False)

print("\n✅ Enhanced spatiotemporal clip generation completed")
print(f"Total enhanced clips created: {len(labels)}")
