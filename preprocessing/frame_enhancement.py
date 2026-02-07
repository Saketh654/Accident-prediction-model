import os
import cv2
import numpy as np
from tqdm import tqdm

# ----------------------------
# Project paths
# ----------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_ROOT = os.path.join(ROOT, "data", "frames", "normal")
OUTPUT_ROOT = os.path.join(ROOT, "data", "frames_enhanced", "normal")

FRAME_EXT = (".jpg", ".png", ".jpeg")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ----------------------------
# Enhancement functions
# ----------------------------
def enhance_frame(img):
    """
    Optimized for low-quality dashcam footage:
    - mild denoising
    - contrast enhancement
    - edge-preserving sharpening
    - light upscaling
    """

    # 1. Mild denoising
    img = cv2.fastNlMeansDenoisingColored(
        img, None,
        h=7, hColor=7,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 2. CLAHE (contrast enhancement)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    l = clahe.apply(l)

    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # 3. Gentle sharpening
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    img = cv2.filter2D(img, -1, kernel)

    # 4. Light upscaling
    img = cv2.resize(
        img, None,
        fx=1.5, fy=1.5,
        interpolation=cv2.INTER_LANCZOS4
    )

    return img


# ----------------------------
# Process dataset (video-wise)
# ----------------------------
video_folders = sorted([
    d for d in os.listdir(INPUT_ROOT)
    if os.path.isdir(os.path.join(INPUT_ROOT, d))
])

print(f"Total videos found: {len(video_folders)}")

for video_folder in video_folders:
    in_dir = os.path.join(INPUT_ROOT, video_folder)
    out_dir = os.path.join(OUTPUT_ROOT, video_folder)

    frames = sorted([
        f for f in os.listdir(in_dir)
        if f.lower().endswith(FRAME_EXT)
    ])

    # Skip already processed videos
    if os.path.exists(out_dir):
        processed_frames = [
            f for f in os.listdir(out_dir)
            if f.lower().endswith(FRAME_EXT)
        ]

        if len(processed_frames) == len(frames):
            print(f"Skipping {video_folder} (already enhanced)")
            continue

    os.makedirs(out_dir, exist_ok=True)

    print(f"\nEnhancing video: {video_folder} ({len(frames)} frames)")

    for f in tqdm(frames, desc=video_folder):
        in_path = os.path.join(in_dir, f)
        out_path = os.path.join(out_dir, f)

        # Skip already enhanced frames (extra safety)
        if os.path.exists(out_path):
            continue

        img = cv2.imread(in_path)
        if img is None:
            continue

        img = enhance_frame(img)
        cv2.imwrite(out_path, img)

print("\n✅ Frame enhancement completed successfully.")
