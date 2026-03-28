"""
generate_optical_flow_png_fast.py

Multiprocessing optical flow generation — uses all available CPU cores.
Saves flow as uint8 PNG (~15-30 KB per frame vs ~400 KB for float32 .npy).

Estimated time with multiprocessing:
    3500 videos × 49 frame pairs = ~171,500 flow PNGs
    Single core  : ~6 hours
    8 cores      : ~45–60 min
    16 cores     : ~25–35 min
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENH_CRASH_ROOT  = os.path.join(ROOT, "data", "frames_enhanced", "crash")
ENH_NORMAL_ROOT = os.path.join(ROOT, "data", "frames_enhanced", "normal")

FLOW_CRASH_ROOT  = os.path.join(ROOT, "data", "optical_flow_png", "crash")
FLOW_NORMAL_ROOT = os.path.join(ROOT, "data", "optical_flow_png", "normal")

FRAME_EXT = (".jpg", ".png", ".jpeg")
FLOW_CLIP = 20.0

# ─────────────────────────────────────────────
# How many cores to use
# Leave 1 core free so your PC stays responsive
# ─────────────────────────────────────────────
NUM_WORKERS = max(1, mp.cpu_count() - 2)


# ─────────────────────────────────────────────
# Core functions (must be module-level for pickle)
# ─────────────────────────────────────────────
def compute_flow(prev_gray, curr_gray):
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )


def flow_to_uint8(flow):
    """float32 (H,W,2) → uint8 PNG (H,W,3)."""
    dx = np.clip(flow[..., 0], -FLOW_CLIP, FLOW_CLIP)
    dy = np.clip(flow[..., 1], -FLOW_CLIP, FLOW_CLIP)

    dx_u8  = ((dx + FLOW_CLIP) / (2 * FLOW_CLIP) * 255).astype(np.uint8)
    dy_u8  = ((dy + FLOW_CLIP) / (2 * FLOW_CLIP) * 255).astype(np.uint8)

    mag    = np.sqrt(dx**2 + dy**2)
    mag    = np.clip(mag, 0, np.sqrt(2) * FLOW_CLIP)
    mag_u8 = (mag / (np.sqrt(2) * FLOW_CLIP) * 255).astype(np.uint8)

    return np.stack([dx_u8, dy_u8, mag_u8], axis=-1)


def process_one_video(args):
    """
    Worker function — processes a single video folder.
    Called by each subprocess independently.

    Args:
        args: (video_id, frame_dir, flow_dir)

    Returns:
        (video_id, num_saved, error_msg or None)
    """
    video_id, frame_dir, flow_dir = args

    try:
        os.makedirs(flow_dir, exist_ok=True)

        frame_files = sorted([
            f for f in os.listdir(frame_dir)
            if f.lower().endswith(FRAME_EXT)
        ])

        if len(frame_files) < 2:
            return (video_id, 0, None)

        prev = cv2.imread(os.path.join(frame_dir, frame_files[0]))
        if prev is None:
            return (video_id, 0, f"Cannot read first frame")

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        saved = 0

        for i in range(1, len(frame_files)):
            curr_path = os.path.join(frame_dir, frame_files[i])
            curr = cv2.imread(curr_path)
            if curr is None:
                continue

            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            flow      = compute_flow(prev_gray, curr_gray)
            flow_img  = flow_to_uint8(flow)

            stem     = os.path.splitext(frame_files[i - 1])[0]
            out_path = os.path.join(flow_dir, stem + ".png")
            cv2.imwrite(out_path, flow_img)

            prev_gray = curr_gray
            saved += 1

        return (video_id, saved, None)

    except Exception as e:
        return (video_id, 0, str(e))


def build_job_list(frame_root, flow_root):
    """Build list of (video_id, frame_dir, flow_dir) tuples."""
    jobs = []
    for video_id in sorted(os.listdir(frame_root)):
        frame_dir = os.path.join(frame_root, video_id)
        if not os.path.isdir(frame_dir):
            continue
        flow_dir = os.path.join(flow_root, video_id)
        jobs.append((video_id, frame_dir, flow_dir))
    return jobs


def batch_process_parallel(frame_root, flow_root, label):
    jobs = build_job_list(frame_root, flow_root)

    print(f"\nProcessing {label} — {len(jobs)} videos across {NUM_WORKERS} cores...")

    total_saved  = 0
    total_errors = 0

    # imap_unordered streams results as workers finish — more memory efficient
    with mp.Pool(processes=NUM_WORKERS) as pool:
        for video_id, saved, err in tqdm(
            pool.imap_unordered(process_one_video, jobs),
            total=len(jobs)
        ):
            if err:
                total_errors += 1
                tqdm.write(f"  [WARN] {video_id}: {err}")
            else:
                total_saved += saved

    print(f"  → {total_saved} flow PNGs saved")
    if total_errors:
        print(f"  → {total_errors} videos had errors (check warnings above)")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # IMPORTANT: multiprocessing on Windows requires this guard
    print(f"CPU cores available : {mp.cpu_count()}")
    print(f"Workers to be used  : {NUM_WORKERS}")

    batch_process_parallel(ENH_CRASH_ROOT,  FLOW_CRASH_ROOT,  "CRASH")
    batch_process_parallel(ENH_NORMAL_ROOT, FLOW_NORMAL_ROOT, "NORMAL")

    print("\n✅ Optical flow PNG generation complete.")
    print(f"   Saved to: data/optical_flow_png/")