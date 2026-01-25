import cv2
import os
from tqdm import tqdm

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0
    with tqdm(total=total_frames, leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_name = f"frame_{frame_id:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            frame_id += 1
            pbar.update(1)

    cap.release()
    return fps, frame_id


def batch_extract(video_root, frame_root):
    os.makedirs(frame_root, exist_ok=True)

    videos = [f for f in os.listdir(video_root)
              if f.lower().endswith(VIDEO_EXTENSIONS)]

    print(f"Found {len(videos)} videos")

    summary = []

    for idx, video in enumerate(videos, start=1):
        video_path = os.path.join(video_root, video)
        video_name = os.path.splitext(video)[0]
        out_dir = os.path.join(frame_root, video_name)

        print(f"\n[{idx}/{len(videos)}] Processing: {video}")
        fps, total = extract_frames(video_path, out_dir)

        summary.append({
            "video": video,
            "fps": fps,
            "total_frames": total
        })

    print("\nAll videos processed.")
    return summary


if __name__ == "__main__":
    VIDEO_FOLDER = "data\\videos\\crash"
    FRAME_FOLDER = "data\\frames\\crash"

    batch_extract(VIDEO_FOLDER, FRAME_FOLDER)
    VIDEO_FOLDER = "data\\videos\\normal"
    FRAME_FOLDER = "data\\frames\\normal"
    
    batch_extract(VIDEO_FOLDER, FRAME_FOLDER)

