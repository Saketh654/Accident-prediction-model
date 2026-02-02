import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

FRAME_EXTENSION = (".jpg", ".png", ".jpeg")

def detect_accident_frame_from_frames(frame_dir):
    frame_files = sorted(os.listdir(frame_dir))
    
    prev = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    motion_scores = []

    for f in frame_files[1:]:
        curr = cv2.imread(os.path.join(frame_dir, f))
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_gray, curr_gray)
        score = np.sum(diff)  # total change
        motion_scores.append(float(score))

        prev_gray = curr_gray

    accident_frame = np.argmax(motion_scores) + 1
    return accident_frame, motion_scores


def update_csv(csv_file, folder, accident_frame, motion_score):
    rows = []
    with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        if row["folder"] == folder:
            row["accident_frame"] = str(accident_frame)
            row["motion_score"] = json.dumps(motion_score)
            break

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["folder", "accident_frame", "motion_score"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved as: {csv_file}")
    

def batch_extract(crash_root,csv_file ):

    for root, dirs, files in os.walk(crash_root):
        frames = [f for f in files if f.lower().endswith(FRAME_EXTENSION)]

        if len(frames) == 0:
            continue  

        print(f"Processing frames in: {root}")

        accident_frame, motion_score = detect_accident_frame_from_frames(root)
        update_csv(csv_file, root, accident_frame, motion_score)
    print(f"\nAll crash videos processed.")
    


if __name__ == "__main__":
    FRAME_DIR = "data\\frames\\crash" 
    csv_file= "data\\excels\\predicted_accident_frames.csv"

    batch_extract(FRAME_DIR,csv_file)
