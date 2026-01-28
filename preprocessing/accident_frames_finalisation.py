import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

FRAME_EXTENSION = (".jpg", ".png", ".jpeg")

# -------------------------------------------------
# Show frames interactively with arrow keys
# -------------------------------------------------
def show_candidate_frames(frame_dir, accident_frame, window=10):
    frame_files = [f for f in os.listdir(frame_dir) if f.lower().endswith(FRAME_EXTENSION)]
    frame_files = sorted(frame_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

    start = max(0, accident_frame - window)
    end = min(len(frame_files), accident_frame + window + 1)

    idx = min(max(accident_frame, start), end - 1)

    while True:
        img = cv2.imread(os.path.join(frame_dir, frame_files[idx]))
        if img is None:
            print(f"Could not read frame: {frame_files[idx]}")
            break

        display = img.copy()
        cv2.putText(display, f"Frame {idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display, f"Predicted Accident Frame: {accident_frame}", (20, 650),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Check Accident Frame", display)
        key = cv2.waitKey(0)

        if key == 27 or key == ord('q'):  # ESC
            break
        elif key == 81 or key == ord('a'):  # Left
            idx = max(0, idx - 1)
        elif key == 83 or key == ord('d'):  # Right
            idx = min(49, idx + 1)

    cv2.destroyAllWindows()


# -------------------------------------------------
# Plot motion score curve
# -------------------------------------------------
def plot_motion_curve(motion_scores):
    plt.figure(figsize=(10, 4))
    plt.plot(motion_scores)
    plt.title("Motion Score Curve")
    plt.xlabel("Frame Index")
    plt.ylabel("Motion Score")
    plt.show()


# -------------------------------------------------
# Verification (Human-in-the-loop)
# -------------------------------------------------
def verify_accident_frames(FRAME_DIR, accident_frame, motion_scores, writer):
    #plot_motion_curve(motion_scores)
    show_candidate_frames(FRAME_DIR, accident_frame, window=15)
    print(f"\nFolder: {FRAME_DIR}")
    result = input("\nIs the predicted accident frame correct and q to quit the program? (y/n/q): ").strip().lower()

    if result == 'y':
        final_frame = accident_frame
        validity = "Verified"
    
    elif result == 'q':
        print("Exiting verification.")
        print(f"Next file to process is {FRAME_DIR}.")
        exit(0)
    else:
        final_frame = int(
            input("Enter correct accident frame number (enter -1 if not present): ")
        )

        if final_frame == -1:
            validity = "Not Present"
        else:
            validity = "Corrected"
    
        
    writer.writerow({
        "folder": FRAME_DIR,
        "accident_frame": final_frame,
        "motion_score": json.dumps(motion_scores),
        "Validity": validity
    })
    
    


# -------------------------------------------------
# Main Driver
# -------------------------------------------------
if __name__ == "__main__":

    in_csv = "data\\excels\\predicted_accident_frames.csv"
    out_csv = "data\\excels\\final_accident_frames.csv"
    file_exists = os.path.isfile(out_csv)

    # Open output CSV once and keep it open
    with open(out_csv, mode="a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=["folder", "accident_frame", "motion_score", "Validity"]
        )
        if not file_exists or os.stat(out_csv).st_size == 0:
            writer.writeheader()


        mem = input("Enter your name (l/a/s): ").strip().lower()
        count = 0
        start = int(input("Enter starting index (or press Enter to start from beginning): ") or 0)
        start-=1
        if mem == 'l':
            print("Logged in as: Lohith")
        elif mem == 'a':
            print("Logged in as: Ashmi")
        elif mem == 's':
            print("Logged in as: Saketh")
        
        # Read predicted CSV
        with open(in_csv, mode='r', newline='', encoding="utf-8") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            

            print("\nControls:")
            print(" d  -> Next frame")
            print(" a  -> Previous frame")
            print(" q  -> Exit frame window\n")

            for i, row in enumerate(csv_reader, start=0):
                if start and i <start:
                    continue
                if count>=500:
                    break
                FRAME_DIR = row["folder"]
                accident_frame = int(row["accident_frame"])
                motion_scores = json.loads(row["motion_score"])

                verify_accident_frames(
                    FRAME_DIR,
                    accident_frame,
                    motion_scores,
                    writer
                )
                count+=1

    print(f"\nFinal verified CSV saved at: {out_csv}")
