"""
enhanced_infer_with_detection.py

Enhanced video inference with:
- Vehicle & pedestrian detection using YOLOv8
- Object tracking across frames
- Bounding box visualization with color-coded risk levels
- Speed estimation
- Risk heatmap overlay
- Detailed frame-by-frame metadata logging
"""

import cv2
import torch
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
MODEL_PATH = os.path.join(ROOT, "checkpoints", "accident_model.pth")
# YOLOv8 for object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ ultralytics not installed. Run: pip install ultralytics")
    YOLO_AVAILABLE = False

from models.accident_3d_cnn import Accident3DCNN

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
VIDEO_PATH = r"D:\College\Accident Prediction\data\videos\crash\001362.mp4"
OUTPUT_PATH = "output_enhanced_detection.avi"

CLIP_LEN = 16
THRESHOLD = 0.7
K = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detection config
DETECTION_CONFIDENCE = 0.5
TRACK_HISTORY_LEN = 30  # frames to track object positions

# YOLO classes for vehicles and pedestrians
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
PEDESTRIAN_CLASSES = [0]  # person

# ─────────────────────────────────────────────
# Initialize Models
# ─────────────────────────────────────────────
# Load accident prediction model
accident_model = Accident3DCNN().to(DEVICE)


accident_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

accident_model.eval()

# Load YOLO detector
if YOLO_AVAILABLE:
    detector = YOLO('yolov8n.pt')  # nano model for speed, use yolov8m.pt for accuracy
    print("✅ YOLOv8 detector loaded")
else:
    detector = None
    print("⚠️ Running without object detection")

# ─────────────────────────────────────────────
# Video I/O
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 10
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"[INFO] FPS={fps:.1f}, Size=({width}x{height}), Frames={total_frames}")

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    raise RuntimeError("❌ VideoWriter failed")

# ─────────────────────────────────────────────
# Tracking & State
# ─────────────────────────────────────────────
frame_buffer = deque(maxlen=CLIP_LEN)
risk_history = deque(maxlen=K)
risk_over_time = []

# Object tracking
object_tracks = defaultdict(lambda: {"positions": deque(maxlen=TRACK_HISTORY_LEN),
                                      "class": None,
                                      "last_seen": 0})


frame_idx = 0

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def get_risk_color(risk):
    """Return color based on risk level (BGR format)"""
    if risk < 0.3:
        return (0, 255, 0)      # Green - Low risk
    elif risk < 0.5:
        return (0, 255, 255)    # Yellow - Medium risk
    elif risk < 0.7:
        return (0, 165, 255)    # Orange - High risk
    else:
        return (0, 0, 255)      # Red - Critical risk


def estimate_speed(positions, fps):
    """Estimate speed in pixels/second from position history"""
    if len(positions) < 2:
        return 0
    
    recent = list(positions)[-10:]  # Last 10 positions
    if len(recent) < 2:
        return 0
    
    distances = []
    for i in range(1, len(recent)):
        x1, y1 = recent[i-1]
        x2, y2 = recent[i]
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        distances.append(dist)
    
    avg_dist_per_frame = np.mean(distances) if distances else 0
    return avg_dist_per_frame * fps


def draw_heatmap_overlay(frame, risk):
    """Create semi-transparent risk heatmap overlay"""
    if risk is None:
        return frame
    
    overlay = frame.copy()
    color = get_risk_color(risk)
    
    # Create gradient overlay
    alpha = 0.2 * risk  # More transparent at low risk
    cv2.rectangle(overlay, (0, 0), (width, height), color, -1)
    
    return cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)


def draw_info_panel(frame, risk, detections, frame_idx, alert):
    """Draw comprehensive info panel"""
    panel_height = 180
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # Dark gray background
    
    y_offset = 25
    
    # Risk score with bar
    cv2.putText(panel, f"Frame: {frame_idx}/{total_frames}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if risk is not None:
        risk_text = f"Risk: {risk:.3f}"
        cv2.putText(panel, risk_text, (10, y_offset + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, get_risk_color(risk), 2)
        
        # Risk bar
        bar_width = int(300 * risk)
        cv2.rectangle(panel, (10, y_offset + 45), (310, y_offset + 60), (100, 100, 100), -1)
        cv2.rectangle(panel, (10, y_offset + 45), (10 + bar_width, y_offset + 60), 
                     get_risk_color(risk), -1)
    
    # Detection counts
    y_offset += 80
    cv2.putText(panel, f"Vehicles: {detections['vehicles']}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
    cv2.putText(panel, f"Pedestrians: {detections['pedestrians']}", 
                (200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
    
    # Alert status
    if alert:
        cv2.putText(panel, "⚠️ ACCIDENT RISK DETECTED!", 
                    (10, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(panel, timestamp, (width - 200, y_offset + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return panel


# ─────────────────────────────────────────────
# Main Inference Loop
# ─────────────────────────────────────────────
print("🚀 Starting enhanced inference...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    
    # ── Object Detection ──────────────────────────────
    detections = {"vehicles": 0, "pedestrians": 0, "objects": []}
    
    if detector is not None:
        results = detector(frame, conf=DETECTION_CONFIDENCE, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Track object
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                track_id = f"{cls}_{int(center[0])}_{int(center[1])}"
                
                object_tracks[track_id]["positions"].append(center)
                object_tracks[track_id]["class"] = cls
                object_tracks[track_id]["last_seen"] = frame_idx
                
                # Classify and count
                is_vehicle = cls in VEHICLE_CLASSES
                is_pedestrian = cls in PEDESTRIAN_CLASSES
                
                if is_vehicle:
                    detections["vehicles"] += 1
                    box_color = (255, 0, 0)  # Blue for vehicles
                    label_prefix = "Vehicle"
                elif is_pedestrian:
                    detections["pedestrians"] += 1
                    box_color = (0, 255, 0)  # Green for pedestrians
                    label_prefix = "Person"
                else:
                    continue
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                             box_color, 2)
                
                # Calculate speed
                speed = estimate_speed(object_tracks[track_id]["positions"], fps)
                
                # Label with confidence and speed
                label = f"{label_prefix} {conf:.2f} | {speed:.1f}px/s"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                # Draw motion trail
                positions = list(object_tracks[track_id]["positions"])
                for i in range(1, len(positions)):
                    pt1 = tuple(map(int, positions[i-1]))
                    pt2 = tuple(map(int, positions[i]))
                    cv2.line(frame, pt1, pt2, box_color, 1)
                
                # Store detection info
                detections["objects"].append({
                    "type": label_prefix,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "speed": speed
                })
    
    # ── Accident Risk Prediction ──────────────────────
    resized = cv2.resize(frame, (224, 224))
    resized = resized.astype(np.float32) / 255.0
    frame_buffer.append(resized)
    
    alert = False
    risk = None
    
    if len(frame_buffer) == CLIP_LEN:
        clip = np.stack(frame_buffer, axis=0)
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logit = accident_model(clip)
            risk = torch.sigmoid(logit).item()
        
        risk_over_time.append(risk)
        risk_history.append(risk)
        
        if len(risk_history) == K and all(r >= THRESHOLD for r in risk_history):
            alert = True
    
    # ── Apply risk heatmap overlay ────────────────────
    if risk is not None and risk > 0.5:
        frame = draw_heatmap_overlay(frame, risk)
    
    # ── Draw alert if triggered ───────────────────────
    if alert:
        # Pulsing alert border
        border_thickness = 15 + int(5 * np.sin(frame_idx * 0.5))
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), border_thickness)
        
        # Alert text with background
        alert_text = "ACCIDENT RISK!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 80
        
        # Background rectangle
        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(frame, alert_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    # ── Create info panel ─────────────────────────────
    info_panel = draw_info_panel(frame, risk, detections, frame_idx, alert)
    
    # ── Combine frame and panel ───────────────────────
    display = np.vstack([frame, info_panel])
    
    # Resize to original dimensions for video writer
    display = cv2.resize(display, (width, height))
    out.write(display)
    
    
    # Progress indicator
    if frame_idx % 30 == 0:
        progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
        risk_str = f"{risk:.3f}" if risk is not None else "N/A"
        print(f"Progress: {progress:.1f}% | Frame {frame_idx} | Risk: {risk_str}")

# ─────────────────────────────────────────────
# Cleanup & Save Metadata
# ─────────────────────────────────────────────
cap.release()
out.release()

# Save risk curve
np.save("risk_enhanced.npy", np.array(risk_over_time))


print(f"""
✅ Enhanced inference complete!
📊 Output video: {OUTPUT_PATH}
📈 Risk data: risk_enhanced.npy
""")