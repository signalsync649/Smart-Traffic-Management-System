import time
import math
import sys
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import requests
from datetime import datetime
import pytz

# Safe globals for Torch serialization
torch.serialization.add_safe_globals([
    torch.nn.modules.container.ModuleList,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.SiLU
])

# Configuration Constants
wc, wb, wt, wbu, wo = 1.0, 0.5, 2.0, 3.0, 0.2
alpha, delta = 0.7, 0.3
T_SAFE = 10
T_MAX = 60
FIXED_TIMER = 30  # fallback timer

# Directory where video files are stored
ASSETS_DIR = "public/assests/signal"

# Load video files
CCTV_URLS = {}
video_files = sorted([
    fname for fname in os.listdir(ASSETS_DIR)
    if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
])
for idx, fname in enumerate(video_files, start=1):
    CCTV_URLS[idx] = os.path.join(ASSETS_DIR, fname)

print("✔ Loaded CCTV videos:")
for idx, path in CCTV_URLS.items():
    print(f"  CCTV {idx}: {path}")

# Load YOLO model
print("Loading YOLO model (this may take a moment)...")
model = YOLO("yolov8n.pt")
print("Model loaded.\n")

EMERGENCY_CLASSES = ["ambulance", "firetruck", "police"]

# Capture a frame from CCTV
def capture_frame(cctv_id):
    path = CCTV_URLS.get(cctv_id)
    if not path or not os.path.exists(path):
        print(f"❌ Video not found for CCTV {cctv_id}")
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"❌ Unable to open video for CCTV {cctv_id}")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"❌ Unable to read frame from CCTV {cctv_id}")
        return None
    return frame

# Capture all frames
def capture_all_frames():
    frames = {}
    for idx in CCTV_URLS.keys():
        frame = capture_frame(idx)
        if frame is not None:
            frames[idx] = frame
        else:
            print(f"⚠️ Skipping CCTV {idx}")
    return frames

# Display all signals with active timer
def show_timer_screen(active_lane_idx, timer_text, all_vehicle_counts, emergency_lanes=None, wait_ms=50):
    # Get screen resolution for fullscreen display
    screen_width = 1920
    screen_height = 1080
    height, width = screen_height, screen_width

    img = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    t = time.time()  # for animation

    signal_count = len(CCTV_URLS)

    # Add extra left and right padding
    left_padding = 150
    right_padding = 150
    usable_width = width - (left_padding + right_padding)
    segment_width = usable_width // signal_count
    base_y = height // 2 - 100  # vertically center signals

    emergency_lanes = emergency_lanes or []

    for idx in sorted(CCTV_URLS.keys()):
        # Center signals horizontally with padding
        start_x = left_padding + (idx - 1) * segment_width + segment_width // 2

        # Pulsing green for active signal
        if idx == active_lane_idx:
            pulse = int(50 * (1 + math.sin(t * 3)))  # oscillate 0-100
            color = (0, 200 + pulse, 0)
        else:
            color = (0, 0, 255)

        # Draw signal circle
        cv2.circle(img, (start_x, base_y), 60, color, -1)

        # Blinking red border for emergency
        if idx in emergency_lanes:
            if int(t * 2) % 2 == 0:  # blink every 0.5s
                cv2.circle(img, (start_x, base_y), 70, (0, 0, 255), 6)

        # Timer above active signal
        if idx == active_lane_idx:
            timer_size = cv2.getTextSize(timer_text, font, 3, 5)[0]
            timer_x = start_x - timer_size[0] // 2
            timer_y = base_y - 120
            cv2.putText(img, timer_text, (timer_x, timer_y), font, 3, (0, 255, 0), 5, cv2.LINE_AA)

        # Label below
        label_text = f"Signal {idx}"
        label_size = cv2.getTextSize(label_text, font, 1.5, 3)[0]
        label_x = start_x - label_size[0] // 2
        label_y = base_y + 120
        cv2.putText(img, label_text, (label_x, label_y), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

        # Vehicle counts
        counts = all_vehicle_counts.get(idx, {})
        counts_str = " ".join([f"{k}:{v}" for k, v in counts.items()])
        counts_size = cv2.getTextSize(counts_str, font, 1, 2)[0]
        counts_x = start_x - counts_size[0] // 2
        counts_y = base_y + 170
        cv2.putText(img, counts_str, (counts_x, counts_y), font, 1, (230, 230, 230), 2, cv2.LINE_AA)

    # Footer
    footer = "Press 'q' to quit / Auto updating every second"
    cv2.putText(img, footer, (40, height - 40), font, 1, (180, 180, 180), 2, cv2.LINE_AA)

    # Fullscreen display
    cv2.namedWindow("TRAFFIC SIGNALS", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("TRAFFIC SIGNALS", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("TRAFFIC SIGNALS", img)
    key = cv2.waitKey(wait_ms) & 0xFF
    return key

# Process frames and compute timers
def process_frames_and_compute_times(frames):
    lane_vehicle_data = []
    Qi_values = []
    emergency_detected = False

    for idx in sorted(CCTV_URLS.keys()):
        frame = frames.get(idx)
        if frame is None:
            vehicle_classes = {"car":0, "bus":0, "truck":0, "motorbike":0, "others":0}
            Qi = 0
        else:
            results = model(frame)
            result = results[0]
            vehicle_classes = {"car":0, "bus":0, "truck":0, "motorbike":0, "others":0}
            for cls in result.boxes.cls:
                cls_idx = int(cls.item()) if hasattr(cls, "item") else int(cls)
                name = model.names.get(cls_idx, str(cls_idx))
                if name in vehicle_classes:
                    vehicle_classes[name] += 1
                else:
                    vehicle_classes["others"] += 1
                if name.lower() in EMERGENCY_CLASSES:
                    emergency_detected = True

            Qi = (
                wc * vehicle_classes["car"] +
                wb * vehicle_classes["motorbike"] +
                wt * vehicle_classes["truck"] +
                wbu * vehicle_classes["bus"] +
                wo * vehicle_classes["others"]
            )

        Qi_values.append(Qi)
        lane_vehicle_data.append((idx, vehicle_classes, Qi))

    total_Q = sum(Qi_values) or 1
    Pi = 0
    Si_values = []
    for (_, _, Qi) in lane_vehicle_data:
        Si = alpha * (Qi / total_Q) + delta * Pi
        Si_values.append(Si)

    total_S = sum(Si_values) or 1
    lane_times = []

    for (idx, vehicle_classes, Qi), Si in zip(lane_vehicle_data, Si_values):
        S_prime = Si / total_S
        K_dynamic = 30 + 10 * math.log(1 + total_Q)
        Ti = min(T_MAX, max(T_SAFE, S_prime * K_dynamic))
        lane_times.append((idx, Ti, Qi, vehicle_classes))

    return lane_times, emergency_detected

# Countdown for signal
def countdown(seconds, lane_idx, vehicle_counts, lane_times=None):
    seconds = int(round(seconds))
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        timer_text = f"{mins:02d}:{secs:02d}"

        # Determine emergency lanes for display
        emergency_lanes = []
        if lane_times:
            emergency_lanes = [
                idx for idx, _, _, counts in lane_times
                if any(k.lower() in EMERGENCY_CLASSES for k in counts.keys())
            ]

        key = show_timer_screen(lane_idx, timer_text, {lane_idx: vehicle_counts}, emergency_lanes)
        if key == ord('q'):
            raise KeyboardInterrupt
        time.sleep(1)

    show_timer_screen(lane_idx, "00:00", {lane_idx: vehicle_counts}, emergency_lanes)

    # Send data to FastAPI
    timestamp = datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
    hour_slot = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%H:00")
    data = {
        "signal_id": lane_idx,
        "timestamp": timestamp,
        "hour_slot": hour_slot,
        "total_vehicles": sum(vehicle_counts.values()),
        "vehicles": vehicle_counts
    }
    try:
        response = requests.post("http://localhost:8000/traffic", json=data)
        if response.status_code == 200:
            print(f"📤 Data sent for lane {lane_idx}")
        else:
            print(f"⚠️ Failed to send data: {response.text}")
    except Exception as e:
        print(f"⚠️ Error sending data: {e}")

# Main loop
def main_loop():
    print("Starting traffic signal loop...")
    try:
        while True:
            frames = capture_all_frames()
            if not frames:
                print("⚠️ No frames available, using fixed timer loop")
                # Fallback loop
                for idx in sorted(CCTV_URLS.keys()):
                    countdown(FIXED_TIMER, idx, {"car":0,"bus":0,"truck":0,"motorbike":0,"others":0})
                continue

            lane_times, emergency = process_frames_and_compute_times(frames)

            # If emergency vehicle detected, immediately green that lane
            if emergency:
                for idx, Ti, Qi, vehicle_counts in lane_times:
                    if any(k.lower() in EMERGENCY_CLASSES for k in vehicle_counts.keys()):
                        print(f"🚨 Emergency detected on Lane {idx}, giving green until it passes")
                        countdown(FIXED_TIMER, idx, vehicle_counts, lane_times)
                        break
                continue

            for idx, Ti, Qi, vehicle_counts in lane_times:
                print(f"🚦 Lane {idx}: Timer {Ti} seconds, Vehicles {vehicle_counts}")
                countdown(Ti, idx, vehicle_counts, lane_times)

            print("\nCycle finished — recapturing frames...\n")

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main_loop()
