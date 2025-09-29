import time
import math
import sys
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Safe globals for Torch serialization
torch.serialization.add_safe_globals([
    torch.nn.modules.container.ModuleList,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.SiLU
])

# Configuration constants
wc, wb, wt, wbu, wo = 1.0, 0.5, 2.0, 3.0, 0.2
alpha, delta = 0.7, 0.3
T_SAFE, T_MAX, FIXED_TIMER = 10, 60, 30

manual_lane = None  # Manual override (persist until cancelled)

# CCTV Video Directory
ASSETS_DIR = "public/assests/signal"

# Load CCTV videos
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
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded.\n")

# Emergency classes
EMERGENCY_CLASSES = ["ambulance", "firetruck", "police"]

# Roboflow API Config (replace key/url as needed)
ROBOFLOW_API_URL = "https://serverless.roboflow.com/vehicle-detection-q8q4n/7"
ROBOFLOW_API_KEY = "tRYNNLp5ApGoKrbXPSmx"


# -------------------------
# Parallel API detection
# -------------------------
def _post_frame_check(session, idx, frame):
    """Worker: send one frame to API, return idx if emergency detected, else None."""
    try:
        _, buffer = cv2.imencode(".jpg", frame)
        img_bytes = buffer.tobytes()
        resp = session.post(
            f"{ROBOFLOW_API_URL}?api_key={ROBOFLOW_API_KEY}",
            files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
            timeout=6
        )
        if resp.status_code != 200:
            # non-fatal: log and skip
            print(f"❌ API status {resp.status_code} for lane {idx}")
            return None
        data = resp.json()
        preds = data.get("predictions", [])
        for pred in preds:
            cls = pred.get("class", "").strip().lower()
            if cls in EMERGENCY_CLASSES:
                return idx
    except Exception as e:
        print(f"API exception for lane {idx}: {e}")
    return None


def detect_emergency_api_parallel(frames):
    """
    Check all frames in parallel. Returns a list of lane indices that had emergency detections.
    Uses ThreadPoolExecutor and a requests.Session for connection reuse.
    """
    if not frames:
        return []

    emergency_lanes = []
    with requests.Session() as session:
        # limit workers to number of frames but at most 8 to avoid too many threads
        max_workers = min(len(frames), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_post_frame_check, session, idx, frame): idx for idx, frame in frames.items()}
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    emergency_lanes.append(res)
    # preserve ordering and make unique
    return list(dict.fromkeys(emergency_lanes))


# -------------------------
# Capture frames utilities
# -------------------------
def capture_frame(cctv_id):
    path = CCTV_URLS.get(cctv_id)
    if not path or not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def capture_all_frames():
    frames = {}
    for idx in CCTV_URLS.keys():
        frame = capture_frame(idx)
        if frame is not None:
            frames[idx] = frame
    return frames


# -------------------------
# UI: warning & signals
# -------------------------
def draw_warning(img, text):
    """
    Draw a non-blinking warning triangle + message near bottom-center.
    Positioned so footer won't overlap.
    """
    h, w = img.shape[:2]
    # place warning higher (so footer below it)
    base_y = h - 160
    center_x = w // 2

    pts = np.array([
        [center_x - 50, base_y + 40],
        [center_x + 50, base_y + 40],
        [center_x, base_y - 40]
    ], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (0, 0, 255))

    # exclamation
    cv2.putText(img, "!", (center_x - 15, base_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5, cv2.LINE_AA)

    # centered smaller text (so footer doesn't overlap)
    text_font_scale = 0.8
    text_thickness = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, text_thickness)
    tx = max(10, center_x - tw // 2)
    ty = base_y + 70
    cv2.putText(img, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)


def show_signals(active_lane_idx, timer_text, lane_count, emergency=False):
    """
    1280x720 display, well-aligned signals, timer, labels, pedestrian icon and footer.
    Returns the key pressed (0 if none).
    """
    screen_w, screen_h = 1280, 720
    img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    t = time.time()

    # slightly smaller header padding to avoid overlap
    margin_x = 80
    seg_w = (screen_w - (2 * margin_x)) // lane_count
    base_y = screen_h // 3

    for idx in range(1, lane_count + 1):
        cx = margin_x + (idx - 1) * seg_w + seg_w // 2

        # active signal color (pulse)
        if idx == active_lane_idx:
            pulse = int(35 * (1 + math.sin(t * 3)))
            color = (0, 200 + pulse, 0)
        else:
            color = (0, 0, 255)

        # main signal
        cv2.circle(img, (cx, base_y), 45, color, -1)

        # timer under active signal
        if idx == active_lane_idx:
            size = cv2.getTextSize(timer_text, font, 2, 4)[0]
            cv2.putText(img, timer_text, (cx - size[0] // 2, base_y + 100),
                        font, 2, (0, 255, 0), 4, cv2.LINE_AA)

        # signal label
        label = f"Signal {idx}"
        label_size = cv2.getTextSize(label, font, 1, 3)[0]
        cv2.putText(img, label, (cx - label_size[0] // 2, base_y + 150),
                    font, 1, (255, 255, 255), 3, cv2.LINE_AA)

        # pedestrian indicator
        ped_y = base_y + 230
        ped_color = (0, 0, 255) if idx == active_lane_idx else (0, 255, 0)
        cv2.circle(img, (cx, ped_y), 22, ped_color, -1)
        cv2.putText(img, "Pedestrian", (cx - 50, ped_y + 45),
                    font, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        # manual label above signal
        if manual_lane == idx:
            cv2.putText(img, "MANUAL", (cx - 40, base_y - 60),
                        font, 0.8, (0, 255, 255), 3, cv2.LINE_AA)

    # fixed emergency warning (placed above footer)
    if emergency:
        draw_warning(img, "EMERGENCY VEHICLE DETECTED - PRIORITY GIVEN")

    # footer (bottom padding)
    footer_y = screen_h - 30
    footer = f"[Controls] 1-{lane_count}=Override | 0=Cancel | q=Quit"
    cv2.putText(img, footer, (20, footer_y),
                font, 0.9, (180, 180, 180), 2, cv2.LINE_AA)

    # show window with correct size
    cv2.namedWindow("TRAFFIC SIGNALS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TRAFFIC SIGNALS", screen_w, screen_h)
    cv2.imshow("TRAFFIC SIGNALS", img)

    key = cv2.waitKey(1) & 0xFF
    return key


# -------------------------
# Detection + timing logic
# -------------------------
def process_frames(frames):
    """
    Run YOLO on each captured frame to compute Qi and return lane times.
    """
    lane_vehicle_data = []
    for idx in sorted(CCTV_URLS.keys()):
        frame = frames.get(idx)
        vehicle_classes = {"car": 0, "bus": 0, "truck": 0, "motorbike": 0, "others": 0}

        if frame is not None:
            results = model(frame)
            result = results[0]
            for cls in result.boxes.cls:
                cls_idx = int(cls.item()) if hasattr(cls, "item") else int(cls)
                name = model.names.get(cls_idx, str(cls_idx))
                if name in vehicle_classes:
                    vehicle_classes[name] += 1
                else:
                    vehicle_classes["others"] += 1

        Qi = wc * vehicle_classes["car"] + wb * vehicle_classes["motorbike"] + wt * vehicle_classes["truck"] + wbu * vehicle_classes["bus"] + wo * vehicle_classes["others"]
        lane_vehicle_data.append((idx, vehicle_classes, Qi))

    total_Q = sum([d[2] for d in lane_vehicle_data]) or 1
    Pi = 0
    Si_values = []
    for _, _, Qi in lane_vehicle_data:
        Si = alpha * (Qi / total_Q) + delta * Pi
        Si_values.append(Si)
    total_S = sum(Si_values) or 1

    lane_times = []
    for (idx, vehicle_classes, Qi), Si in zip(lane_vehicle_data, Si_values):
        S_prime = Si / total_S
        K_dynamic = 30 + 10 * math.log(1 + total_Q)
        Ti = min(T_MAX, max(T_SAFE, S_prime * K_dynamic))
        lane_times.append((idx, Ti, vehicle_classes))

    return lane_times


# -------------------------
# Countdown & input handling
# -------------------------
def countdown(seconds, lane_idx, emergency=False):
    """
    Run countdown for 'seconds' while showing display.
    Returns tuple: ("manual", lane), ("cancel", None), ("quit", None) or ("done", None).
    """
    seconds = int(round(seconds))
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        timer_text = f"{mins:02d}:{secs:02d}"
        key = show_signals(lane_idx, timer_text, len(CCTV_URLS), emergency)

        global manual_lane
        # numeric keys 1-9 (support up to 9 lanes)
        if ord('1') <= key <= ord('9'):
            num = key - ord('0')
            if 1 <= num <= len(CCTV_URLS):
                manual_lane = num
                print(f"⚡ Manual override pressed: Lane {manual_lane}")
                return ("manual", manual_lane)
        elif key == ord('0'):
            manual_lane = None
            print("⚡ Manual override cancelled")
            return ("cancel", None)
        elif key == ord('q'):
            return ("quit", None)

        time.sleep(1)

    # final display at 00:00
    show_signals(lane_idx, "00:00", len(CCTV_URLS), emergency)
    return ("done", None)


# -------------------------
# Main loop
# -------------------------
def main_loop():
    global manual_lane
    print("Starting traffic signal loop...")
    try:
        while True:
            frames = capture_all_frames()
            if not frames:
                # fallback countdown for each lane if frames not available
                for idx in sorted(CCTV_URLS.keys()):
                    action, val = countdown(FIXED_TIMER, idx, emergency=False)
                    if action == "manual":
                        # serve manual lane immediately
                        while manual_lane is not None:
                            a, v = countdown(FIXED_TIMER, manual_lane, emergency=False)
                            if a == "manual":
                                manual_lane = v
                                continue
                            elif a == "cancel":
                                manual_lane = None
                                break
                            elif a == "quit":
                                return
                        break
                    elif action == "quit":
                        return
                continue

            # compute lane times via YOLO (no API inside)
            lane_times = process_frames(frames)

            # parallel API check for emergency lanes
            emergency_lanes = detect_emergency_api_parallel(frames)

            if emergency_lanes:
                # If multiple emergency lanes detected, service them in detected order.
                for lane in emergency_lanes:
                    action, val = countdown(FIXED_TIMER, lane, emergency=True)
                    if action == "manual":
                        # user pressed manual during emergency: service manual now
                        while manual_lane is not None:
                            a, v = countdown(FIXED_TIMER, manual_lane, emergency=False)
                            if a == "manual":
                                manual_lane = v
                                continue
                            elif a == "cancel":
                                manual_lane = None
                                break
                            elif a == "quit":
                                return
                        break  # break out of emergency queue to re-evaluate cycle
                    elif action == "cancel":
                        # user canceled; continue to next emergency
                        continue
                    elif action == "quit":
                        return
                # after emergencies handled (or manual override), continue main loop
                continue

            # no emergency — handle manual or normal cycling
            if manual_lane is not None:
                # service manual lane until canceled or changed
                a, v = countdown(FIXED_TIMER, manual_lane, emergency=False)
                if a == "manual":
                    manual_lane = v
                    continue
                elif a == "cancel":
                    manual_lane = None
                    continue
                elif a == "quit":
                    return
                else:
                    # done — keep manual lane active until canceled by user (design choice)
                    continue

            # normal cycling through lanes using computed times
            for idx, Ti, _ in lane_times:
                a, v = countdown(Ti, idx, emergency=False)
                if a == "manual":
                    manual_lane = v
                    # service manual immediately
                    while manual_lane is not None:
                        aa, vv = countdown(FIXED_TIMER, manual_lane, emergency=False)
                        if aa == "manual":
                            manual_lane = vv
                            continue
                        elif aa == "cancel":
                            manual_lane = None
                            break
                        elif aa == "quit":
                            return
                    break
                elif a == "cancel":
                    # cancelled without manual set — keep cycling
                    continue
                elif a == "quit":
                    return
                # else 'done' -> go to next lane

    except Exception as e:
        print("Unexpected error:", e)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()
