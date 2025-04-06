import cv2
import numpy as np

# --- CONFIG ---
video_path = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_parameter_variation1.mp4"
video_path = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_parameter_variation_resample_False.mp4"
video_path = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_parameter_variation_flow_threshold_0p2.mp4"
#video_path = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop\CP_parameter_variation_flow_threshold_0p6.mp4"





# --- GLOBAL STATE ---
zoom = 1.0
pan = [0, 0]  # Center pan position
drag_start = None
paused = True
frame_idx = 0
frames = []
zoom_center = [0, 0]  # Track the center of the zoom (relative to the frame)

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def on_mouse(event, x, y, flags, param):
    global zoom, pan, drag_start, zoom_center

    if event == cv2.EVENT_MOUSEWHEEL:
        zoom_factor = 1.2 if flags > 0 else 0.8
        zoom *= zoom_factor
        zoom = clamp(zoom, 1.0, 10.0)

        # Update the zoom center when zooming in/out
        zoom_center = [x, y]

    elif event == cv2.EVENT_RBUTTONDOWN:
        drag_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON:
        if drag_start is not None:
            dx = x - drag_start[0]
            dy = y - drag_start[1]
            pan[0] += dx
            pan[1] += dy
            drag_start = (x, y)
    elif event == cv2.EVENT_RBUTTONUP:
        drag_start = None

def display_zoomed(frame):
    global zoom_center, zoom, pan
    h, w = frame.shape[:2]

    # Calculate the zoom center in terms of the frame size
    center_x = zoom_center[0] - pan[0]
    center_y = zoom_center[1] - pan[1]

    crop_w = int(w / zoom)
    crop_h = int(h / zoom)

    # Keep the zoom center within the frame bounds
    x1 = clamp(center_x - crop_w // 2, 0, w - crop_w)
    y1 = clamp(center_y - crop_h // 2, 0, h - crop_h)
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped = frame[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized

# --- MAIN LOOP ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {video_path}")

# Load all frames into memory
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

cv2.namedWindow("ZoomViewer", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("ZoomViewer", on_mouse)

while True:
    if not paused:
        if frame_idx >= len(frames):
            print("End of video. Staying on last frame.")
            paused = True
            continue
        frame = frames[frame_idx]
        frame_idx += 1
        paused = True  # pause after each frame

    if frame is not None:
        view = display_zoomed(frame)
        cv2.putText(view, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("ZoomViewer", view)

    key = cv2.waitKey(0) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('d') or key == 83:  # Step forward
        paused = False
    elif key == ord('a') or key == 81:  # Step backward
        if frame_idx > 0:
            frame_idx -= 1
            frame = frames[frame_idx]
        else:
            print("No frames to go back to!")
    elif key == ord(' '):  # Spacebar toggles play/pause
        paused = not paused

cv2.destroyAllWindows()
