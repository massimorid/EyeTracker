import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

# Eye landmarks
left_eye_outer, left_eye_inner = 33, 133
right_eye_outer, right_eye_inner = 362, 263
left_iris_indices = [474, 475, 476, 477]
right_iris_indices = [469, 470, 471, 472]

look_times = {"Left": 0, "Middle": 0, "Right": 0}
duration = 30  # seconds

def get_landmark_px(landmark, shape):
    return int(landmark.x * shape[1]), int(landmark.y * shape[0])

def get_iris_gaze_ratio(landmarks, eye_outer, eye_inner, iris_indices, shape):
    eye_outer_px = get_landmark_px(landmarks[eye_outer], shape)
    eye_inner_px = get_landmark_px(landmarks[eye_inner], shape)
    eye_width = eye_inner_px[0] - eye_outer_px[0]

    iris_pts = [get_landmark_px(landmarks[i], shape) for i in iris_indices]
    iris_center = np.mean(iris_pts, axis=0)

    iris_offset = iris_center[0] - eye_outer_px[0]
    ratio = iris_offset / eye_width  # Normalize
    return ratio, iris_center

print("Calibration: Look straight ahead at the center for 3 seconds...")
time.sleep(2)

# Capture neutral (center) gaze values
calibration_samples = []
for _ in range(30):  # Collect frames for calibration
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ratio, _ = get_iris_gaze_ratio(landmarks, left_eye_outer, left_eye_inner, left_iris_indices, frame.shape)
        right_ratio, _ = get_iris_gaze_ratio(landmarks, right_eye_outer, right_eye_inner, right_iris_indices, frame.shape)
        avg_ratio = (left_ratio + right_ratio) / 2
        calibration_samples.append(avg_ratio)

neutral_gaze = np.mean(calibration_samples)  # Baseline for center gaze

# Define dynamic thresholds based on calibration
left_threshold = neutral_gaze + 0.10
right_threshold = neutral_gaze - 0.10

print(f"Calibration complete. Neutral gaze: {neutral_gaze:.2f}")
print(f"Left threshold: {left_threshold:.2f}, Right threshold: {right_threshold:.2f}")

start_time = time.time()
print("Starting eye-tracking session...")

while time.time() - start_time < duration:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    h, w, _ = frame.shape
    zone = "Unknown"

    third_w = w // 3
    cv2.rectangle(frame, (0, 0), (third_w, h), (255, 0, 0), 2)       # Left
    cv2.rectangle(frame, (third_w, 0), (2*third_w, h), (0, 255, 0), 2)  # Middle
    cv2.rectangle(frame, (2*third_w, 0), (w, h), (0, 0, 255), 2)     # Right

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ratio, left_iris_px = get_iris_gaze_ratio(landmarks, left_eye_outer, left_eye_inner, left_iris_indices, (h, w))
        right_ratio, right_iris_px = get_iris_gaze_ratio(landmarks, right_eye_outer, right_eye_inner, right_iris_indices, (h, w))

        # Draw both iris centers
        cv2.circle(frame, (int(left_iris_px[0]), int(left_iris_px[1])), 5, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_iris_px[0]), int(right_iris_px[1])), 5, (0, 255, 0), -1)

        avg_ratio = (left_ratio + right_ratio) / 2

        # Determine zone based on calibrated thresholds
        if avg_ratio < right_threshold:
            zone = "Right"
        elif avg_ratio > left_threshold:
            zone = "Left"
        else:
            zone = "Middle"

        look_times[zone] += 1
        cv2.putText(frame, f"Gaze: {zone} ({avg_ratio:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Eye Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Normalize time
for k in look_times:
    look_times[k] = round(look_times[k] / 30, 2)

print("\n=== Eye Tracking Results ===")
for k, v in look_times.items():
    print(f"{k}: {v} seconds")
