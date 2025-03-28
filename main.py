import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe 
mp_face_mesh = mp.solutions.face_mesh
face_mesh_model = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Start webcam
webcam = cv2.VideoCapture(0)

# Landmark indices for eyes
left_eye_outer_corner, left_eye_inner_corner = 33, 133
right_eye_outer_corner, right_eye_inner_corner = 362, 263
left_iris_landmarks = [474, 475, 476, 477]
right_iris_landmarks = [469, 470, 471, 472]

# Time spent looking in each direction
look_times = {"Left": 0, "Middle": 0, "Right": 0}
tracking_duration_seconds = 30

# Converts normalized Mediapipe landmarks to pixel coordinates
def get_pixel_coordinates(landmark, image_shape):
    return int(landmark.x * image_shape[1]), int(landmark.y * image_shape[0])

# Calculates iris position ratio within the eye
def get_eye_gaze_ratio(landmarks, outer_corner_idx, inner_corner_idx, iris_indices, image_shape):
    outer_corner = get_pixel_coordinates(landmarks[outer_corner_idx], image_shape)
    inner_corner = get_pixel_coordinates(landmarks[inner_corner_idx], image_shape)
    eye_width = inner_corner[0] - outer_corner[0]

    iris_points = [get_pixel_coordinates(landmarks[i], image_shape) for i in iris_indices]
    iris_center = np.mean(iris_points, axis=0)

    iris_offset = iris_center[0] - outer_corner[0]
    ratio = iris_offset / eye_width
    return ratio, iris_center

# Calibration prompt
print("Calibration: Please look straight in the center for 3 seconds")
time.sleep(2)

calibration_ratios = []

# Collect 30 frames of center gaze
for _ in range(30):
    frame_captured, calibration_image = webcam.read()
    if not frame_captured:
        break

    rgb_calibration_image = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2RGB)
    detection_result = face_mesh_model.process(rgb_calibration_image)

    if detection_result.multi_face_landmarks:
        landmarks = detection_result.multi_face_landmarks[0].landmark
        left_eye_ratio, _ = get_eye_gaze_ratio(landmarks, left_eye_outer_corner, left_eye_inner_corner, left_iris_landmarks, calibration_image.shape)
        right_eye_ratio, _ = get_eye_gaze_ratio(landmarks, right_eye_outer_corner, right_eye_inner_corner, right_iris_landmarks, calibration_image.shape)
        average_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        calibration_ratios.append(average_eye_ratio)

# Calculate neutral gaze and thresholds
neutral_gaze_ratio = np.mean(calibration_ratios)
left_gaze_threshold = neutral_gaze_ratio + 0.10
right_gaze_threshold = neutral_gaze_ratio - 0.10

print(f"Calibration completed great job")
print(f"Neutral gaze ratio: {neutral_gaze_ratio:.2f}")
print(f"Left threshold: {left_gaze_threshold:.2f}, Right threshold: {right_gaze_threshold:.2f}")

start_time = time.time()
print("Tracking started. Press 'q' or ESC to quit early.")

# Begin gaze tracking
while time.time() - start_time < tracking_duration_seconds:
    frame_captured, video_frame = webcam.read()
    if not frame_captured:
        break

    rgb_video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    detection_result = face_mesh_model.process(rgb_video_frame)

    image_height, image_width, _ = video_frame.shape
    gaze_direction = "Unknown"

    # Draw visual guidance zones
    one_third_width = image_width // 3
    cv2.rectangle(video_frame, (0, 0), (one_third_width, image_height), (255, 0, 0), 2)
    cv2.rectangle(video_frame, (one_third_width, 0), (2 * one_third_width, image_height), (0, 255, 0), 2)
    cv2.rectangle(video_frame, (2 * one_third_width, 0), (image_width, image_height), (0, 0, 255), 2)

    if detection_result.multi_face_landmarks:
        landmarks = detection_result.multi_face_landmarks[0].landmark
        left_eye_ratio, left_iris_center = get_eye_gaze_ratio(landmarks, left_eye_outer_corner, left_eye_inner_corner, left_iris_landmarks, (image_height, image_width))
        right_eye_ratio, right_iris_center = get_eye_gaze_ratio(landmarks, right_eye_outer_corner, right_eye_inner_corner, right_iris_landmarks, (image_height, image_width))

        # Draw iris circles
        cv2.circle(video_frame, (int(left_iris_center[0]), int(left_iris_center[1])), 5, (0, 255, 0), -1)
        cv2.circle(video_frame, (int(right_iris_center[0]), int(right_iris_center[1])), 5, (0, 255, 0), -1)

        average_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Decide gaze direction
        if average_eye_ratio < right_gaze_threshold:
            gaze_direction = "Left"
        elif average_eye_ratio > left_gaze_threshold:
            gaze_direction = "Right"
        else:
            gaze_direction = "Middle"

        # Track time spent in each zone
        look_times[gaze_direction] += 1

        # Display gaze info
        cv2.putText(video_frame, f"Gaze: {gaze_direction} ({average_eye_ratio:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display webcam feed
    cv2.imshow("Eye Tracking", video_frame)

    # Exit on ESC or 'q'
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == 27 or key_pressed == ord('q'):
        break

# Release camera and close window
webcam.release()
cv2.destroyAllWindows()

# Convert frame counts to seconds (~30 FPS)
for zone in look_times:
    look_times[zone] = round(look_times[zone] / 30, 2)

# Print session summary
print("\n=== Eye Tracking Results ===")
for zone, time_spent in look_times.items():
    print(f"{zone}: {time_spent} seconds")
