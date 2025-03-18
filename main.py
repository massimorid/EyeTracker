import cv2
import numpy as np
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define UI elements (Three boxes: Left, Center, Right)
screen_width, screen_height = 640, 480
box_positions = {
    "Left": (50, 150, 200, 300),   # (x1, y1, x2, y2)
    "Center": (220, 150, 420, 300),
    "Right": (440, 150, 590, 300)
}

gaze_time = {"Left": 0, "Center": 0, "Right": 0}
start_time = time.time()
end_time = start_time + 60  # Run for 1 minute

while time.time() < end_time:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        break

    frame = cv2.resize(frame, (screen_width, screen_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw UI Boxes
    cv2.rectangle(frame, box_positions["Left"][:2], box_positions["Left"][2:], (0, 0, 255), -1)
    cv2.rectangle(frame, box_positions["Center"][:2], box_positions["Center"][2:], (0, 255, 0), -1)
    cv2.rectangle(frame, box_positions["Right"][:2], box_positions["Right"][2:], (255, 0, 0), -1)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]

            # Apply Gaussian blur and thresholding
            blurred = cv2.GaussianBlur(eye_gray, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                pupil_x = int(cx + ex + x)
                pupil_y = int(cy + ey + y)

                # Determine which box was looked at
                for box_name, (x1, y1, x2, y2) in box_positions.items():
                    if x1 < pupil_x < x2 and y1 < pupil_y < y2:
                        gaze_time[box_name] += 1

    # Show frame
    cv2.imshow('Eye Tracker UI', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

total_time = sum(gaze_time.values())
print("Final Gaze Time Distribution:")
for box, time_looked in gaze_time.items():
    avg_time = (time_looked / total_time) * 60 if total_time > 0 else 0
    print(f"{box}: {avg_time:.2f} seconds on average")
