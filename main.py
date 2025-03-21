import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Dictionary to hold calibration data for left, center, and right
calibration_data = {'Left': None, 'Center': None, 'Right': None}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Collect ratios for detected eyes in this frame
    current_ratios = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Preprocess the eye region for pupil detection
            blurred = cv2.GaussianBlur(eye_gray, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Filter contours by area
            valid_contours = [cnt for cnt in contours if 50 < cv2.contourArea(cnt) < 500]

            if valid_contours:
                # Choose the largest contour (assumed to be the pupil)
                largest = max(valid_contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(largest)
                
                # Draw detected pupil in the eye ROI
                cv2.circle(eye_color, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)
                cv2.line(eye_color, (int(cx)-10, int(cy)), (int(cx)+10, int(cy)), (0, 255, 0), 2)
                cv2.line(eye_color, (int(cx), int(cy)-10), (int(cx), int(cy)+10), (0, 255, 0), 2)

                # Calculate horizontal ratio (pupil position / eye width)
                ratio = cx / float(ew)
                current_ratios.append(ratio)

    # Compute the average ratio from all detected eyes (if any)
    if current_ratios:
        avg_ratio = np.mean(current_ratios)
    else:
        avg_ratio = None

    # Display instructions or calibration info on the frame
    if any(v is None for v in calibration_data.values()):
        instruction = "Calibrate: Press L, C, or R to capture calibration for Left, Center, Right."
    else:
        instruction = "Detection Mode: Calibration complete."

    cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show current average ratio (if available)
    if avg_ratio is not None:
        cv2.putText(frame, f"Ratio: {avg_ratio:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # If calibration is complete, determine gaze direction by comparing to calibration data
    if all(v is not None for v in calibration_data.values()) and avg_ratio is not None:
        # Compute absolute difference between current ratio and each calibration value
        distances = {k: abs(avg_ratio - v) for k, v in calibration_data.items()}
        gaze_direction = min(distances, key=distances.get)
        cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        # Show the current calibration values if available
        cal_text = f"Calib - L:{calibration_data['Left']}, C:{calibration_data['Center']}, R:{calibration_data['Right']}"
        cv2.putText(frame, cal_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Gaze Calibration & Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Calibration key presses: save the current avg_ratio for the corresponding direction
    if avg_ratio is not None:
        if key == ord('l'):
            calibration_data['Left'] = avg_ratio
        elif key == ord('c'):
            calibration_data['Center'] = avg_ratio
        elif key == ord('r'):
            calibration_data['Right'] = avg_ratio

cap.release()
cv2.destroyAllWindows()
