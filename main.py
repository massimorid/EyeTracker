import cv2
import numpy as np
import pyautogui

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Load Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    eye_centers = []  # List to store pupil positions

    for (x, y, w, h) in faces:
        # Region of interest (ROI) for the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            # Region of interest (ROI) for the eye
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(eye_gray, (7, 7), 0)

            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on area (remove small and large noise)
            min_contour_area = 50  
            max_contour_area = 500  
            contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

            if contours:
                # Find the largest contour (assumed to be the pupil)
                largest_contour = max(contours, key=cv2.contourArea)

                # Get the minimum enclosing circle for the largest contour
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

                # Convert cx, cy to absolute position in frame
                abs_cx, abs_cy = x + ex + int(cx), y + ey + int(cy)
                
                # Store the detected pupil position
                eye_centers.append((abs_cx, abs_cy))

                # Draw the circle around the pupil
                cv2.circle(eye_color, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)

                # Draw crosshairs for pupil tracking
                cv2.line(eye_color, (int(cx) - 10, int(cy)), (int(cx) + 10, int(cy)), (0, 255, 0), 2)  
                cv2.line(eye_color, (int(cx), int(cy) - 10), (int(cx), int(cy) + 10), (0, 255, 0), 2)  

    # Compute the average pupil position
    if len(eye_centers) > 0:
        avg_x = int(np.mean([pt[0] for pt in eye_centers]))
        avg_y = int(np.mean([pt[1] for pt in eye_centers]))

        # Normalize pupil coordinates to screen dimensions
        mapped_x = int((avg_x / frame.shape[1]) * screen_width)
        mapped_y = int((avg_y / frame.shape[0]) * screen_height)

        # Draw cursor on the screen (a red dot)
        cv2.circle(frame, (avg_x, avg_y), 10, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow('Eye Tracking', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
