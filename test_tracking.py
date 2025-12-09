"""Simple test to diagnose gaze tracking issues"""
import cv2
from calibration import CalibrationUI
from gaze import GazeTracker

# Run calibration with debug mode
print("Starting calibration test...")
calibrator = CalibrationUI(screen_width=1920, screen_height=1080, grid_size=4)
tracker = calibrator.run_calibration()

if not tracker:
    print("Calibration failed!")
    exit()

# Enable debug mode
tracker.debug = True

# Test with webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n=== Testing Gaze Prediction ===")
print("Move your eyes around and watch the console output")
print("Press 'q' to quit\n")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Predict gaze
    gaze_point = tracker.predict_gaze(frame)

    # Display simple visualization
    if gaze_point:
        # Scale to display window
        display_x = int(gaze_point[0] * 640 / 1920)
        display_y = int(gaze_point[1] * 480 / 1080)

        # Draw on frame
        small_frame = cv2.resize(frame, (640, 480))
        cv2.circle(small_frame, (display_x, display_y), 10, (0, 255, 0), 2)
        cv2.putText(small_frame, f"Gaze: ({int(gaze_point[0])}, {int(gaze_point[1])})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Gaze Test', small_frame)
    else:
        cv2.imshow('Gaze Test', cv2.resize(frame, (640, 480)))

    frame_count += 1
    if frame_count % 30 == 0:  # Print every 30 frames
        print(f"--- Frame {frame_count} ---")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
