import cv2
import numpy as np
import time
from calibration import CalibrationUI
from gaze import GazeTracker

class GazeDemo:
    def __init__(self, tracker, screen_width=1920, screen_height=1080):
        self.tracker = tracker
        self.screen_width = screen_width
        self.screen_height = screen_height

        # For visualization
        self.gaze_trail = []
        self.max_trail_length = 20
        self.debug_mode = False  # Toggle to show face landmarks

    def run(self):
        """Run the gaze tracking demo"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("=== Gaze Tracking Demo ===")
        print("Press 'd' to toggle debug visualization")
        print("Press 'r' to recalibrate")
        print("Press 'q' or ESC to quit")

        # Create fullscreen window for gaze visualization
        cv2.namedWindow('Gaze Point', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Gaze Point', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        fps_counter = []
        last_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Predict gaze (with optional debug info)
            if self.debug_mode:
                gaze_point, landmarks = self.tracker.predict_gaze(frame, return_debug_info=True)
            else:
                gaze_point = self.tracker.predict_gaze(frame)
                landmarks = None

            # Create visualization canvas
            canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

            if gaze_point:
                gaze_x, gaze_y = gaze_point

                # Clamp to screen bounds
                gaze_x = np.clip(gaze_x, 0, self.screen_width)
                gaze_y = np.clip(gaze_y, 0, self.screen_height)

                # Add to trail
                self.gaze_trail.append((int(gaze_x), int(gaze_y)))
                if len(self.gaze_trail) > self.max_trail_length:
                    self.gaze_trail.pop(0)

                # Draw gaze trail with fading effect
                for i, (tx, ty) in enumerate(self.gaze_trail):
                    alpha = i / len(self.gaze_trail)
                    color = (
                        int(100 * alpha),
                        int(255 * alpha),
                        int(100 * alpha)
                    )
                    radius = int(3 + 7 * alpha)
                    cv2.circle(canvas, (tx, ty), radius, color, -1)

                # Draw current gaze point
                cv2.circle(canvas, (int(gaze_x), int(gaze_y)), 15, (0, 255, 0), 3)
                cv2.circle(canvas, (int(gaze_x), int(gaze_y)), 5, (0, 255, 0), -1)

                # Draw crosshair
                cv2.line(canvas, (int(gaze_x) - 30, int(gaze_y)),
                        (int(gaze_x) + 30, int(gaze_y)), (0, 255, 0), 2)
                cv2.line(canvas, (int(gaze_x), int(gaze_y) - 30),
                        (int(gaze_x), int(gaze_y) + 30), (0, 255, 0), 2)

                # Display coordinates
                coord_text = f"Gaze: ({int(gaze_x)}, {int(gaze_y)})"
                cv2.putText(canvas, coord_text, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                # No face detected
                cv2.putText(canvas, "No face detected", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time)
            last_time = current_time
            fps_counter.append(fps)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            avg_fps = np.mean(fps_counter)

            cv2.putText(canvas, f"FPS: {avg_fps:.1f}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Show instructions
            cv2.putText(canvas, "Press 'r' to recalibrate, 'q' to quit",
                       (20, self.screen_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # Display
            cv2.imshow('Gaze Point', canvas)

            # Show webcam feed in small window with optional debug overlay
            if self.debug_mode and landmarks is not None:
                frame = self.tracker.draw_debug_landmarks(frame, landmarks)

            small_frame = cv2.resize(frame, (640, 480))
            cv2.imshow('Webcam', small_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('d'):
                # Toggle debug mode
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('r'):
                # Recalibrate
                cv2.destroyAllWindows()
                calibrator = CalibrationUI(self.screen_width, self.screen_height)
                new_tracker = calibrator.run_calibration()
                if new_tracker:
                    self.tracker = new_tracker
                    self.gaze_trail = []
                # Recreate fullscreen window
                cv2.namedWindow('Gaze Point', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Gaze Point', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting gaze tracker...")
    print("First, we need to calibrate the system.\n")

    # Run calibration
    calibrator = CalibrationUI(screen_width=1920, screen_height=1080, grid_size=4)
    tracker = calibrator.run_calibration()

    if tracker:
        # Run demo
        demo = GazeDemo(tracker, screen_width=1920, screen_height=1080)
        demo.run()
    else:
        print("Calibration failed. Exiting.")
