import cv2
import numpy as np
import time
from gaze import GazeTracker

class CalibrationUI:
    def __init__(self, screen_width=1920, screen_height=1080, grid_size=3):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_size = grid_size

        # Generate calibration points in a grid pattern
        self.calibration_targets = self._generate_grid_points()
        self.current_target_idx = 0

        self.tracker = GazeTracker()
        self.samples_per_point = 1  # Auto-advance after each sample
        self.current_samples = 0
        self.current_point_samples = []  # Store samples for outlier rejection

        self.calibration_complete = False
        self.manual_mode = True  # User presses SPACE to collect samples
        self.show_live_gaze = False  # Show gaze preview after first calibration point
        self.cycle_count = 0  # Track how many times user has cycled through all points

    def _generate_grid_points(self):
        """Generate calibration points in a grid across the screen"""
        margin_x = self.screen_width * 0.1  # 10% margin
        margin_y = self.screen_height * 0.1

        x_coords = np.linspace(margin_x, self.screen_width - margin_x, self.grid_size)
        y_coords = np.linspace(margin_y, self.screen_height - margin_y, self.grid_size)

        points = []
        for y in y_coords:
            for x in x_coords:
                points.append((int(x), int(y)))

        return points

    def _filter_outliers(self, samples):
        """Remove outlier samples using median absolute deviation (MAD)"""
        if len(samples) < 5:
            return samples

        samples_array = np.array(samples)
        median = np.median(samples_array, axis=0)
        mad = np.median(np.abs(samples_array - median), axis=0)

        # Modified Z-score threshold (more lenient than standard 3.5)
        threshold = 2.5
        z_scores = np.abs((samples_array - median) / (mad + 1e-6))

        # Keep samples where both dimensions are within threshold
        mask = np.all(z_scores < threshold, axis=1)
        filtered = samples_array[mask]

        return filtered.tolist() if len(filtered) > 0 else samples

    def draw_target(self, frame, target_pos, pulse_phase):
        """Draw an animated calibration target"""
        x, y = target_pos

        # Convert screen coords to webcam frame coords
        frame_h, frame_w = frame.shape[:2]
        frame_x = int(x * frame_w / self.screen_width)
        frame_y = int(y * frame_h / self.screen_height)

        # Pulsing animation
        radius = int(20 + 10 * np.sin(pulse_phase))

        # Draw target
        cv2.circle(frame, (frame_x, frame_y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (frame_x, frame_y), 5, (0, 255, 0), -1)

        # Draw crosshair
        cv2.line(frame, (frame_x - 30, frame_y), (frame_x + 30, frame_y), (0, 255, 0), 1)
        cv2.line(frame, (frame_x, frame_y - 30), (frame_x, frame_y + 30), (0, 255, 0), 1)

        return frame

    def run_calibration(self):
        """Run the interactive calibration process"""
        cap = cv2.VideoCapture(0)

        # Try to set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("=== Gaze Tracker Calibration ===")
        print(f"Total calibration points: {len(self.calibration_targets)} (4x4 grid)")
        print("Look at the GREEN target and press SPACE to collect a sample.")
        print("The target will automatically advance to the next point.")
        print("Keep pressing SPACE to cycle through all points multiple times for best accuracy.")
        print("Press ENTER when you have enough samples (recommended: 2-3 cycles), ESC to quit.")

        # Create fullscreen window for calibration targets
        cv2.namedWindow('Calibration Targets', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibration Targets', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        start_time = time.time()
        collecting_sample = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror the image

            # Create fullscreen canvas for calibration targets
            canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

            if self.current_target_idx < len(self.calibration_targets):
                # Still calibrating
                current_target = self.calibration_targets[self.current_target_idx]

                # Draw target on fullscreen canvas
                pulse_phase = (time.time() - start_time) * 3
                x, y = current_target
                radius = int(20 + 10 * np.sin(pulse_phase))

                # Draw target
                cv2.circle(canvas, (x, y), radius, (0, 255, 0), 2)
                cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)

                # Draw crosshair
                cv2.line(canvas, (x - 30, y), (x + 30, y), (0, 255, 0), 1)
                cv2.line(canvas, (x, y - 30), (x, y + 30), (0, 255, 0), 1)

                # Show progress on canvas
                total_samples = len(self.tracker.calibration_points)
                progress_text = f"Point {self.current_target_idx + 1}/{len(self.calibration_targets)} | Cycle {self.cycle_count + 1} | Total Samples: {total_samples}"
                cv2.putText(canvas, progress_text, (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                # Show instructions on canvas
                cv2.putText(canvas, "Press SPACE to collect & auto-advance", (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Show status on canvas
                if total_samples >= 11:
                    status_color = (0, 255, 0)  # Green - ready
                    status_text = f"Ready! Press ENTER to finish (or continue for more accuracy)"
                else:
                    status_color = (0, 165, 255)  # Orange - need more
                    status_text = f"Need {11 - total_samples} more samples (keep pressing SPACE)"
                cv2.putText(canvas, status_text, (20, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # Show live gaze preview on fullscreen canvas if enabled
                if self.show_live_gaze and self.tracker.is_calibrated:
                    gaze_point = self.tracker.predict_gaze(frame)
                    if gaze_point:
                        gaze_x, gaze_y = int(gaze_point[0]), int(gaze_point[1])

                        # Draw predicted gaze on canvas
                        cv2.circle(canvas, (gaze_x, gaze_y), 15, (255, 0, 255), 3)
                        cv2.putText(canvas, "Predicted", (gaze_x + 20, gaze_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                # Handle sample collection
                if collecting_sample:
                    features = self.tracker.get_features_only(frame)
                    if features is not None:
                        # Add the sample directly to calibration
                        self.tracker.add_calibration_features(
                            features,
                            current_target[0],
                            current_target[1]
                        )
                        print(f"  Sample collected at point {self.current_target_idx + 1} (Total: {len(self.tracker.calibration_points)})")

                        # Auto-advance to next point
                        self.current_target_idx += 1
                        if self.current_target_idx >= len(self.calibration_targets):
                            # Completed a full cycle, loop back to start
                            self.current_target_idx = 0
                            self.cycle_count += 1
                            print(f"\n✓ Cycle {self.cycle_count} complete! Starting cycle {self.cycle_count + 1}...")

                            # Rebuild model after each cycle for live preview
                            if len(self.tracker.calibration_points) >= 11:
                                self.tracker.calibrate()
                                self.show_live_gaze = True

                    collecting_sample = False
            else:
                # This shouldn't happen since we cycle continuously
                cv2.putText(canvas, "Press ENTER to finish calibration",
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Show fullscreen calibration canvas
            cv2.imshow('Calibration Targets', canvas)

            # Show small webcam preview for debugging
            small_frame = cv2.resize(frame, (320, 240))
            cv2.imshow('Webcam Preview', small_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Calibration cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return None
            elif key == 13:  # ENTER - finish calibration
                break
            elif key == 32:  # SPACE - collect sample and auto-advance
                if self.current_target_idx < len(self.calibration_targets):
                    collecting_sample = True

        cap.release()
        cv2.destroyAllWindows()

        # Build the calibration model
        print("\nBuilding calibration model...")
        num_points = len(self.tracker.calibration_points)
        print(f"Collected {num_points} calibration points")

        if num_points < 11:
            print(f"✗ Not enough calibration points! Need at least 11, got {num_points}")
            print("  Tip: Collect 4+ samples per target point, then press N to move to next")
            return None

        if self.tracker.calibrate():
            print("✓ Calibration successful!")
            print(f"  Model trained on {num_points} points")

            # Auto-save calibration
            self.tracker.save_calibration()

            print("  Tip: If tracking is inaccurate, press 'r' in demo to recalibrate")
            self.calibration_complete = True
            return self.tracker
        else:
            print("✗ Calibration failed - model building error")
            return None

if __name__ == "__main__":
    # Run calibration
    calibrator = CalibrationUI(screen_width=1920, screen_height=1080, grid_size=4)
    tracker = calibrator.run_calibration()

    if tracker:
        print("\nCalibration complete! You can now use the tracker.")
        print("Run the demo application to see gaze tracking in action.")
