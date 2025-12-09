import cv2
import numpy as np
import time
from calibration import CalibrationUI

class Boid:
    def __init__(self, x, y, screen_width, screen_height):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.random.randn(2) * 2
        self.acc = np.array([0.0, 0.0])
        self.max_speed = 6.0
        self.max_force = 0.3
        self.screen_width = screen_width
        self.screen_height = screen_height

    def seek(self, target):
        """Steer towards a target"""
        desired = target - self.pos
        dist = np.linalg.norm(desired)

        if dist > 0:
            desired = (desired / dist) * self.max_speed

            # Slow down as we approach the target
            if dist < 100:
                desired *= dist / 100

            steer = desired - self.vel
            steer_mag = np.linalg.norm(steer)
            if steer_mag > self.max_force:
                steer = (steer / steer_mag) * self.max_force

            return steer
        return np.array([0.0, 0.0])

    def separate(self, neighbors, desired_separation=25):
        """Avoid crowding neighbors (optimized with pre-filtered neighbors)"""
        steer = np.array([0.0, 0.0])
        count = 0

        for other in neighbors:
            diff_vec = self.pos - other.pos
            dist = np.linalg.norm(diff_vec)
            if dist > 0 and dist < desired_separation:
                diff_vec = diff_vec / dist  # Weight by distance
                steer += diff_vec
                count += 1

        if count > 0:
            steer /= count
            steer_mag = np.linalg.norm(steer)
            if steer_mag > 0:
                steer = (steer / steer_mag) * self.max_speed - self.vel
                steer_mag = np.linalg.norm(steer)
                if steer_mag > self.max_force:
                    steer = (steer / steer_mag) * self.max_force

        return steer

    def align(self, neighbors, neighbor_dist=50):
        """Align with neighbors' average velocity (optimized)"""
        sum_vel = np.array([0.0, 0.0])
        count = 0

        for other in neighbors:
            dist = np.linalg.norm(self.pos - other.pos)
            if dist > 0 and dist < neighbor_dist:
                sum_vel += other.vel
                count += 1

        if count > 0:
            sum_vel /= count
            sum_vel_mag = np.linalg.norm(sum_vel)
            if sum_vel_mag > 0:
                sum_vel = (sum_vel / sum_vel_mag) * self.max_speed

            steer = sum_vel - self.vel
            steer_mag = np.linalg.norm(steer)
            if steer_mag > self.max_force:
                steer = (steer / steer_mag) * self.max_force
            return steer

        return np.array([0.0, 0.0])

    def cohesion(self, neighbors, neighbor_dist=50):
        """Move towards the average position of neighbors (optimized)"""
        sum_pos = np.array([0.0, 0.0])
        count = 0

        for other in neighbors:
            dist = np.linalg.norm(self.pos - other.pos)
            if dist > 0 and dist < neighbor_dist:
                sum_pos += other.pos
                count += 1

        if count > 0:
            sum_pos /= count
            return self.seek(sum_pos)

        return np.array([0.0, 0.0])

    def get_neighbors(self, boids, radius=100):
        """Get nearby boids within radius (simple optimization)"""
        neighbors = []
        for other in boids:
            if other is self:
                continue
            # Quick distance check using squared distance (avoids sqrt)
            dx = other.pos[0] - self.pos[0]
            dy = other.pos[1] - self.pos[1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < radius * radius:
                neighbors.append(other)
        return neighbors

    def flock(self, boids, gaze_target):
        """Apply flocking behaviors (optimized with neighbor finding)"""
        # Only check nearby boids (major optimization)
        neighbors = self.get_neighbors(boids, radius=100)

        # Calculate the three flocking forces using only nearby boids
        sep = self.separate(neighbors)
        ali = self.align(neighbors)
        coh = self.cohesion(neighbors)

        # Calculate seeking force toward gaze target
        seek_force = self.seek(gaze_target)

        # Combine forces with weights
        # Higher separation prevents overcrowding
        # Higher seek makes boids more responsive to gaze
        sep *= 1.5
        ali *= 1.0
        coh *= 1.0
        seek_force *= 2.0

        # Apply combined force to acceleration
        self.acc = sep + ali + coh + seek_force

    def update(self):
        """Update position and velocity"""
        self.vel += self.acc
        vel_mag = np.linalg.norm(self.vel)
        if vel_mag > self.max_speed:
            self.vel = (self.vel / vel_mag) * self.max_speed

        self.pos += self.vel
        self.acc *= 0  # Reset acceleration

        # Wrap around screen edges
        if self.pos[0] < 0:
            self.pos[0] = self.screen_width
        elif self.pos[0] > self.screen_width:
            self.pos[0] = 0

        if self.pos[1] < 0:
            self.pos[1] = self.screen_height
        elif self.pos[1] > self.screen_height:
            self.pos[1] = 0

    def draw(self, canvas, color=(0, 255, 100)):
        """Draw the boid as a triangle pointing in direction of velocity"""
        if np.linalg.norm(self.vel) > 0:
            angle = np.arctan2(self.vel[1], self.vel[0])
        else:
            angle = 0

        size = 8
        points = np.array([
            [size, 0],
            [-size, size/2],
            [-size, -size/2]
        ])

        # Rotation matrix
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        rotated = points @ rot_matrix.T
        translated = rotated + self.pos

        pts = translated.astype(np.int32)
        cv2.fillPoly(canvas, [pts], color)
        cv2.circle(canvas, tuple(self.pos.astype(int)), 2, (255, 255, 255), -1)


class BoidsGazeFollower:
    def __init__(self, tracker, screen_width=1920, screen_height=1080, num_boids=100):
        self.tracker = tracker
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Create boids
        self.boids = [
            Boid(
                np.random.randint(0, screen_width),
                np.random.randint(0, screen_height),
                screen_width,
                screen_height
            )
            for _ in range(num_boids)
        ]

        self.gaze_target = np.array([screen_width / 2, screen_height / 2])
        self.show_gaze_cursor = True
        self.debug_mode = False  # Toggle to show face landmarks

    def run(self):
        """Run the boids simulation with gaze tracking"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("=== Boids Eye Follower ===")
        print("The particles will follow your gaze!")
        print("Press 'c' to toggle gaze cursor")
        print("Press 'd' to toggle debug visualization")
        print("Press 'r' to recalibrate")
        print("Press 'q' or ESC to quit")

        # Create fullscreen window
        cv2.namedWindow('Boids', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Boids', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

            if gaze_point:
                gaze_x, gaze_y = gaze_point
                gaze_x = np.clip(gaze_x, 0, self.screen_width)
                gaze_y = np.clip(gaze_y, 0, self.screen_height)
                self.gaze_target = np.array([gaze_x, gaze_y])

            # Create canvas
            canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

            # Update and draw all boids
            for boid in self.boids:
                boid.flock(self.boids, self.gaze_target)
                boid.update()

                # Color based on distance to gaze target
                dist = np.linalg.norm(boid.pos - self.gaze_target)
                max_dist = 500
                intensity = max(0, 1 - dist / max_dist)

                color = (
                    int(100 + 155 * intensity),
                    int(255 * intensity),
                    int(100)
                )
                boid.draw(canvas, color)

            # Draw gaze cursor
            if self.show_gaze_cursor and gaze_point:
                gx, gy = int(self.gaze_target[0]), int(self.gaze_target[1])
                cv2.circle(canvas, (gx, gy), 20, (255, 255, 255), 2)
                cv2.circle(canvas, (gx, gy), 3, (255, 255, 255), -1)

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time)
            last_time = current_time
            fps_counter.append(fps)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            avg_fps = np.mean(fps_counter)

            cv2.putText(canvas, f"FPS: {avg_fps:.1f}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(canvas, f"Boids: {len(self.boids)}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('Boids', canvas)

            # Show webcam feed with optional debug overlay in corner
            if self.debug_mode and landmarks is not None:
                debug_frame = frame.copy()
                debug_frame = self.tracker.draw_debug_landmarks(debug_frame, landmarks)
                small_debug = cv2.resize(debug_frame, (320, 240))
                cv2.imshow('Debug View', small_debug)
            elif not self.debug_mode:
                try:
                    cv2.destroyWindow('Debug View')
                except:
                    pass  # Window doesn't exist yet

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                self.show_gaze_cursor = not self.show_gaze_cursor
            elif key == ord('d'):
                # Toggle debug mode
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('r'):
                cv2.destroyAllWindows()
                calibrator = CalibrationUI(self.screen_width, self.screen_height)
                new_tracker = calibrator.run_calibration()
                if new_tracker:
                    self.tracker = new_tracker
                cv2.namedWindow('Boids', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Boids', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting Eye-Following Boids simulation...")
    print("First, we need to calibrate the gaze tracker.\n")

    # Run calibration
    calibrator = CalibrationUI(screen_width=1920, screen_height=1080, grid_size=4)
    tracker = calibrator.run_calibration()

    if tracker:
        # Run boids simulation (optimized - can handle 200+ boids now)
        sim = BoidsGazeFollower(tracker, screen_width=1920, screen_height=1080, num_boids=200)
        sim.run()
    else:
        print("Calibration failed. Exiting.")
