"""Highly optimized boids with Pygame - uses NumPy vectorization"""
import pygame
import numpy as np
import cv2
import os
from calibration import CalibrationUI
from gaze import GazeTracker

class BoidsFast:
    def __init__(self, tracker, screen_width=1920, screen_height=1080, num_boids=500):
        self.tracker = tracker
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_boids = num_boids

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN | pygame.DOUBLEBUF)
        pygame.display.set_caption('Boids Eye Follower - Optimized')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # Boid properties (vectorized with NumPy)
        self.positions = np.random.rand(num_boids, 2) * [screen_width, screen_height]
        self.velocities = (np.random.rand(num_boids, 2) - 0.5) * 4
        self.accelerations = np.zeros((num_boids, 2))

        # Boid parameters
        self.max_speed = 6.0
        self.max_force = 0.3

        # Gaze tracking
        self.gaze_target = np.array([screen_width / 2, screen_height / 2])
        self.show_gaze_cursor = True

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower res for speed
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Pre-allocate surface for boids
        self.boid_surface = pygame.Surface((screen_width, screen_height))

    def limit_vector(self, vec, max_val):
        """Limit magnitude of vectors"""
        mags = np.linalg.norm(vec, axis=1, keepdims=True)
        mags = np.maximum(mags, 1e-6)  # Avoid division by zero
        scale = np.minimum(mags, max_val) / mags
        return vec * scale

    def flock(self, gaze_pos):
        """Vectorized flocking behavior - MUCH faster!"""
        # Reset accelerations
        self.accelerations.fill(0)

        # For each boid, calculate forces (vectorized where possible)
        for i in range(self.num_boids):
            pos = self.positions[i]
            vel = self.velocities[i]

            # Calculate distances to all other boids
            diff = self.positions - pos
            dists = np.linalg.norm(diff, axis=1)
            dists[i] = np.inf  # Ignore self

            # Separation (avoid crowding) - only nearby boids
            sep_mask = (dists < 25) & (dists > 0)
            if np.any(sep_mask):
                sep_diff = diff[sep_mask]
                sep_dists = dists[sep_mask][:, np.newaxis]
                sep_force = np.sum(sep_diff / sep_dists, axis=0)
                if np.linalg.norm(sep_force) > 0:
                    sep_force = (sep_force / np.linalg.norm(sep_force)) * self.max_speed - vel
                    sep_force = self.limit_vector(sep_force.reshape(1, -1), self.max_force)[0]
                    self.accelerations[i] += sep_force * 0.8  # Reduced from 1.5

            # Alignment (match velocity) - medium range
            align_mask = (dists < 50) & (dists > 0)
            if np.any(align_mask):
                avg_vel = np.mean(self.velocities[align_mask], axis=0)
                if np.linalg.norm(avg_vel) > 0:
                    avg_vel = (avg_vel / np.linalg.norm(avg_vel)) * self.max_speed
                align_force = avg_vel - vel
                align_force = self.limit_vector(align_force.reshape(1, -1), self.max_force)[0]
                self.accelerations[i] += align_force * 0.6  # Reduced from 1.0

            # Cohesion (move toward center) - medium range
            coh_mask = (dists < 50) & (dists > 0)
            if np.any(coh_mask):
                avg_pos = np.mean(self.positions[coh_mask], axis=0)
                desired = avg_pos - pos
                dist_to_center = np.linalg.norm(desired)
                if dist_to_center > 0:
                    desired = (desired / dist_to_center) * self.max_speed
                    if dist_to_center < 100:
                        desired *= dist_to_center / 100
                    coh_force = desired - vel
                    coh_force = self.limit_vector(coh_force.reshape(1, -1), self.max_force)[0]
                    self.accelerations[i] += coh_force * 0.6  # Reduced from 1.0

            # Seek gaze target - SUBTLE influence
            to_gaze = gaze_pos - pos
            dist_to_gaze = np.linalg.norm(to_gaze)
            if dist_to_gaze > 0:
                desired = (to_gaze / dist_to_gaze) * self.max_speed
                if dist_to_gaze < 100:
                    desired *= dist_to_gaze / 100
                seek_force = desired - vel
                seek_force = self.limit_vector(seek_force.reshape(1, -1), self.max_force)[0]
                # REDUCED gaze influence from 2.0 to 0.5 for subtle effect
                self.accelerations[i] += seek_force * 0.5

    def update(self):
        """Update all boid positions (vectorized)"""
        # Update velocities
        self.velocities += self.accelerations

        # Limit speed (vectorized)
        self.velocities = self.limit_vector(self.velocities, self.max_speed)

        # Update positions
        self.positions += self.velocities

        # Wrap around screen edges
        self.positions[:, 0] = np.mod(self.positions[:, 0], self.screen_width)
        self.positions[:, 1] = np.mod(self.positions[:, 1], self.screen_height)

    def draw(self):
        """Draw boids - optimized rendering"""
        # Clear boid surface
        self.boid_surface.fill((0, 0, 0))

        # Calculate colors based on distance to gaze
        dists = np.linalg.norm(self.positions - self.gaze_target, axis=1)
        intensities = np.maximum(0, 1 - dists / 500)

        # Draw each boid as a small circle (faster than polygons)
        for i in range(self.num_boids):
            intensity = intensities[i]
            color = (
                int(100 + 155 * intensity),
                int(255 * intensity),
                100
            )
            pos = self.positions[i].astype(int)
            pygame.draw.circle(self.boid_surface, color, pos, 4)

        # Blit to screen
        self.screen.blit(self.boid_surface, (0, 0))

    def run(self):
        """Main game loop"""
        print("=== Boids Eye Follower (Highly Optimized) ===")
        print(f"Running with {self.num_boids} boids")
        print("Press 'c' to toggle gaze cursor")
        print("Press 'q' or ESC to quit")

        running = True
        frame_count = 0

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c:
                        self.show_gaze_cursor = not self.show_gaze_cursor

            # Update gaze every frame
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                gaze_point = self.tracker.predict_gaze(frame)

                if gaze_point:
                    gaze_x, gaze_y = gaze_point
                    self.gaze_target = np.array([
                        np.clip(gaze_x, 0, self.screen_width),
                        np.clip(gaze_y, 0, self.screen_height)
                    ])

            # Update physics
            self.flock(self.gaze_target)
            self.update()

            # Render
            self.draw()

            # Draw gaze cursor
            if self.show_gaze_cursor:
                gx, gy = self.gaze_target.astype(int)
                pygame.draw.circle(self.screen, (255, 255, 255), (gx, gy), 20, 2)
                pygame.draw.circle(self.screen, (255, 255, 255), (gx, gy), 3)

            # Display FPS
            fps = self.clock.get_fps()
            fps_text = self.font.render(f"FPS: {fps:.1f} | Boids: {self.num_boids}", True, (255, 255, 255))
            self.screen.blit(fps_text, (20, 20))

            # Update display
            pygame.display.flip()
            self.clock.tick(60)

            frame_count += 1

        self.cap.release()
        pygame.quit()


if __name__ == "__main__":
    print("Starting Eye-Following Boids (Fast Version)...")

    # Try to load existing calibration
    tracker = GazeTracker()
    if os.path.exists('calibration.pkl'):
        print("\nFound saved calibration!")
        if tracker.load_calibration():
            print("Using saved calibration. Press 'r' to recalibrate if needed.\n")
        else:
            print("Failed to load. Running calibration...\n")
            calibrator = CalibrationUI(screen_width=1920, screen_height=1080, grid_size=4)
            tracker = calibrator.run_calibration()
    else:
        print("No saved calibration found. Running calibration...\n")
        calibrator = CalibrationUI(screen_width=1920, screen_height=1080, grid_size=4)
        tracker = calibrator.run_calibration()

    if tracker and tracker.is_calibrated:
        # Run optimized boids simulation
        sim = BoidsFast(tracker, screen_width=1920, screen_height=1080, num_boids=500)
        sim.run()
    else:
        print("Calibration failed. Exiting.")
