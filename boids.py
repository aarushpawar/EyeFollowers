"""Ultra-optimized boids with spatial partitioning and dynamic boid count"""
import pygame
import numpy as np
import cv2
import os
from calibration import CalibrationUI
from gaze import GazeTracker
from collections import defaultdict


class SpatialGrid:
    """Spatial partitioning grid for O(n) neighbor lookups"""
    def __init__(self, width, height, cell_size=50):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = int(np.ceil(width / cell_size))
        self.rows = int(np.ceil(height / cell_size))
        self.grid = defaultdict(list)

    def clear(self):
        self.grid.clear()

    def get_cell(self, x, y):
        """Get grid cell coordinates for position"""
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        return (max(0, min(col, self.cols - 1)),
                max(0, min(row, self.rows - 1)))

    def insert(self, boid_idx, x, y):
        """Insert boid into grid"""
        cell = self.get_cell(x, y)
        self.grid[cell].append(boid_idx)

    def get_neighbors(self, x, y, radius=2):
        """Get all boids in neighboring cells (radius in cells)"""
        col, row = self.get_cell(x, y)
        neighbors = []

        for dc in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                c = col + dc
                r = row + dr
                if 0 <= c < self.cols and 0 <= r < self.rows:
                    neighbors.extend(self.grid.get((c, r), []))

        return neighbors


class BoidsOptimized:
    def __init__(self, tracker, screen_width=1920, screen_height=1080,
                 initial_boids=300, target_fps=60):
        self.tracker = tracker
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.target_fps = target_fps

        # Dynamic boid management
        self.num_boids = initial_boids
        self.min_boids = 50
        self.max_boids = 2000
        self.fps_history = []
        self.adjustment_cooldown = 0

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (screen_width, screen_height),
            pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
        )
        pygame.display.set_caption('Boids Eye Follower - Ultra Optimized')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # Spatial partitioning
        self.spatial_grid = SpatialGrid(screen_width, screen_height, cell_size=75)

        # Boid properties (vectorized)
        self._initialize_boids(initial_boids)

        # Boid parameters
        self.max_speed = 6.0
        self.max_force = 0.3
        self.perception_radius = 75

        # Gaze tracking
        self.gaze_target = np.array([screen_width / 2, screen_height / 2])
        self.show_gaze_cursor = True
        self.show_stats = True

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Performance tracking
        self.frame_count = 0
        self.last_adjustment_time = 0

    def _initialize_boids(self, count):
        """Initialize or resize boid arrays"""
        self.positions = np.random.rand(count, 2) * [self.screen_width, self.screen_height]
        self.velocities = (np.random.rand(count, 2) - 0.5) * 4
        self.accelerations = np.zeros((count, 2))
        self.num_boids = count

    def _add_boids(self, count):
        """Add more boids dynamically"""
        new_positions = np.random.rand(count, 2) * [self.screen_width, self.screen_height]
        new_velocities = (np.random.rand(count, 2) - 0.5) * 4
        new_accelerations = np.zeros((count, 2))

        self.positions = np.vstack([self.positions, new_positions])
        self.velocities = np.vstack([self.velocities, new_velocities])
        self.accelerations = np.vstack([self.accelerations, new_accelerations])
        self.num_boids += count

    def _remove_boids(self, count):
        """Remove boids dynamically"""
        if self.num_boids - count < self.min_boids:
            count = self.num_boids - self.min_boids

        if count > 0:
            self.positions = self.positions[:-count]
            self.velocities = self.velocities[:-count]
            self.accelerations = self.accelerations[:-count]
            self.num_boids -= count

    def adjust_boid_count(self, current_fps):
        """Dynamically adjust boid count to maintain target FPS"""
        self.fps_history.append(current_fps)
        if len(self.fps_history) > 60:
            self.fps_history.pop(0)

        # Only adjust every 2 seconds
        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
            return

        if len(self.fps_history) < 30:
            return

        avg_fps = np.mean(self.fps_history)
        fps_margin = 5

        # If FPS is too low, reduce boids
        if avg_fps < self.target_fps - fps_margin and self.num_boids > self.min_boids:
            reduction = max(10, int(self.num_boids * 0.05))
            self._remove_boids(reduction)
            self.adjustment_cooldown = 120  # 2 seconds at 60fps
            print(f"FPS low ({avg_fps:.1f}), reducing to {self.num_boids} boids")

        # If FPS is comfortably high, add more boids
        elif avg_fps > self.target_fps + 10 and self.num_boids < self.max_boids:
            increase = max(10, int(self.num_boids * 0.05))
            self._add_boids(increase)
            self.adjustment_cooldown = 120
            print(f"FPS high ({avg_fps:.1f}), increasing to {self.num_boids} boids")

    def limit_magnitude(self, vec, max_val):
        """Limit magnitude of a 2D vector"""
        mag = np.linalg.norm(vec)
        if mag > max_val:
            return (vec / mag) * max_val
        return vec

    def flock_optimized(self):
        """Optimized flocking using spatial partitioning"""
        # Reset accelerations
        self.accelerations.fill(0)

        # Build spatial grid
        self.spatial_grid.clear()
        for i in range(self.num_boids):
            self.spatial_grid.insert(i, self.positions[i, 0], self.positions[i, 1])

        # Process each boid
        for i in range(self.num_boids):
            pos = self.positions[i]
            vel = self.velocities[i]

            # Get nearby boids using spatial grid (HUGE optimization!)
            nearby_indices = self.spatial_grid.get_neighbors(pos[0], pos[1], radius=1)

            if len(nearby_indices) > 1:  # More than just self
                # Pre-compute differences and distances for nearby boids
                nearby_positions = self.positions[nearby_indices]
                nearby_velocities = self.velocities[nearby_indices]

                diff = nearby_positions - pos
                dist_sq = np.sum(diff * diff, axis=1)
                dists = np.sqrt(dist_sq)

                # Remove self
                mask = dists > 0
                diff = diff[mask]
                dists = dists[mask]
                nearby_velocities = nearby_velocities[mask]
                nearby_positions = nearby_positions[mask]

                if len(dists) > 0:
                    # Separation
                    sep_mask = dists < 25
                    if np.any(sep_mask):
                        sep_diff = diff[sep_mask]
                        sep_dists = dists[sep_mask][:, np.newaxis]
                        sep_force = np.sum(sep_diff / (sep_dists + 1e-6), axis=0)
                        sep_mag = np.linalg.norm(sep_force)
                        if sep_mag > 0:
                            sep_force = (sep_force / sep_mag) * self.max_speed - vel
                            sep_force = self.limit_magnitude(sep_force, self.max_force)
                            self.accelerations[i] += sep_force * 1.5

                    # Alignment
                    align_mask = dists < 50
                    if np.any(align_mask):
                        avg_vel = np.mean(nearby_velocities[align_mask], axis=0)
                        avg_mag = np.linalg.norm(avg_vel)
                        if avg_mag > 0:
                            avg_vel = (avg_vel / avg_mag) * self.max_speed
                            align_force = avg_vel - vel
                            align_force = self.limit_magnitude(align_force, self.max_force)
                            self.accelerations[i] += align_force * 1.0

                    # Cohesion
                    coh_mask = dists < 50
                    if np.any(coh_mask):
                        avg_pos = np.mean(nearby_positions[coh_mask], axis=0)
                        desired = avg_pos - pos
                        dist_to_center = np.linalg.norm(desired)
                        if dist_to_center > 0:
                            desired = (desired / dist_to_center) * self.max_speed
                            if dist_to_center < 100:
                                desired *= dist_to_center / 100
                            coh_force = desired - vel
                            coh_force = self.limit_magnitude(coh_force, self.max_force)
                            self.accelerations[i] += coh_force * 1.0

            # Seek gaze target
            to_gaze = self.gaze_target - pos
            dist_to_gaze = np.linalg.norm(to_gaze)
            if dist_to_gaze > 0:
                desired = (to_gaze / dist_to_gaze) * self.max_speed
                if dist_to_gaze < 100:
                    desired *= dist_to_gaze / 100
                seek_force = desired - vel
                seek_force = self.limit_magnitude(seek_force, self.max_force)
                self.accelerations[i] += seek_force * 2.0

    def update(self):
        """Update all boid positions"""
        # Update velocities
        self.velocities += self.accelerations

        # Limit speed (vectorized)
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        mask = speeds > self.max_speed
        self.velocities[mask.flatten()] = (
            self.velocities[mask.flatten()] / speeds[mask] * self.max_speed
        )

        # Update positions
        self.positions += self.velocities

        # Wrap around screen edges (vectorized)
        self.positions[:, 0] = np.mod(self.positions[:, 0], self.screen_width)
        self.positions[:, 1] = np.mod(self.positions[:, 1], self.screen_height)

    def draw_optimized(self):
        """Optimized batch rendering"""
        # Clear screen
        self.screen.fill((0, 0, 0))

        # Calculate colors based on distance to gaze (vectorized)
        dists = np.linalg.norm(self.positions - self.gaze_target, axis=1)
        intensities = np.maximum(0, 1 - dists / 500)

        # Batch draw circles (faster than polygons)
        for i in range(self.num_boids):
            intensity = intensities[i]
            color = (
                int(100 + 155 * intensity),
                int(255 * intensity),
                100
            )
            pos = self.positions[i].astype(int)
            pygame.draw.circle(self.screen, color, pos, 4)

    def run(self):
        """Main game loop"""
        print("=== Boids Eye Follower (Ultra Optimized) ===")
        print(f"Starting with {self.num_boids} boids")
        print(f"Target FPS: {self.target_fps}")
        print("Dynamic boid adjustment enabled")
        print("\nControls:")
        print("  'c' - Toggle gaze cursor")
        print("  's' - Toggle stats")
        print("  'q' or ESC - Quit")

        running = True

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
                    elif event.key == pygame.K_s:
                        self.show_stats = not self.show_stats

            # Update gaze (skip some frames for performance)
            if self.frame_count % 2 == 0:
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
            self.flock_optimized()
            self.update()

            # Render
            self.draw_optimized()

            # Draw gaze cursor
            if self.show_gaze_cursor:
                gx, gy = self.gaze_target.astype(int)
                pygame.draw.circle(self.screen, (255, 255, 255), (gx, gy), 20, 2)
                pygame.draw.circle(self.screen, (255, 255, 255), (gx, gy), 3)

            # Display stats
            if self.show_stats:
                fps = self.clock.get_fps()
                fps_text = self.font.render(
                    f"FPS: {fps:.1f} | Boids: {self.num_boids} | Target: {self.target_fps}",
                    True, (255, 255, 255)
                )
                self.screen.blit(fps_text, (20, 20))

                # FPS color indicator
                if len(self.fps_history) > 10:
                    avg_fps = np.mean(self.fps_history[-30:])
                    if avg_fps < self.target_fps - 5:
                        status_color = (255, 100, 100)  # Red
                        status = "LOW"
                    elif avg_fps > self.target_fps + 5:
                        status_color = (100, 255, 100)  # Green
                        status = "HIGH"
                    else:
                        status_color = (255, 255, 100)  # Yellow
                        status = "GOOD"

                    status_text = self.font.render(f"Status: {status}", True, status_color)
                    self.screen.blit(status_text, (20, 60))

            # Update display
            pygame.display.flip()

            # Dynamic FPS adjustment
            current_fps = self.clock.get_fps()
            if current_fps > 0:
                self.adjust_boid_count(current_fps)

            self.clock.tick(self.target_fps)
            self.frame_count += 1

        self.cap.release()
        pygame.quit()

        # Print final stats
        if self.fps_history:
            print(f"\nFinal Stats:")
            print(f"  Average FPS: {np.mean(self.fps_history):.1f}")
            print(f"  Final boid count: {self.num_boids}")


if __name__ == "__main__":
    print("Starting Eye-Following Boids (Ultra Optimized)...")

    # Try to load existing calibration
    tracker = GazeTracker()
    if os.path.exists('calibration.pkl'):
        print("\nFound saved calibration!")
        if tracker.load_calibration():
            print("Using saved calibration.\n")
        else:
            print("Failed to load. Running calibration...\n")
            calibrator = CalibrationUI(screen_width=1920, screen_height=1080, grid_size=4)
            tracker = calibrator.run_calibration()
    else:
        print("No saved calibration found. Running calibration...\n")
        calibrator = CalibrationUI(screen_width=1920, screen_height=1080, grid_size=4)
        tracker = calibrator.run_calibration()

    if tracker and tracker.is_calibrated:
        # Run ultra-optimized boids simulation
        sim = BoidsOptimized(
            tracker,
            screen_width=1920,
            screen_height=1080,
            initial_boids=300,
            target_fps=60
        )
        sim.run()
    else:
        print("Calibration failed. Exiting.")
