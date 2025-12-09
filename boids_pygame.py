import pygame
import numpy as np
import cv2
from calibration import CalibrationUI
from boids import Boid

class BoidsGazeFollowerPygame:
    def __init__(self, tracker, screen_width=1920, screen_height=1080, num_boids=200):
        self.tracker = tracker
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
        pygame.display.set_caption('Boids Eye Follower')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

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
        self.debug_mode = False

        # Webcam for gaze tracking
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def run(self):
        """Run the boids simulation with gaze tracking"""
        print("=== Boids Eye Follower (Pygame) ===")
        print("The particles will follow your gaze!")
        print("Press 'c' to toggle gaze cursor")
        print("Press 'd' to toggle debug visualization")
        print("Press 'q' or ESC to quit")

        running = True
        frame_times = []

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
                    elif event.key == pygame.K_d:
                        self.debug_mode = not self.debug_mode
                        print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")

            # Read webcam frame
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                # Predict gaze
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

            # Clear screen
            self.screen.fill((0, 0, 0))

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
                    100
                )

                # Draw boid as triangle
                if np.linalg.norm(boid.vel) > 0:
                    angle = np.arctan2(boid.vel[1], boid.vel[0])
                else:
                    angle = 0

                size = 8
                points = [
                    [size, 0],
                    [-size, size/2],
                    [-size, -size/2]
                ]

                # Rotation
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rotated = []
                for p in points:
                    rx = p[0] * cos_a - p[1] * sin_a
                    ry = p[0] * sin_a + p[1] * cos_a
                    rotated.append([rx + boid.pos[0], ry + boid.pos[1]])

                pygame.draw.polygon(self.screen, color, rotated)

            # Draw gaze cursor
            if self.show_gaze_cursor and gaze_point:
                gx, gy = int(self.gaze_target[0]), int(self.gaze_target[1])
                pygame.draw.circle(self.screen, (255, 255, 255), (gx, gy), 20, 2)
                pygame.draw.circle(self.screen, (255, 255, 255), (gx, gy), 3)

            # Calculate and display FPS
            current_fps = self.clock.get_fps()
            frame_times.append(current_fps)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = np.mean(frame_times) if frame_times else 0

            fps_text = self.font.render(f"FPS: {avg_fps:.1f}", True, (255, 255, 255))
            self.screen.blit(fps_text, (20, 20))

            boids_text = self.font.render(f"Boids: {len(self.boids)}", True, (255, 255, 255))
            self.screen.blit(boids_text, (20, 60))

            # Show debug webcam if enabled
            if self.debug_mode and ret:
                if landmarks is not None:
                    debug_frame = self.tracker.draw_debug_landmarks(frame.copy(), landmarks)
                else:
                    debug_frame = frame

                # Convert OpenCV BGR to Pygame RGB
                debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
                debug_frame = cv2.resize(debug_frame, (320, 240))
                debug_surface = pygame.surfarray.make_surface(np.rot90(debug_frame))
                self.screen.blit(debug_surface, (self.screen_width - 330, 10))

            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # Target 60 FPS

        self.cap.release()
        pygame.quit()


if __name__ == "__main__":
    print("Starting Eye-Following Boids simulation (Pygame version)...")
    print("First, we need to calibrate the gaze tracker.\n")

    # Run calibration
    calibrator = CalibrationUI(screen_width=1920, screen_height=1080, grid_size=4)
    tracker = calibrator.run_calibration()

    if tracker:
        # Run boids simulation with Pygame (much faster!)
        sim = BoidsGazeFollowerPygame(tracker, screen_width=1920, screen_height=1080, num_boids=300)
        sim.run()
    else:
        print("Calibration failed. Exiting.")
