import cv2
import numpy as np
import mediapipe as mp
from scipy.interpolate import RBFInterpolator
from collections import deque
import pickle
import os

class GazeTracker:
    def __init__(self, debug=False):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,  # Increased for better quality
            min_tracking_confidence=0.7     # Increased for better quality
        )

        self.calibration_points = []
        self.calibration_features = []

        self.model_x = None
        self.model_y = None

        self.gaze_history = deque(maxlen=15)  # Larger buffer for more stable smoothing
        self.is_calibrated = False
        self.debug = debug

    def extract_head_pose(self, landmarks, frame_shape):
        """Extract head pose features for normalization"""
        h, w = frame_shape[:2]

        # Key face points for head pose estimation
        nose_tip = landmarks[1]
        chin = landmarks[152]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]

        # Convert to pixel coordinates
        points_2d = np.array([
            [nose_tip.x * w, nose_tip.y * h],
            [chin.x * w, chin.y * h],
            [left_eye.x * w, left_eye.y * h],
            [right_eye.x * w, right_eye.y * h],
            [left_mouth.x * w, left_mouth.y * h],
            [right_mouth.x * w, right_mouth.y * h]
        ])

        # Simple head pose features: normalized distances and angles
        eye_center = (points_2d[2] + points_2d[3]) / 2
        nose_to_eye = nose_tip.x * w - eye_center[0], nose_tip.y * h - eye_center[1]
        head_width = np.linalg.norm(points_2d[2] - points_2d[3])

        return np.array([
            nose_to_eye[0] / head_width,
            nose_to_eye[1] / head_width,
            (right_eye.x - left_eye.x),
            (right_eye.y - left_eye.y)
        ])

    def extract_eye(self, landmarks, frame_shape, iris_indices, corner_indices, upper_lid, lower_lid):
        """Extract eye features including iris position and eye openness"""
        h, w = frame_shape[:2]

        # Iris center
        iris = [landmarks[i] for i in iris_indices]
        iris_center = np.mean([[p.x * w, p.y * h] for p in iris], axis=0)

        # Eye corners
        corners = [landmarks[i] for i in corner_indices]
        corner_coords = np.array([[p.x * w, p.y * h] for p in corners])

        # Eye width
        eye_width = np.linalg.norm(corner_coords[1] - corner_coords[0])

        # Relative iris position (normalized by eye width)
        relative_pos = (iris_center - corner_coords[0]) / eye_width

        # Eye openness (vertical distance between lids)
        upper_coord = np.array([landmarks[upper_lid].x * w, landmarks[upper_lid].y * h])
        lower_coord = np.array([landmarks[lower_lid].x * w, landmarks[lower_lid].y * h])
        eye_openness = np.linalg.norm(upper_coord - lower_coord) / eye_width

        return np.concatenate([relative_pos, [eye_openness]])

    def extract_features(self, landmarks, frame_shape):
        """Extract all gaze-relevant features"""
        # Extract features for both eyes
        left_features = self.extract_eye(
            landmarks, frame_shape,
            iris_indices=range(468, 473),
            corner_indices=[33, 133],
            upper_lid=159,
            lower_lid=145
        )

        right_features = self.extract_eye(
            landmarks, frame_shape,
            iris_indices=range(473, 478),
            corner_indices=[362, 263],
            upper_lid=386,
            lower_lid=374
        )

        # Extract head pose features
        head_pose = self.extract_head_pose(landmarks, frame_shape)

        # Combine all features
        return np.concatenate([left_features, right_features, head_pose])

    def get_features_only(self, frame):
        """Extract features without adding to calibration (for outlier filtering)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            features = self.extract_features(landmarks, frame.shape)
            return features
        return None

    def add_calibration_features(self, features, target_x, target_y):
        """Add pre-computed features to calibration (used after outlier filtering)"""
        self.calibration_features.append(features)
        self.calibration_points.append([target_x, target_y])

    def add_calibration_point(self, frame, target_x, target_y):
        """Add a calibration sample"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            features = self.extract_features(landmarks, frame.shape)

            self.calibration_features.append(features)
            self.calibration_points.append([target_x, target_y])
            return True
        return False

    def calibrate(self):
        """Build the gaze mapping model from calibration data"""
        # RBF with 10-dimensional features needs at least 11 points
        min_points = 11
        if len(self.calibration_points) < min_points:
            print(f"Not enough calibration points. Need at least {min_points}, got {len(self.calibration_points)}")
            return False

        features = np.array(self.calibration_features)
        points = np.array(self.calibration_points)

        # Use RBF interpolation for smooth mapping
        # Smoothing parameter balances fit vs generalization (0.1 = tight fit, 1.0 = very smooth)
        try:
            self.model_x = RBFInterpolator(features, points[:, 0], kernel='thin_plate_spline', smoothing=0.3)
            self.model_y = RBFInterpolator(features, points[:, 1], kernel='thin_plate_spline', smoothing=0.3)
            self.is_calibrated = True
            print(f"Calibration successful with {len(self.calibration_points)} points")
            return True
        except Exception as e:
            print(f"Calibration failed: {e}")
            return False

    def predict_gaze(self, frame, return_debug_info=False):
        """Predict gaze point on screen

        Args:
            frame: Input frame from webcam
            return_debug_info: If True, returns (gaze_point, landmarks), else just gaze_point
        """
        if not self.is_calibrated:
            return (None, None) if return_debug_info else None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            features = self.extract_features(landmarks, frame.shape)

            # Predict gaze coordinates
            gaze_x = float(self.model_x(features.reshape(1, -1))[0])
            gaze_y = float(self.model_y(features.reshape(1, -1))[0])

            if self.debug:
                print(f"Raw prediction: ({gaze_x:.1f}, {gaze_y:.1f})")

            # Simple exponential moving average smoothing
            self.gaze_history.append((gaze_x, gaze_y))

            if len(self.gaze_history) >= 3:
                # Use last 3 frames with exponential weighting
                # 50% current, 30% previous, 20% older
                recent = list(self.gaze_history)[-3:]
                smoothed_x = recent[2][0] * 0.5 + recent[1][0] * 0.3 + recent[0][0] * 0.2
                smoothed_y = recent[2][1] * 0.5 + recent[1][1] * 0.3 + recent[0][1] * 0.2
                gaze_point = (smoothed_x, smoothed_y)
            else:
                # Not enough history, use raw
                gaze_point = (gaze_x, gaze_y)

            if self.debug:
                print(f"Smoothed: ({gaze_point[0]:.1f}, {gaze_point[1]:.1f})")

            if return_debug_info:
                return (gaze_point, landmarks)

            return gaze_point

        return (None, None) if return_debug_info else None

    def draw_debug_landmarks(self, frame, landmarks):
        """Draw face mesh landmarks and iris positions for debugging"""
        h, w = frame.shape[:2]

        # Define key landmark groups
        LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7]
        RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249]
        LEFT_IRIS_INDICES = list(range(468, 473))
        RIGHT_IRIS_INDICES = list(range(473, 478))
        FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        # Draw face oval (light gray)
        for idx in FACE_OVAL_INDICES:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 1, (100, 100, 100), -1)

        # Draw left eye contour (green)
        for idx in LEFT_EYE_INDICES:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Draw right eye contour (green)
        for idx in RIGHT_EYE_INDICES:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Draw left iris (bright cyan)
        for idx in LEFT_IRIS_INDICES:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

        # Draw right iris (bright cyan)
        for idx in RIGHT_IRIS_INDICES:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

        # Draw iris centers with crosshairs
        if len(landmarks) > max(LEFT_IRIS_INDICES):
            left_iris_center = np.mean([[landmarks[i].x * w, landmarks[i].y * h]
                                        for i in LEFT_IRIS_INDICES], axis=0)
            lx, ly = int(left_iris_center[0]), int(left_iris_center[1])
            cv2.circle(frame, (lx, ly), 5, (0, 255, 255), 2)
            cv2.line(frame, (lx - 10, ly), (lx + 10, ly), (0, 255, 255), 1)
            cv2.line(frame, (lx, ly - 10), (lx, ly + 10), (0, 255, 255), 1)

        if len(landmarks) > max(RIGHT_IRIS_INDICES):
            right_iris_center = np.mean([[landmarks[i].x * w, landmarks[i].y * h]
                                         for i in RIGHT_IRIS_INDICES], axis=0)
            rx, ry = int(right_iris_center[0]), int(right_iris_center[1])
            cv2.circle(frame, (rx, ry), 5, (0, 255, 255), 2)
            cv2.line(frame, (rx - 10, ry), (rx + 10, ry), (0, 255, 255), 1)
            cv2.line(frame, (rx, ry - 10), (rx, ry + 10), (0, 255, 255), 1)

        # Draw key face points for head pose (red)
        key_points = [1, 152, 33, 263, 61, 291]  # nose, chin, eyes, mouth corners
        for idx in key_points:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # Add feature value overlay
        if self.is_calibrated:
            features = self.extract_features(landmarks, frame.shape)
            cv2.putText(frame, f"Features: {len(features)} dims", (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Show iris position features (first 6 values)
            feature_str = f"L_iris: ({features[0]:.2f}, {features[1]:.2f}) R_iris: ({features[3]:.2f}, {features[4]:.2f})"
            cv2.putText(frame, feature_str, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def save_calibration(self, filename='calibration.pkl'):
        """Save calibration data to file"""
        if not self.is_calibrated:
            print("No calibration to save!")
            return False

        data = {
            'calibration_points': self.calibration_points,
            'calibration_features': self.calibration_features
        }

        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Calibration saved to {filename}")
            return True
        except Exception as e:
            print(f"✗ Failed to save calibration: {e}")
            return False

    def load_calibration(self, filename='calibration.pkl'):
        """Load calibration data from file"""
        if not os.path.exists(filename):
            print(f"Calibration file {filename} not found")
            return False

        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            self.calibration_points = data['calibration_points']
            self.calibration_features = data['calibration_features']

            # Rebuild the models
            if self.calibrate():
                print(f"✓ Calibration loaded from {filename}")
                return True
            else:
                print("✗ Failed to rebuild calibration model")
                return False
        except Exception as e:
            print(f"✗ Failed to load calibration: {e}")
            return False

    def reset_calibration(self):
        """Clear calibration data"""
        self.calibration_points = []
        self.calibration_features = []
        self.model_x = None
        self.model_y = None
        self.is_calibrated = False
        self.gaze_history.clear()
