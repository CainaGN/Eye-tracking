import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from enum import Enum, auto

pyautogui.FAILSAFE = False

class FaceMeshConfig:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        self.tesselation_style = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()

class CalibrationPhase(Enum):
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()

class EyeTracker:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.video_capture = cv2.VideoCapture(0)
        
        self.eye_aspect_ratio_threshold = 0.21
        self.consecutive_frames_for_blink = 3
        self.blink_counter = 0
        self.blink_detected = False
        
        self.calibration_bounds = {
            CalibrationPhase.TOP: None,
            CalibrationPhase.BOTTOM: None,
            CalibrationPhase.LEFT: None,
            CalibrationPhase.RIGHT: None
        }
        self.current_calibration_phase = CalibrationPhase.TOP
        self.calibration_in_progress = True
        
        self.face_processor = FaceMeshConfig()
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]

    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        eye_points = np.array([(landmarks[idx].x, landmarks[idx].y) for idx in eye_indices])
        eye_points *= np.array([self.frame_width, self.frame_height])
        
        vertical_distances = [
            np.linalg.norm(eye_points[1] - eye_points[5]),
            np.linalg.norm(eye_points[2] - eye_points[4])
        ]
        horizontal_distance = np.linalg.norm(eye_points[0] - eye_points[3])
        
        return sum(vertical_distances) / (2.0 * horizontal_distance)

    def map_head_position_to_screen(self, head_position):
        calibration = self.calibration_bounds
        screen_x = np.interp(head_position[0], 
                           [calibration[CalibrationPhase.LEFT][0], calibration[CalibrationPhase.RIGHT][0]], 
                           [10, self.screen_width - 10])
        screen_y = np.interp(head_position[1],
                           [calibration[CalibrationPhase.TOP][1], calibration[CalibrationPhase.BOTTOM][1]],
                           [10, self.screen_height - 10])
        return np.clip([screen_x, screen_y], [0, 0], [self.screen_width, self.screen_height])
        
    def smooth_mouse_movement(self, target_position):
        current_x, current_y = pyautogui.position()
        smoothing_factor = 0.2
        new_x = int(current_x + (target_position[0] - current_x) * smoothing_factor)
        new_y = int(current_y + (target_position[1] - current_y) * smoothing_factor)
        return new_x, new_y

    def process_calibration(self, landmarks):
        cv2.putText(self.frame, 
                   f"Look {self.current_calibration_phase.name.lower()} and press 'c'",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if cv2.waitKey(1) & 0xFF == ord('c'):
            self.calibration_bounds[self.current_calibration_phase] = (
                landmarks[1].x * self.frame_width,
                landmarks[1].y * self.frame_height
            )
            
            if self.current_calibration_phase == CalibrationPhase.RIGHT:
                self.calibration_in_progress = False
                print("Calibration complete:", self.calibration_bounds)
            else:
                self.current_calibration_phase = CalibrationPhase(self.current_calibration_phase.value + 1)

    def process_face_movement(self, landmarks):
        nose_position = (
            landmarks[1].x * self.frame_width,
            landmarks[1].y * self.frame_height
        )
        
        screen_position = self.map_head_position_to_screen(nose_position)
        smoothed_position = self.smooth_mouse_movement(screen_position)
        pyautogui.moveTo(*smoothed_position)
        
        cv2.putText(self.frame, f"Cursor: {smoothed_position}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def detect_blink(self, landmarks):
        left_eye_ratio = self.calculate_eye_aspect_ratio(landmarks, self.left_eye_indices)
        right_eye_ratio = self.calculate_eye_aspect_ratio(landmarks, self.right_eye_indices)
        average_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if average_ratio < self.eye_aspect_ratio_threshold:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.consecutive_frames_for_blink:
                self.blink_detected = True
            self.blink_counter = 0

        if self.blink_detected:
            pyautogui.click()
            self.blink_detected = False

    def run(self):
        while self.video_capture.isOpened():
            success, self.frame = self.video_capture.read()
            if not success:
                break
                
            self.frame = cv2.flip(self.frame, 1)
            self.frame_height, self.frame_width, _ = self.frame.shape
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            
            face_landmarks = self.face_processor.face_mesh.process(rgb_frame)
            
            if face_landmarks.multi_face_landmarks:
                for landmarks in face_landmarks.multi_face_landmarks:
                    if self.calibration_in_progress:
                        self.process_calibration(landmarks.landmark)
                        break
                    else:
                        self.process_face_movement(landmarks.landmark)
                        self.detect_blink(landmarks.landmark)
                    
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=self.frame,
                        landmark_list=landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.face_processor.drawing_spec,
                        connection_drawing_spec=self.face_processor.tesselation_style
                    )

            cv2.imshow('Head Tracking Interface', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = EyeTracker()
    tracker.run()
