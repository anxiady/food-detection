"""
MediaPipe keypoint detection and tracking module.
"""

import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import config
from utils import smooth_coordinates, normalize_coordinates, denormalize_coordinates


class KeypointTracker:
    """
    Handles MediaPipe-based keypoint detection for hands and face.
    """
    
    def __init__(self):
        """Initialize MediaPipe models and tracking buffers."""
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.HANDS_MAX_NUM,
            min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_DETECTION_CONFIDENCE
        )
        
        # Initialize face mesh detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_DETECTION_CONFIDENCE
        )
        
        # Tracking buffers for smoothing
        self.hand_buffers = {}  # Buffer for each hand
        self.face_buffer = {}  # Buffer for face keypoints
        
        # Frame dimensions
        self.frame_width = 0
        self.frame_height = 0
        
    def set_frame_dimensions(self, width: int, height: int):
        """Set frame dimensions for coordinate normalization."""
        self.frame_width = width
        self.frame_height = height
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame to detect hands and face keypoints.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Dictionary containing detected keypoints and metadata
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set frame dimensions if not set
        if self.frame_width == 0 or self.frame_height == 0:
            self.set_frame_dimensions(frame.shape[1], frame.shape[0])
        
        # Process hands
        hand_results = self.hands.process(rgb_frame)
        hands_data = self._extract_hand_keypoints(hand_results, frame.shape)
        
        # Process face
        face_results = self.face_mesh.process(rgb_frame)
        face_data = self._extract_face_keypoints(face_results, frame.shape)
        
        return {
            'hands': hands_data,
            'face': face_data,
            'frame_shape': frame.shape
        }
    
    def _extract_hand_keypoints(self, hand_results, frame_shape: Tuple[int, int, int]) -> List[Dict]:
        """
        Extract hand keypoints from MediaPipe results.
        
        Args:
            hand_results: MediaPipe hand detection results
            frame_shape: Frame dimensions (height, width, channels)
        
        Returns:
            List of hand data dictionaries
        """
        hands_data = []
        
        if not hand_results.multi_hand_landmarks:
            return hands_data
        
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, 
                                            hand_results.multi_handedness):
            # Fix handedness for chest-mounted camera view
            # MediaPipe determines handedness from camera perspective, but we want person's perspective
            original_handedness = handedness.classification[0].label
            corrected_handedness = 'Left' if original_handedness == 'Right' else 'Right'
            
            hand_data = {
                'landmarks': hand_landmarks,
                'handedness': corrected_handedness,  # Use corrected handedness
                'confidence': handedness.classification[0].score,
                'keypoints': {},
                'bounding_box': None
            }
            
            # Extract keypoints
            keypoints = {}
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = 0, 0
            
            for landmark_name, landmark_idx in config.HAND_KEYPOINTS.items():
                landmark = hand_landmarks.landmark[landmark_idx]
                
                # Normalize coordinates
                x_norm, y_norm = normalize_coordinates(
                    landmark.x * self.frame_width,
                    landmark.y * self.frame_height,
                    self.frame_width,
                    self.frame_height
                )
                
                keypoints[landmark_name] = (x_norm, y_norm)
                
                # Track bounding box
                min_x = min(min_x, x_norm)
                min_y = min(min_y, y_norm)
                max_x = max(max_x, x_norm)
                max_y = max(max_y, y_norm)
            
            hand_data['keypoints'] = keypoints
            hand_data['bounding_box'] = (min_x, min_y, max_x, max_y)
            
            # Apply smoothing if enabled
            if config.ENABLE_SMOOTHING:
                hand_id = f"hand_{len(hands_data)}"
                if hand_id not in self.hand_buffers:
                    self.hand_buffers[hand_id] = {}
                
                # Add current keypoints to buffer
                for landmark_name, coords in keypoints.items():
                    if landmark_name not in self.hand_buffers[hand_id]:
                        self.hand_buffers[hand_id][landmark_name] = deque(maxlen=config.BUFFER_SIZE)
                    self.hand_buffers[hand_id][landmark_name].append(coords)
                
                # Apply smoothing
                smoothed_keypoints = {}
                for landmark_name in keypoints.keys():
                    if landmark_name in self.hand_buffers[hand_id]:
                        smoothed_coords = smooth_coordinates(
                            self.hand_buffers[hand_id][landmark_name],
                            config.SMOOTHING_ALPHA
                        )
                        if smoothed_coords:
                            smoothed_keypoints[landmark_name] = smoothed_coords
                
                if smoothed_keypoints:
                    hand_data['keypoints'] = smoothed_keypoints
            
            hands_data.append(hand_data)
        
        return hands_data
    
    def _extract_face_keypoints(self, face_results, frame_shape: Tuple[int, int, int]) -> Optional[Dict]:
        """
        Extract face keypoints from MediaPipe results.
        
        Args:
            face_results: MediaPipe face mesh results
            frame_shape: Frame dimensions (height, width, channels)
        
        Returns:
            Face data dictionary or None if no face detected
        """
        if not face_results.multi_face_landmarks:
            return None
        
        # Use the first detected face
        face_landmarks = face_results.multi_face_landmarks[0]
        
        face_data = {
            'landmarks': face_landmarks,
            'keypoints': {},
            'bounding_box': None
        }
        
        # Extract keypoints
        keypoints = {}
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        
        for landmark_name, landmark_idx in config.FACE_KEYPOINTS.items():
            landmark = face_landmarks.landmark[landmark_idx]
            
            # Normalize coordinates
            x_norm, y_norm = normalize_coordinates(
                landmark.x * self.frame_width,
                landmark.y * self.frame_height,
                self.frame_width,
                self.frame_height
            )
            
            keypoints[landmark_name] = (x_norm, y_norm)
            
            # Track bounding box
            min_x = min(min_x, x_norm)
            min_y = min(min_y, y_norm)
            max_x = max(max_x, x_norm)
            max_y = max(max_y, y_norm)
        
        face_data['keypoints'] = keypoints
        face_data['bounding_box'] = (min_x, min_y, max_x, max_y)
        
        # Apply smoothing if enabled
        if config.ENABLE_SMOOTHING:
            # Add current keypoints to buffer
            for landmark_name, coords in keypoints.items():
                if landmark_name not in self.face_buffer:
                    self.face_buffer[landmark_name] = deque(maxlen=config.BUFFER_SIZE)
                self.face_buffer[landmark_name].append(coords)
            
            # Apply smoothing
            smoothed_keypoints = {}
            for landmark_name in keypoints.keys():
                if landmark_name in self.face_buffer:
                    smoothed_coords = smooth_coordinates(
                        self.face_buffer[landmark_name],
                        config.SMOOTHING_ALPHA
                    )
                    if smoothed_coords:
                        smoothed_keypoints[landmark_name] = smoothed_coords
            
            if smoothed_keypoints:
                face_data['keypoints'] = smoothed_keypoints
        
        return face_data
    
    def get_hand_keypoint(self, hand_data: Dict, keypoint_name: str) -> Optional[Tuple[float, float]]:
        """
        Get a specific keypoint from hand data.
        
        Args:
            hand_data: Hand data dictionary
            keypoint_name: Name of the keypoint to retrieve
        
        Returns:
            Keypoint coordinates or None if not found
        """
        return hand_data.get('keypoints', {}).get(keypoint_name)
    
    def get_face_keypoint(self, face_data: Dict, keypoint_name: str) -> Optional[Tuple[float, float]]:
        """
        Get a specific keypoint from face data.
        
        Args:
            face_data: Face data dictionary
            keypoint_name: Name of the keypoint to retrieve
        
        Returns:
            Keypoint coordinates or None if not found
        """
        return face_data.get('keypoints', {}).get(keypoint_name)
    
    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()
        self.face_mesh.close() 