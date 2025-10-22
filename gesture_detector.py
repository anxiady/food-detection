"""
Gesture detection module for analyzing hand-to-mouth proximity and detecting feeding gestures.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import config
from utils import calculate_distance, get_dominant_hand


class GestureDetector:
    """
    Analyzes hand-to-mouth proximity and detects feeding gestures.
    """
    
    def __init__(self):
        """Initialize gesture detection with buffers and state tracking."""
        # Buffers for tracking motion over time
        self.distance_buffer = deque(maxlen=config.BUFFER_SIZE)
        self.proximity_buffer = deque(maxlen=config.BUFFER_SIZE)
        self.time_buffer = deque(maxlen=config.BUFFER_SIZE)
        
        # State tracking
        self.feeding_start_time = None
        self.is_feeding = False
        self.last_feeding_time = 0
        self.feeding_count = 0
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def detect_gestures(self, detection_data: Dict) -> Dict:
        """
        Analyze detection data and identify gestures.
        
        Args:
            detection_data: Dictionary containing hands and face keypoints
        
        Returns:
            Dictionary with gesture analysis results
        """
        hands_data = detection_data.get('hands', [])
        face_data = detection_data.get('face')
        
        # Initialize result structure
        result = {
            'feeding_detected': False,
            'feeding_duration': 0.0,
            'proximity_distance': None,
            'dominant_hand': None,
            'mouth_position': None,
            'confidence': 0.0,
            'gesture_state': 'none',
            'frame_count': self.frame_count
        }
        
        # Update frame count
        self.frame_count += 1
        
        # Check if we have valid data
        if not hands_data or not face_data:
            self._update_buffers(None, None, None)
            return result
        
        # Filter hands based on chest-mounted camera entry zones
        # Hands should enter from bottom or sides, not from top
        filtered_hands = self._filter_hands_by_entry_zone(hands_data)
        if not filtered_hands:
            self._update_buffers(None, None, None)
            return result
        
        # Get dominant hand from filtered hands
        dominant_hand = get_dominant_hand(filtered_hands)
        if not dominant_hand:
            self._update_buffers(None, None, None)
            return result
        
        # Get mouth position
        mouth_position = self._get_mouth_position(face_data)
        if not mouth_position:
            self._update_buffers(None, None, None)
            return result
        
        # Get hand position (prefer index finger tip, fallback to wrist)
        hand_position = self._get_hand_position(dominant_hand)
        if not hand_position:
            self._update_buffers(None, None, None)
            return result
        
        # Calculate distance
        distance = calculate_distance(hand_position, mouth_position)
        
        # Check proximity
        is_proximate = distance <= config.PROXIMITY_THRESHOLD
        
        # Update buffers
        current_time = time.time()
        self._update_buffers(distance, is_proximate, current_time)
        
        # Analyze feeding gesture
        feeding_result = self._analyze_feeding_gesture(current_time)
        
        # Update result
        result.update({
            'feeding_detected': feeding_result['detected'],
            'feeding_duration': feeding_result['duration'],
            'proximity_distance': distance,
            'dominant_hand': dominant_hand,
            'mouth_position': mouth_position,
            'hand_position': hand_position,
            'confidence': dominant_hand.get('confidence', 0.0),
            'gesture_state': feeding_result['state']
        })
        
        return result
    
    def _get_mouth_position(self, face_data: Dict) -> Optional[Tuple[float, float]]:
        """
        Get the mouth position from face data for chest-mounted upward-facing camera.
        Prioritizes chin/lower jaw region since camera sees underside of face.
        
        Args:
            face_data: Face keypoints data
        
        Returns:
            Chin/mouth position coordinates or None
        """
        keypoints = face_data.get('keypoints', {})
        
        # For chest-mounted camera, prioritize chin/lower jaw region
        # This is the most visible and relevant area from below
        chin = keypoints.get('chin')
        if chin:
            return chin
        
        # Fallback to mouth center (still visible from below)
        mouth_center = keypoints.get('mouth_center')
        if mouth_center:
            return mouth_center
        
        # Secondary fallback to nose tip (visible from below)
        nose_tip = keypoints.get('nose_tip')
        if nose_tip:
            return nose_tip
        
        return None
    
    def _get_hand_position(self, hand_data: Dict) -> Optional[Tuple[float, float]]:
        """
        Get the best hand position for gesture analysis in chest-mounted camera setup.
        Prioritizes wrist and palm center since they're most stable from below.
        
        Args:
            hand_data: Hand keypoints data
        
        Returns:
            Hand position coordinates or None
        """
        keypoints = hand_data.get('keypoints', {})
        
        # For chest-mounted camera, prioritize wrist and palm center
        # These are more stable and visible from below than fingertips
        wrist = keypoints.get('wrist')
        if wrist:
            return wrist
        
        # Fallback to palm center (index finger pip as approximation)
        palm_center = keypoints.get('palm_center')
        if palm_center:
            return palm_center
        
        # Secondary fallback to index finger tip
        index_tip = keypoints.get('index_tip')
        if index_tip:
            return index_tip
        
        # Final fallback to middle finger tip
        middle_tip = keypoints.get('middle_tip')
        if middle_tip:
            return middle_tip
        
        return None
    
    def _filter_hands_by_entry_zone(self, hands_data: List[Dict]) -> List[Dict]:
        """
        Filter hands based on chest-mounted camera entry zones.
        Hands should enter from bottom or sides, not from top.
        
        Args:
            hands_data: List of detected hands
        
        Returns:
            Filtered list of hands that entered from valid zones
        """
        filtered_hands = []
        
        for hand_data in hands_data:
            keypoints = hand_data.get('keypoints', {})
            wrist = keypoints.get('wrist')
            
            if wrist:
                # In chest-mounted camera, Y coordinates are inverted
                # Y=0 is top of frame, Y=1 is bottom of frame
                # Hands should be in lower portion of frame (Y > 0.3)
                # or from sides (X < 0.2 or X > 0.8)
                x, y = wrist
                
                # Check if hand is in valid entry zone
                in_lower_zone = y > config.HAND_ENTRY_ZONE_LOWER  # Lower portion of frame
                in_side_zone = x < config.HAND_ENTRY_ZONE_SIDES or x > (1 - config.HAND_ENTRY_ZONE_SIDES)  # Left or right sides
                
                if in_lower_zone or in_side_zone:
                    filtered_hands.append(hand_data)
        
        return filtered_hands
    
    def _update_buffers(self, distance: Optional[float], is_proximate: Optional[bool], 
                       timestamp: Optional[float]):
        """
        Update tracking buffers with new data.
        
        Args:
            distance: Distance between hand and mouth
            is_proximate: Whether hand is proximate to mouth
            timestamp: Current timestamp
        """
        if distance is not None:
            self.distance_buffer.append(distance)
        
        if is_proximate is not None:
            self.proximity_buffer.append(is_proximate)
        
        if timestamp is not None:
            self.time_buffer.append(timestamp)
    
    def _analyze_feeding_gesture(self, current_time: float) -> Dict:
        """
        Analyze the feeding gesture based on proximity over time.
        
        Args:
            current_time: Current timestamp
        
        Returns:
            Dictionary with feeding analysis results
        """
        result = {
            'detected': False,
            'duration': 0.0,
            'state': 'none'
        }
        
        # Need minimum buffer size for analysis
        if len(self.proximity_buffer) < 5:
            return result
        
        # Calculate how long hand has been proximate to mouth
        proximate_frames = sum(self.proximity_buffer)
        total_frames = len(self.proximity_buffer)
        
        # Calculate duration based on frame rate
        if total_frames > 0:
            proximate_ratio = proximate_frames / total_frames
            
            # Estimate duration (assuming 30 FPS)
            estimated_duration = proximate_ratio * (total_frames / 30.0)
            
            # Check if duration exceeds threshold
            if estimated_duration >= config.FEEDING_THRESHOLD_DURATION:
                # Check if this is a new feeding gesture
                time_since_last = current_time - self.last_feeding_time
                
                if not self.is_feeding or time_since_last > 2.0:  # 2 second cooldown
                    self.is_feeding = True
                    self.feeding_start_time = current_time
                    self.feeding_count += 1
                    result['detected'] = True
                    result['state'] = 'feeding_start'
                else:
                    result['detected'] = True
                    result['state'] = 'feeding_ongoing'
                
                result['duration'] = estimated_duration
                self.last_feeding_time = current_time
            else:
                # Reset feeding state if not proximate
                if self.is_feeding:
                    self.is_feeding = False
                    result['state'] = 'feeding_end'
        
        return result
    
    def get_statistics(self) -> Dict:
        """
        Get gesture detection statistics.
        
        Returns:
            Dictionary with statistics
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        return {
            'total_frames': self.frame_count,
            'fps': self.frame_count / elapsed_time if elapsed_time > 0 else 0,
            'feeding_count': self.feeding_count,
            'is_feeding': self.is_feeding,
            'buffer_size': len(self.distance_buffer),
            'average_distance': np.mean(self.distance_buffer) if self.distance_buffer else 0,
            'proximity_ratio': sum(self.proximity_buffer) / len(self.proximity_buffer) if self.proximity_buffer else 0
        }
    
    def reset(self):
        """Reset all buffers and state."""
        self.distance_buffer.clear()
        self.proximity_buffer.clear()
        self.time_buffer.clear()
        self.feeding_start_time = None
        self.is_feeding = False
        self.last_feeding_time = 0
        self.feeding_count = 0
        self.frame_count = 0
        self.start_time = time.time() 