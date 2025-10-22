"""
Hand object detection module.
Detects if hands are holding objects by analyzing hand geometry and finger positions.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import config
from utils import calculate_distance, calculate_angle


class HandObjectDetector:
    """
    Detects if hands are holding objects by analyzing hand geometry.
    """
    
    def __init__(self):
        """Initialize hand object detector."""
        # Detection thresholds
        self.palm_closed_threshold = 0.15  # Threshold for palm closure
        self.finger_curl_threshold = 0.8   # Threshold for finger curling
        self.grip_confidence_threshold = 0.3  # Lowered threshold for better detection
        
        # Tracking buffers
        self.grip_history = {}  # Track grip state over time
        self.grip_buffer_size = 10  # Frames to track for stability
        
        # Object detection parameters
        self.min_grip_duration = 0.2  # Reduced for faster detection
        self.grip_stability_threshold = 0.7  # Lowered for small items
        
        # Small item detection parameters
        self.small_item_threshold = 0.3  # Threshold for detecting small items like chips
        self.pinch_sensitivity = 0.4  # Sensitivity for pinch detection
        
    def detect_hand_objects(self, hands_data: List[Dict]) -> List[Dict]:
        """
        Detect if hands are holding objects.
        
        Args:
            hands_data: List of hand detection data from KeypointTracker
        
        Returns:
            List of hand data with object detection results
        """
        if not hands_data:
            return []
        
        processed_hands = []
        
        for hand_data in hands_data:
            # Analyze hand geometry for object detection
            grip_analysis = self._analyze_hand_grip(hand_data)
            
            # Add detection results to hand data
            hand_data['object_detection'] = {
                'is_holding_object': grip_analysis['is_holding_object'],
                'grip_confidence': grip_analysis['confidence'],
                'grip_type': grip_analysis['grip_type'],
                'palm_openness': grip_analysis['palm_openness'],
                'finger_curls': grip_analysis['finger_curls'],
                'thumb_opposition': grip_analysis['thumb_opposition'],
                'hand_geometry': grip_analysis['geometry']
            }
            
            processed_hands.append(hand_data)
        
        return processed_hands
    
    def _analyze_hand_grip(self, hand_data: Dict) -> Dict:
        """
        Analyze hand geometry to detect grip patterns using improved algorithm.
        
        Args:
            hand_data: Hand detection data with keypoints
        
        Returns:
            Dictionary with grip analysis results
        """
        keypoints = hand_data.get('keypoints', {})
        if not keypoints:
            return self._empty_grip_analysis()
        
        # Calculate finger curl based on fingertip-to-second knuckle distance
        finger_curls = self._calculate_finger_curls_improved(keypoints)
        
        # Calculate palm openness based on finger spread
        palm_openness = self._calculate_palm_openness(keypoints)
        
        # Calculate thumb opposition (distance between thumb tip and index fingertip)
        thumb_opposition = self._calculate_thumb_opposition(keypoints)
        
        # Analyze grip geometry
        grip_geometry = self._analyze_grip_geometry(keypoints)
        
        # Determine if hand is holding object based on scale-invariant criteria
        is_holding = self._determine_holding_posture(finger_curls, palm_openness, thumb_opposition, keypoints)
        
        # Determine grip type
        grip_type = self._classify_grip_type_improved(finger_curls, palm_openness, thumb_opposition)
        
        # Calculate confidence
        confidence = self._calculate_grip_confidence_improved(finger_curls, palm_openness, thumb_opposition)
        
        return {
            'palm_openness': palm_openness,
            'finger_curls': finger_curls,
            'thumb_opposition': thumb_opposition,
            'geometry': grip_geometry,
            'grip_type': grip_type,
            'confidence': confidence,
            'is_holding_object': is_holding
        }
    
    def _calculate_palm_closure(self, keypoints: Dict) -> float:
        """
        Calculate how closed the palm is (0 = open, 1 = closed).
        
        Args:
            keypoints: Hand keypoints dictionary
        
        Returns:
            Palm closure value (0-1)
        """
        # Use a simpler approach based on finger tip distances to palm center
        palm_center = keypoints.get('palm_center')
        if not palm_center:
            return 0.0
        
        # Get finger tips
        finger_tips = [
            keypoints.get('thumb_tip'),
            keypoints.get('index_tip'),
            keypoints.get('middle_tip'),
            keypoints.get('ring_tip'),
            keypoints.get('pinky_tip')
        ]
        
        # Filter out None values
        finger_tips = [tip for tip in finger_tips if tip is not None]
        
        if len(finger_tips) < 3:
            return 0.0
        
        # Calculate distances from finger tips to palm center
        distances = []
        for tip in finger_tips:
            dist = calculate_distance(tip, palm_center)
            distances.append(dist)
        
        # Calculate average distance
        avg_distance = np.mean(distances)
        
        # Normalize distance to closure value
        # Smaller distances = more closed palm
        # Use a reasonable threshold for normalization
        max_expected_distance = 0.3  # Adjust based on testing
        closure = max(0.0, min(1.0, 1.0 - (avg_distance / max_expected_distance)))
        
        return closure
    
    def _calculate_finger_curls(self, keypoints: Dict) -> Dict:
        """
        Calculate curl values for each finger (0 = straight, 1 = curled).
        
        Args:
            keypoints: Hand keypoints dictionary
        
        Returns:
            Dictionary with curl values for each finger
        """
        finger_curls = {}
        
        # Define finger chains (tip to base)
        finger_chains = {
            'thumb': ['thumb_tip', 'thumb_ip', 'thumb_mcp'],
            'index': ['index_tip', 'index_pip', 'index_mcp'],
            'middle': ['middle_tip', 'middle_pip', 'middle_mcp'],
            'ring': ['ring_tip', 'ring_pip', 'ring_mcp'],
            'pinky': ['pinky_tip', 'pinky_pip', 'pinky_mcp']
        }
        
        for finger_name, chain in finger_chains.items():
            curl_value = self._calculate_finger_curl(keypoints, chain)
            finger_curls[finger_name] = curl_value
        
        return finger_curls
    
    def _calculate_finger_curl(self, keypoints: Dict, finger_chain: List[str]) -> float:
        """
        Calculate curl value for a single finger.
        
        Args:
            keypoints: Hand keypoints dictionary
            finger_chain: List of keypoint names from tip to base
        
        Returns:
            Curl value (0-1)
        """
        # Get keypoints for this finger
        finger_points = []
        for point_name in finger_chain:
            point = keypoints.get(point_name)
            if point:
                finger_points.append(point)
        
        if len(finger_points) < 3:
            return 0.0
        
        # Calculate angles between finger segments
        angles = []
        for i in range(len(finger_points) - 2):
            p1, p2, p3 = finger_points[i], finger_points[i+1], finger_points[i+2]
            angle = calculate_angle(p1, p2, p3)
            angles.append(angle)
        
        if not angles:
            return 0.0
        
        # Calculate curl based on angles
        # Lower angles indicate more curl
        avg_angle = np.mean(angles)
        curl_value = max(0.0, min(1.0, (180 - avg_angle) / 180))
        
        return curl_value
    
    def _calculate_finger_curls_improved(self, keypoints: Dict) -> Dict:
        """
        Calculate finger curl based on distance between fingertip and second knuckle (PIP).
        
        Args:
            keypoints: Hand keypoints dictionary
        
        Returns:
            Dictionary with curl values for each finger (0 = extended, 1 = curled)
        """
        finger_curls = {}
        
        # Define finger chains (tip to PIP joint)
        finger_chains = {
            'thumb': ['thumb_tip', 'thumb_ip'],
            'index': ['index_tip', 'index_pip'],
            'middle': ['middle_tip', 'middle_pip'],
            'ring': ['ring_tip', 'ring_pip'],
            'pinky': ['pinky_tip', 'pinky_pip']
        }
        
        for finger_name, chain in finger_chains.items():
            curl_value = self._calculate_finger_curl_distance(keypoints, chain)
            finger_curls[finger_name] = curl_value
        
        return finger_curls
    
    def _calculate_finger_curl_distance(self, keypoints: Dict, finger_chain: List[str]) -> float:
        """
        Calculate curl based on distance between fingertip and second knuckle.
        
        Args:
            keypoints: Hand keypoints dictionary
            finger_chain: List of keypoint names [tip, pip]
        
        Returns:
            Curl value (0-1) where 0 = extended, 1 = curled
        """
        tip = keypoints.get(finger_chain[0])
        pip = keypoints.get(finger_chain[1])
        
        if not tip or not pip:
            return 0.0
        
        # Calculate distance between tip and PIP
        distance = calculate_distance(tip, pip)
        
        # Normalize distance to curl value
        # Smaller distance = more curled
        # Use reasonable thresholds based on testing
        max_distance = 0.15  # Maximum expected distance for extended finger
        min_distance = 0.02  # Minimum distance for fully curled finger
        
        if distance <= min_distance:
            return 1.0  # Fully curled
        elif distance >= max_distance:
            return 0.0  # Fully extended
        else:
            # Linear interpolation
            return 1.0 - ((distance - min_distance) / (max_distance - min_distance))
    
    def _calculate_palm_openness(self, keypoints: Dict) -> float:
        """
        Calculate palm openness based on spread between fingers.
        
        Args:
            keypoints: Hand keypoints dictionary
        
        Returns:
            Palm openness value (0 = closed, 1 = open)
        """
        # Get finger tips
        finger_tips = [
            keypoints.get('index_tip'),
            keypoints.get('middle_tip'),
            keypoints.get('ring_tip'),
            keypoints.get('pinky_tip')
        ]
        
        # Filter out None values
        finger_tips = [tip for tip in finger_tips if tip is not None]
        
        if len(finger_tips) < 3:
            return 0.5  # Default value
        
        # Calculate average distance between finger tips
        total_distance = 0
        count = 0
        for i in range(len(finger_tips)):
            for j in range(i + 1, len(finger_tips)):
                total_distance += calculate_distance(finger_tips[i], finger_tips[j])
                count += 1
        
        if count == 0:
            return 0.5
        
        avg_distance = total_distance / count
        
        # Normalize to openness value
        # Higher distance = more open palm
        min_expected = 0.05  # Minimum expected spread
        max_expected = 0.25  # Maximum expected spread
        
        if avg_distance <= min_expected:
            return 0.0  # Closed
        elif avg_distance >= max_expected:
            return 1.0  # Open
        else:
            return (avg_distance - min_expected) / (max_expected - min_expected)
    
    def _calculate_thumb_opposition(self, keypoints: Dict) -> float:
        """
        Calculate thumb opposition (distance between thumb tip and index fingertip).
        
        Args:
            keypoints: Hand keypoints dictionary
        
        Returns:
            Thumb opposition value (0 = close, 1 = far)
        """
        thumb_tip = keypoints.get('thumb_tip')
        index_tip = keypoints.get('index_tip')
        
        if not thumb_tip or not index_tip:
            return 0.5  # Default value
        
        distance = calculate_distance(thumb_tip, index_tip)
        
        # Normalize distance
        # Smaller distance = better opposition (more likely holding)
        min_distance = 0.02  # Very close (good opposition)
        max_distance = 0.15  # Far apart (poor opposition)
        
        if distance <= min_distance:
            return 0.0  # Excellent opposition
        elif distance >= max_distance:
            return 1.0  # Poor opposition
        else:
            return (distance - min_distance) / (max_distance - min_distance)
    
    def _determine_holding_posture(self, finger_curls: Dict, palm_openness: float, thumb_opposition: float, keypoints: Dict) -> bool:
        """
        Determine if hand is in a holding posture using scale-invariant detection.
        
        Args:
            finger_curls: Finger curl values
            palm_openness: Palm openness value
            thumb_opposition: Thumb opposition value
            keypoints: Hand keypoints for scale-invariant analysis
        
        Returns:
            True if holding object, False otherwise
        """
        # Method 1: Scale-invariant distance normalization
        normalized_grip = self._calculate_normalized_grip(keypoints)
        
        # Method 2: Finger curl angle detection
        curl_based_grip = self._calculate_curl_based_grip(keypoints)
        
        # Method 3: Original criteria (as backup)
        traditional_grip = self._calculate_traditional_grip(finger_curls, palm_openness, thumb_opposition)
        
        # Combine all methods - if any two agree, consider it holding
        grip_methods = [normalized_grip, curl_based_grip, traditional_grip]
        agreement_count = sum(grip_methods)
        
        # Require at least 2 out of 3 methods to agree
        return agreement_count >= 2
    
    def _calculate_normalized_grip(self, keypoints: Dict) -> bool:
        """
        Calculate grip using scale-invariant distance normalization.
        
        Args:
            keypoints: Hand keypoints
        
        Returns:
            True if normalized grip detected
        """
        # Get required keypoints
        thumb_tip = keypoints.get('thumb_tip')
        index_tip = keypoints.get('index_tip')
        wrist = keypoints.get('wrist')
        middle_tip = keypoints.get('middle_tip')
        
        if not all([thumb_tip, index_tip, wrist, middle_tip]):
            return False
        
        # Calculate thumb-index distance (grip distance)
        grip_distance = calculate_distance(thumb_tip, index_tip)
        
        # Calculate reference distance (wrist to middle fingertip)
        reference_distance = calculate_distance(wrist, middle_tip)
        
        if reference_distance == 0:
            return False
        
        # Normalize grip distance by reference distance
        normalized_grip_ratio = grip_distance / reference_distance
        
        # Threshold for normalized grip (empirically determined)
        # Smaller ratio = tighter grip
        grip_threshold = 0.3  # Adjust based on testing
        
        return normalized_grip_ratio < grip_threshold
    
    def _calculate_curl_based_grip(self, keypoints: Dict) -> bool:
        """
        Calculate grip using finger curl angle detection.
        
        Args:
            keypoints: Hand keypoints
        
        Returns:
            True if curl-based grip detected
        """
        # Check thumb curl
        thumb_curl = self._calculate_finger_curl_angle(keypoints, 'thumb')
        
        # Check index finger curl
        index_curl = self._calculate_finger_curl_angle(keypoints, 'index')
        
        # Check if thumb and index are curled (gripping posture)
        thumb_curled = thumb_curl < 60  # Less than 60 degrees = curled
        index_curled = index_curl < 60  # Less than 60 degrees = curled
        
        # Both thumb and index should be curled for a grip
        return thumb_curled and index_curled
    
    def _calculate_finger_curl_angle(self, keypoints: Dict, finger: str) -> float:
        """
        Calculate the curl angle for a specific finger.
        
        Args:
            keypoints: Hand keypoints
            finger: Finger name ('thumb', 'index', 'middle', 'ring', 'pinky')
        
        Returns:
            Angle in degrees (smaller = more curled)
        """
        if finger == 'thumb':
            # Thumb: MCP -> IP -> Tip
            mcp = keypoints.get('thumb_mcp')
            ip = keypoints.get('thumb_ip')
            tip = keypoints.get('thumb_tip')
        else:
            # Other fingers: MCP -> PIP -> Tip
            mcp = keypoints.get(f'{finger}_mcp')
            pip = keypoints.get(f'{finger}_pip')
            tip = keypoints.get(f'{finger}_tip')
            ip = pip  # Use PIP for other fingers
        
        if not all([mcp, ip, tip]):
            return 180.0  # Return straight angle if missing keypoints
        
        # Calculate angle at the middle joint (IP for thumb, PIP for others)
        angle = calculate_angle(mcp, ip, tip)
        
        return angle
    
    def _calculate_traditional_grip(self, finger_curls: Dict, palm_openness: float, thumb_opposition: float) -> bool:
        """
        Calculate grip using traditional criteria (as backup method).
        
        Args:
            finger_curls: Finger curl values
            palm_openness: Palm openness value
            thumb_opposition: Thumb opposition value
        
        Returns:
            True if traditional grip detected
        """
        # Calculate average finger curl (excluding thumb)
        non_thumb_curls = [finger_curls.get('index', 0.0), 
                          finger_curls.get('middle', 0.0),
                          finger_curls.get('ring', 0.0),
                          finger_curls.get('pinky', 0.0)]
        avg_finger_curl = np.mean(non_thumb_curls)
        
        # Criteria for holding object:
        # 1. Fingers are curled inward (not fully extended)
        fingers_curled = avg_finger_curl > 0.4
        
        # 2. Thumb is close to index or middle finger (suggesting a grip)
        thumb_close = thumb_opposition < 0.3
        
        # 3. Palm is not fully open
        palm_not_open = palm_openness < 0.5
        
        # Must meet at least 2 out of 3 criteria, with at least one being finger curl
        criteria_met = sum([fingers_curled, thumb_close, palm_not_open])
        
        # Require finger curl AND at least one other criterion
        return fingers_curled and criteria_met >= 2
    
    def _classify_grip_type_improved(self, finger_curls: Dict, palm_openness: float, thumb_opposition: float) -> str:
        """
        Classify grip type using improved algorithm.
        
        Args:
            finger_curls: Finger curl values
            palm_openness: Palm openness value
            thumb_opposition: Thumb opposition value
        
        Returns:
            Grip type classification
        """
        # Get individual finger curls
        thumb_curl = finger_curls.get('thumb', 0.0)
        index_curl = finger_curls.get('index', 0.0)
        middle_curl = finger_curls.get('middle', 0.0)
        ring_curl = finger_curls.get('ring', 0.0)
        pinky_curl = finger_curls.get('pinky', 0.0)
        
        # Calculate average curl for non-thumb fingers
        non_thumb_curls = [index_curl, middle_curl, ring_curl, pinky_curl]
        avg_non_thumb_curl = np.mean(non_thumb_curls)
        
        # Small item pinch (chips, small food items)
        # Thumb and index curled, others relatively straight
        if (thumb_curl > 0.4 and index_curl > 0.4 and avg_non_thumb_curl < 0.3):
            return 'small_item_pinch'
        
        # Tight grip (fist-like)
        # All fingers significantly curled
        if avg_non_thumb_curl > 0.6 and thumb_curl > 0.5:
            return 'tight_grip'
        
        # Loose grip (holding larger objects)
        # Moderate curl and not fully open palm
        if avg_non_thumb_curl > 0.3 and palm_openness < 0.6:
            return 'loose_grip'
        
        # Pinch grip (precision grip)
        # Thumb and index curled, others less so
        if (thumb_curl > 0.4 and index_curl > 0.4 and 
            middle_curl < 0.4 and ring_curl < 0.4 and pinky_curl < 0.4):
            return 'pinch_grip'
        
        # Open hand
        # Low curl and high palm openness
        if avg_non_thumb_curl < 0.2 and palm_openness > 0.7:
            return 'open_hand'
        
        return 'unknown'
    
    def _calculate_grip_confidence_improved(self, finger_curls: Dict, palm_openness: float, thumb_opposition: float) -> float:
        """
        Calculate confidence in grip detection using improved algorithm.
        
        Args:
            finger_curls: Finger curl values
            palm_openness: Palm openness value
            thumb_opposition: Thumb opposition value
        
        Returns:
            Confidence value (0-1)
        """
        # Calculate confidence based on multiple factors
        
        # 1. Finger curl consistency (40%)
        curl_values = list(finger_curls.values())
        avg_curl = np.mean(curl_values)
        curl_variance = np.var(curl_values) if len(curl_values) > 1 else 0
        curl_confidence = avg_curl * 0.4 * (1.0 - curl_variance)
        
        # 2. Palm openness clarity (30%)
        # Higher confidence when palm is clearly open or closed
        palm_confidence = 0.0
        if palm_openness < 0.3 or palm_openness > 0.7:
            palm_confidence = 0.3
        else:
            palm_confidence = 0.15
        
        # 3. Thumb opposition clarity (20%)
        # Higher confidence when thumb is clearly close or far
        thumb_confidence = 0.0
        if thumb_opposition < 0.3 or thumb_opposition > 0.7:
            thumb_confidence = 0.2
        else:
            thumb_confidence = 0.1
        
        # 4. Overall consistency (10%)
        consistency_confidence = 0.1
        
        total_confidence = curl_confidence + palm_confidence + thumb_confidence + consistency_confidence
        
        return min(1.0, max(0.0, total_confidence))
    
    def _analyze_grip_geometry(self, keypoints: Dict) -> Dict:
        """
        Analyze overall grip geometry.
        
        Args:
            keypoints: Hand keypoints dictionary
        
        Returns:
            Dictionary with geometry analysis
        """
        geometry = {
            'palm_width': 0.0,
            'palm_height': 0.0,
            'finger_spread': 0.0,
            'thumb_opposition': 0.0
        }
        
        # Calculate palm dimensions
        palm_center = keypoints.get('palm_center')
        if palm_center:
            # Calculate palm width (distance between pinky and index MCPs)
            pinky_mcp = keypoints.get('pinky_mcp')
            index_mcp = keypoints.get('index_finger_mcp')
            
            if pinky_mcp and index_mcp:
                geometry['palm_width'] = calculate_distance(pinky_mcp, index_mcp)
            
            # Calculate palm height (distance from wrist to middle finger MCP)
            wrist = keypoints.get('wrist')
            middle_mcp = keypoints.get('middle_finger_mcp')
            
            if wrist and middle_mcp:
                geometry['palm_height'] = calculate_distance(wrist, middle_mcp)
        
        # Calculate finger spread
        finger_tips = [
            keypoints.get('index_tip'),
            keypoints.get('middle_tip'),
            keypoints.get('ring_tip'),
            keypoints.get('pinky_tip')
        ]
        
        finger_tips = [tip for tip in finger_tips if tip is not None]
        
        if len(finger_tips) >= 2:
            # Calculate average distance between finger tips
            total_distance = 0
            count = 0
            for i in range(len(finger_tips)):
                for j in range(i + 1, len(finger_tips)):
                    total_distance += calculate_distance(finger_tips[i], finger_tips[j])
                    count += 1
            
            if count > 0:
                geometry['finger_spread'] = total_distance / count
        
        # Calculate thumb opposition
        thumb_tip = keypoints.get('thumb_tip')
        palm_center = keypoints.get('palm_center')
        
        if thumb_tip and palm_center:
            geometry['thumb_opposition'] = calculate_distance(thumb_tip, palm_center)
        
        return geometry
    
    def _classify_grip_type(self, palm_closed: float, finger_curls: Dict, geometry: Dict) -> str:
        """
        Classify the type of grip being performed.
        
        Args:
            palm_closed: Palm closure value
            finger_curls: Finger curl values
            geometry: Hand geometry analysis
        
        Returns:
            Grip type classification
        """
        # Calculate average finger curl
        avg_curl = np.mean(list(finger_curls.values()))
        
        # Get individual finger curls
        thumb_curl = finger_curls.get('thumb', 0.0)
        index_curl = finger_curls.get('index', 0.0)
        middle_curl = finger_curls.get('middle', 0.0)
        ring_curl = finger_curls.get('ring', 0.0)
        pinky_curl = finger_curls.get('pinky', 0.0)
        
        # Calculate other fingers average (excluding thumb and index)
        other_fingers_curl = np.mean([middle_curl, ring_curl, pinky_curl])
        
        # Small item pinch detection (chips, small food items)
        # Thumb and index curled, other fingers relatively straight
        if (thumb_curl > 0.3 and index_curl > 0.3 and other_fingers_curl < 0.4):
            return 'small_item_pinch'
        
        # Tight grip detection (fist-like)
        # All fingers significantly curled
        if avg_curl > 0.6 and palm_closed > 0.5:
            return 'tight_grip'
        
        # Loose grip detection (holding larger objects)
        # Moderate curl and palm closure
        if avg_curl > 0.4 and palm_closed > 0.3:
            return 'loose_grip'
        
        # Pinch grip detection (precision grip)
        # Thumb and index curled, others less so
        if (thumb_curl > 0.4 and index_curl > 0.4 and 
            middle_curl < 0.5 and ring_curl < 0.5 and pinky_curl < 0.5):
            return 'pinch_grip'
        
        # Open hand detection
        # Low curl and high finger spread
        if avg_curl < 0.3 and geometry['finger_spread'] > 0.15:
            return 'open_hand'
        
        # Default to unknown if no clear pattern
        return 'unknown'
    
    def _calculate_grip_confidence(self, palm_closed: float, finger_curls: Dict, geometry: Dict) -> float:
        """
        Calculate confidence in grip detection.
        
        Args:
            palm_closed: Palm closure value
            finger_curls: Finger curl values
            geometry: Hand geometry analysis
        
        Returns:
            Confidence value (0-1)
        """
        # Calculate confidence based on multiple factors
        
        # 1. Palm closure confidence (30%)
        palm_confidence = palm_closed * 0.3
        
        # 2. Finger curl consistency (40%)
        curl_values = list(finger_curls.values())
        avg_curl = np.mean(curl_values)
        curl_variance = np.var(curl_values) if len(curl_values) > 1 else 0
        # Higher confidence for consistent curl patterns
        curl_confidence = avg_curl * 0.4 * (1.0 - curl_variance)
        
        # 3. Geometry consistency (20%)
        geometry_confidence = 0.0
        if geometry['palm_width'] > 0 and geometry['palm_height'] > 0:
            aspect_ratio = geometry['palm_width'] / geometry['palm_height']
            if 0.3 < aspect_ratio < 2.5:  # More permissive aspect ratio
                geometry_confidence = 0.2
        
        # 4. Finger spread consistency (10%)
        spread_confidence = 0.0
        if geometry['finger_spread'] > 0:
            # Normalize finger spread (0-1)
            normalized_spread = min(1.0, geometry['finger_spread'] / 0.3)
            spread_confidence = normalized_spread * 0.1
        
        # Combine all confidence factors
        total_confidence = palm_confidence + curl_confidence + geometry_confidence + spread_confidence
        
        return min(1.0, max(0.0, total_confidence))
    
    def _determine_object_holding(self, hand_data: Dict, grip_analysis: Dict) -> bool:
        """
        Determine if hand is holding an object based on grip analysis.
        
        Args:
            hand_data: Hand detection data
            grip_analysis: Grip analysis results
        
        Returns:
            True if holding object, False otherwise
        """
        # Get hand ID for tracking
        hand_id = f"{hand_data.get('handedness', 'unknown')}_{len(hand_data.get('keypoints', {}))}"
        
        # Initialize grip history if needed
        if hand_id not in self.grip_history:
            self.grip_history[hand_id] = deque(maxlen=self.grip_buffer_size)
        
        # Add current grip state to history
        current_grip = {
            'confidence': grip_analysis['confidence'],
            'grip_type': grip_analysis['grip_type'],
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
        }
        
        self.grip_history[hand_id].append(current_grip)
        
        # Determine if holding object based on recent history
        if len(self.grip_history[hand_id]) < 2:  # Reduced from 3
            return False
        
        # Check for consistent grip detection
        recent_grips = list(self.grip_history[hand_id])[-3:]  # Reduced from 5 to 3 frames
        
        # Count high-confidence grips
        high_confidence_count = sum(1 for grip in recent_grips 
                                  if grip['confidence'] > self.grip_confidence_threshold)
        
        # Check for appropriate grip types
        holding_grip_types = ['tight_grip', 'loose_grip', 'pinch_grip', 'small_item_pinch']
        holding_count = sum(1 for grip in recent_grips 
                           if grip['grip_type'] in holding_grip_types)
        
        # Determine if holding object
        confidence_ratio = high_confidence_count / len(recent_grips)
        grip_type_ratio = holding_count / len(recent_grips)
        
        # More permissive thresholds for detection
        is_holding = (confidence_ratio > 0.3 and grip_type_ratio > 0.3)  # Reduced from 0.6
        
        return is_holding
    
    def _empty_grip_analysis(self) -> Dict:
        """Return empty grip analysis when no keypoints are available."""
        return {
            'palm_openness': 0.5,
            'finger_curls': {
                'thumb': 0.0,
                'index': 0.0,
                'middle': 0.0,
                'ring': 0.0,
                'pinky': 0.0
            },
            'thumb_opposition': 0.5,
            'geometry': {
                'palm_width': 0.0,
                'palm_height': 0.0,
                'finger_spread': 0.0,
                'thumb_opposition': 0.0
            },
            'grip_type': 'unknown',
            'confidence': 0.0,
            'is_holding_object': False
        } 