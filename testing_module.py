"""
Testing module for interactive gesture controls and accuracy visualization.
Provides volume control via fingertip motion and real-time gesture accuracy plotting.
"""

import cv2
import numpy as np
import time
import threading
import math
from typing import Dict, Optional, Tuple, List
from collections import deque
import config
from utils import calculate_distance, denormalize_coordinates


class VolumeController:
    """
    Controls system volume based on finger distance.
    """
    
    def __init__(self):
        """Initialize volume controller."""
        self.current_volume = 50  # Start at 50%
        self.min_volume = 0
        self.max_volume = 100
        self.volume_history = deque(maxlen=30)  # Track volume changes
        
        # Volume control method (platform-specific)
        self.volume_control_method = self._detect_volume_control_method()
        
    def _detect_volume_control_method(self) -> str:
        """Detect the appropriate volume control method for the platform."""
        import platform
        system = platform.system()
        
        if system == "Darwin":  # macOS
            return "osascript"
        elif system == "Linux":
            return "amixer"
        elif system == "Windows":
            return "nircmd"
        else:
            return "none"
    
    def set_volume(self, volume: int) -> bool:
        """
        Set system volume using platform-specific method.
        
        Args:
            volume: Volume level (0-100)
        
        Returns:
            True if successful, False otherwise
        """
        volume = max(self.min_volume, min(self.max_volume, volume))
        
        try:
            if self.volume_control_method == "osascript":
                # macOS
                import subprocess
                script = f'set volume output volume {volume}'
                subprocess.run(['osascript', '-e', script], check=True)
                
            elif self.volume_control_method == "amixer":
                # Linux
                import subprocess
                subprocess.run(['amixer', 'set', 'Master', f'{volume}%'], check=True)
                
            elif self.volume_control_method == "nircmd":
                # Windows (requires nircmd)
                import subprocess
                subprocess.run(['nircmd', 'setsysvolume', str(volume * 655)], check=True)
            
            self.current_volume = volume
            self.volume_history.append((time.time(), volume))
            return True
            
        except Exception as e:
            print(f"Volume control failed: {e}")
            return False
    
    def get_volume(self) -> int:
        """Get current system volume."""
        return self.current_volume
    
    def process_finger_distance(self, index_tip: Optional[Tuple[float, float]], 
                               thumb_tip: Optional[Tuple[float, float]]) -> Dict:
        """
        Process finger distance to control volume.
        
        Args:
            index_tip: Index fingertip position (normalized)
            thumb_tip: Thumb tip position (normalized)
        
        Returns:
            Dictionary with volume control results
        """
        result = {
            'volume_changed': False,
            'new_volume': self.current_volume,
            'distance_detected': False,
            'finger_distance': None,
            'volume_percentage': 0.0
        }
        
        if index_tip is None or thumb_tip is None:
            return result
        
        # Calculate distance between index and thumb
        distance = calculate_distance(index_tip, thumb_tip)
        result['finger_distance'] = distance
        result['distance_detected'] = True
        
        # Use adaptive thresholds based on hand size and position
        # When hand is close to camera, use smaller thresholds
        # When hand is far from camera, use larger thresholds
        
        # Calculate hand size indicator (distance from wrist to middle finger tip)
        # This helps determine if hand is close or far from camera
        hand_size_indicator = self._estimate_hand_size()
        
        # Adaptive thresholds based on hand size
        if hand_size_indicator > 0.3:  # Hand is close to camera (large in frame)
            min_distance = 0.01  # Very small distance for close hand
            max_distance = 0.08  # Smaller max distance for close hand
        elif hand_size_indicator > 0.15:  # Hand is at medium distance
            min_distance = 0.015  # Medium min distance
            max_distance = 0.12   # Medium max distance
        else:  # Hand is far from camera (small in frame)
            min_distance = 0.02   # Larger min distance for far hand
            max_distance = 0.15   # Larger max distance for far hand
        
        # Calculate volume percentage with smooth mapping
        if distance <= min_distance:
            volume_percentage = 0.0  # Muted when fingers are very close
        elif distance >= max_distance:
            volume_percentage = 100.0  # Max volume when comfortably spread
        else:
            # Smooth interpolation between min and max
            normalized_distance = (distance - min_distance) / (max_distance - min_distance)
            volume_percentage = normalized_distance * 100.0
        
        # Fix floating-point precision issues
        if volume_percentage < 0.1:
            volume_percentage = 0.0
        elif volume_percentage > 99.9:
            volume_percentage = 100.0
        
        result['volume_percentage'] = volume_percentage
        
        # Special case: Always mute when fingers are very close
        if distance <= min_distance and self.current_volume > 0:
            if self.set_volume(0):
                result['volume_changed'] = True
                result['new_volume'] = 0
        # Otherwise, only update volume if there's a significant change
        else:
            volume_diff = abs(volume_percentage - self.current_volume)
            if volume_diff > 2.0:  # 2% threshold for more stable control
                new_volume = int(round(volume_percentage))
                if self.set_volume(new_volume):
                    result['volume_changed'] = True
                    result['new_volume'] = new_volume
        
        return result
    
    def _estimate_hand_size(self) -> float:
        """
        Estimate hand size to determine distance from camera.
        This is a simplified approach - in a real implementation,
        you'd want to track hand keypoints over time.
        
        Returns:
            Hand size indicator (0-1, larger = closer to camera)
        """
        # For now, return a default value
        # In a full implementation, you'd calculate this from hand keypoints
        return 0.2  # Default medium distance
    
    def process_finger_distance_scale_invariant(self, index_tip: Optional[Tuple[float, float]], 
                                              thumb_tip: Optional[Tuple[float, float]],
                                              wrist: Optional[Tuple[float, float]],
                                              middle_tip: Optional[Tuple[float, float]]) -> Dict:
        """
        Process finger distance for volume control using scale-invariant detection.
        
        Args:
            index_tip: Index finger tip position (normalized)
            thumb_tip: Thumb tip position (normalized)
            wrist: Wrist position (normalized)
            middle_tip: Middle finger tip position (normalized)
        
        Returns:
            Dictionary with volume control results
        """
        result = {
            'volume_changed': False,
            'new_volume': self.current_volume,
            'distance_detected': False,
            'finger_distance': None,
            'normalized_ratio': None,
            'volume_percentage': 0.0
        }
        
        if not all([index_tip, thumb_tip, wrist, middle_tip]):
            return result
        
        # Calculate raw distance between index and thumb
        raw_distance = calculate_distance(index_tip, thumb_tip)
        result['finger_distance'] = raw_distance
        
        # Calculate reference distance (wrist to middle fingertip)
        reference_distance = calculate_distance(wrist, middle_tip)
        
        if reference_distance == 0:
            return result
        
        # Normalize grip distance by reference distance
        normalized_ratio = raw_distance / reference_distance
        result['normalized_ratio'] = normalized_ratio
        result['distance_detected'] = True
        
        # Map normalized ratio to volume (0-100%)
        # Smaller ratio = tighter grip = lower volume
        min_ratio = 0.1  # Very tight grip (almost touching)
        max_ratio = 0.6  # Comfortably spread
        
        # Calculate volume percentage with smooth mapping
        if normalized_ratio <= min_ratio:
            volume_percentage = 0.0  # Muted when fingers are very close
        elif normalized_ratio >= max_ratio:
            volume_percentage = 100.0  # Max volume when comfortably spread
        else:
            # Smooth interpolation between min and max
            normalized_volume = (normalized_ratio - min_ratio) / (max_ratio - min_ratio)
            volume_percentage = normalized_volume * 100.0
        
        # Fix floating-point precision issues
        if volume_percentage < 0.1:
            volume_percentage = 0.0
        elif volume_percentage > 99.9:
            volume_percentage = 100.0
        
        result['volume_percentage'] = volume_percentage
        
        # Special case: Always mute when fingers are very close
        if normalized_ratio <= min_ratio and self.current_volume > 0:
            if self.set_volume(0):
                result['volume_changed'] = True
                result['new_volume'] = 0
        # Otherwise, only update volume if there's a significant change
        else:
            volume_diff = abs(volume_percentage - self.current_volume)
            if volume_diff > 2.0:  # 2% threshold for more stable control
                new_volume = int(round(volume_percentage))
                if self.set_volume(new_volume):
                    result['volume_changed'] = True
                    result['new_volume'] = new_volume
        
        return result


class GestureAccuracyVisualizer:
    """
    Visualizes gesture detection accuracy with real-time plotting.
    """
    
    def __init__(self, width: int = 300, height: int = 200):
        """
        Initialize gesture accuracy visualizer.
        
        Args:
            width: Plot width in pixels
            height: Plot height in pixels
        """
        self.width = width
        self.height = height
        self.distance_history = deque(maxlen=100)  # Store last 100 distance measurements
        self.time_history = deque(maxlen=100)
        self.start_time = time.time()
        
        # Plot settings
        self.plot_margin = 20
        self.plot_width = width - 2 * self.plot_margin
        self.plot_height = height - 2 * self.plot_margin
        
        # Colors
        self.bg_color = (50, 50, 50)
        self.grid_color = (100, 100, 100)
        self.line_color = (0, 255, 0)
        self.threshold_color = (0, 0, 255)
        self.text_color = (255, 255, 255)
        
    def update_distance(self, distance: float):
        """
        Update distance measurement for plotting.
        
        Args:
            distance: Current hand-to-mouth distance
        """
        current_time = time.time() - self.start_time
        self.distance_history.append(distance)
        self.time_history.append(current_time)
    
    def create_plot(self) -> np.ndarray:
        """
        Create real-time distance plot.
        
        Returns:
            Plot image as numpy array
        """
        # Create background
        plot_img = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        
        if len(self.distance_history) < 2:
            return plot_img
        
        # Calculate plot bounds
        min_dist = min(self.distance_history)
        max_dist = max(self.distance_history)
        dist_range = max_dist - min_dist if max_dist > min_dist else 1.0
        
        # Draw grid
        self._draw_grid(plot_img, min_dist, max_dist, dist_range)
        
        # Draw threshold line
        threshold_y = self._map_value_to_y(config.PROXIMITY_THRESHOLD, min_dist, max_dist)
        cv2.line(plot_img, 
                (self.plot_margin, threshold_y),
                (self.width - self.plot_margin, threshold_y),
                self.threshold_color, 2)
        
        # Draw distance line
        points = []
        for i, (t, d) in enumerate(zip(self.time_history, self.distance_history)):
            x = self.plot_margin + int((t / max(self.time_history)) * self.plot_width)
            y = self._map_value_to_y(d, min_dist, max_dist)
            points.append((x, y))
        
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(plot_img, points[i-1], points[i], self.line_color, 2)
        
        # Draw labels
        self._draw_labels(plot_img, min_dist, max_dist)
        
        return plot_img
    
    def _map_value_to_y(self, value: float, min_val: float, max_val: float) -> int:
        """Map a value to Y coordinate in the plot."""
        normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        return self.height - self.plot_margin - int(normalized * self.plot_height)
    
    def _draw_grid(self, img: np.ndarray, min_val: float, max_val: float, val_range: float):
        """Draw grid lines on the plot."""
        # Horizontal grid lines
        for i in range(5):
            y = self.plot_margin + int(i * self.plot_height / 4)
            cv2.line(img, (self.plot_margin, y), (self.width - self.plot_margin, y),
                    self.grid_color, 1)
        
        # Vertical grid lines
        for i in range(5):
            x = self.plot_margin + int(i * self.plot_width / 4)
            cv2.line(img, (x, self.plot_margin), (x, self.height - self.plot_margin),
                    self.grid_color, 1)
    
    def _draw_labels(self, img: np.ndarray, min_val: float, max_val: float):
        """Draw axis labels and values."""
        # Title
        cv2.putText(img, "Hand-to-Mouth Distance", (10, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)
        
        # Y-axis labels
        for i in range(5):
            y = self.plot_margin + int(i * self.plot_height / 4)
            value = max_val - (i * (max_val - min_val) / 4)
            label = f"{value:.2f}"
            cv2.putText(img, label, (5, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.text_color, 1)
        
        # Current value
        if self.distance_history:
            current_dist = self.distance_history[-1]
            cv2.putText(img, f"Current: {current_dist:.3f}", (10, self.height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)


class TestingModule:
    """
    Main testing module that integrates volume control and accuracy visualization.
    """
    
    def __init__(self):
        """Initialize testing module."""
        self.volume_controller = VolumeController()
        self.accuracy_visualizer = GestureAccuracyVisualizer()
        
        # Mode tracking
        self.current_mode = 'normal'  # 'normal', 'volume', 'gesture_test', 'hand_objects'
        self.mode_names = {
            'normal': 'Normal Mode',
            'volume': 'Volume Control Mode',
            'gesture_test': 'Gesture Accuracy Test',
            'hand_objects': 'Hand Object Detection Mode'
        }
        
        # Gesture test state
        self.feeding_overlay_active = False
        self.overlay_start_time = None
        self.overlay_duration = 2.0  # seconds
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
    def process_frame(self, frame: np.ndarray, detection_data: Dict, 
                     gesture_data: Dict) -> np.ndarray:
        """
        Process frame based on current testing mode.
        
        Args:
            frame: Input frame
            detection_data: Keypoint detection results
            gesture_data: Gesture analysis results
        
        Returns:
            Processed frame with testing overlays
        """
        current_time = time.time()
        self.frame_times.append(current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
        # Create a copy of the frame
        processed_frame = frame.copy()
        
        # Process based on current mode
        if self.current_mode == 'volume':
            processed_frame = self._process_volume_mode(processed_frame, detection_data)
        elif self.current_mode == 'gesture_test':
            processed_frame = self._process_gesture_test_mode(processed_frame, detection_data, gesture_data)
        elif self.current_mode == 'hand_objects':
            processed_frame = self._process_hand_objects_mode(processed_frame, detection_data)
        
        # Add mode indicator
        processed_frame = self._add_mode_indicator(processed_frame)
        
        # Add performance metrics
        processed_frame = self._add_performance_metrics(processed_frame)
        
        return processed_frame
    
    def _process_volume_mode(self, frame: np.ndarray, detection_data: Dict) -> np.ndarray:
        """Process frame for volume control mode."""
        # Get dominant hand
        hands_data = detection_data.get('hands', [])
        if not hands_data:
            return frame
        
        # Use the first detected hand
        hand_data = hands_data[0]
        keypoints = hand_data.get('keypoints', {})
        
        # Get required keypoints for scale-invariant detection
        index_tip = keypoints.get('index_tip')
        thumb_tip = keypoints.get('thumb_tip')
        wrist = keypoints.get('wrist')
        middle_tip = keypoints.get('middle_tip')
        
        if all([index_tip, thumb_tip, wrist, middle_tip]):
            # Use scale-invariant volume control
            volume_result = self.volume_controller.process_finger_distance_scale_invariant(
                index_tip, thumb_tip, wrist, middle_tip)
            
            # Draw volume control indicators
            frame = self._draw_volume_indicators(frame, index_tip, thumb_tip, volume_result)
        elif index_tip and thumb_tip:
            # Fallback to original method if missing keypoints
            volume_result = self.volume_controller.process_finger_distance(index_tip, thumb_tip)
            frame = self._draw_volume_indicators(frame, index_tip, thumb_tip, volume_result)
        
        return frame
    
    def _process_gesture_test_mode(self, frame: np.ndarray, detection_data: Dict, 
                                  gesture_data: Dict) -> np.ndarray:
        """Process frame for gesture accuracy test mode."""
        # Update distance for plotting
        if gesture_data.get('proximity_distance') is not None:
            self.accuracy_visualizer.update_distance(gesture_data['proximity_distance'])
        
        # Check for feeding gesture
        if gesture_data.get('feeding_detected'):
            if not self.feeding_overlay_active:
                self.feeding_overlay_active = True
                self.overlay_start_time = time.time()
        
        # Draw feeding overlay
        if self.feeding_overlay_active:
            frame = self._draw_feeding_overlay(frame)
            
            # Check if overlay should be removed
            if self.overlay_start_time and (time.time() - self.overlay_start_time) > self.overlay_duration:
                self.feeding_overlay_active = False
                self.overlay_start_time = None
        
        # Create and overlay accuracy plot
        plot_img = self.accuracy_visualizer.create_plot()
        frame = self._overlay_plot(frame, plot_img)
        
        return frame
    
    def _process_hand_objects_mode(self, frame: np.ndarray, detection_data: Dict) -> np.ndarray:
        """Process frame for hand object detection mode."""
        hands_data = detection_data.get('hands', [])
        
        # Draw detailed hand object analysis
        for i, hand_data in enumerate(hands_data):
            handedness = hand_data.get('handedness', 'Unknown')
            object_detection = hand_data.get('object_detection', {})
            
            # Get hand bounding box for positioning
            bbox = hand_data.get('bounding_box')
            if bbox:
                min_x, min_y, max_x, max_y = bbox
                x1, y1 = denormalize_coordinates(min_x, min_y, frame.shape[1], frame.shape[0])
                x2, y2 = denormalize_coordinates(max_x, max_y, frame.shape[1], frame.shape[0])
                
                # Draw hand bounding box
                is_holding = object_detection.get('is_holding_object', False)
                box_color = (0, 255, 0) if is_holding else (0, 0, 255)  # Green if holding, red if empty
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                
                # Draw detailed analysis
                frame = self._draw_hand_analysis(frame, hand_data, (x1, y1), i)
        
        # Draw summary statistics
        frame = self._draw_hand_objects_summary(frame, hands_data)
        
        return frame
    
    def _draw_hand_analysis(self, frame: np.ndarray, hand_data: Dict, 
                           position: Tuple[int, int], hand_index: int) -> np.ndarray:
        """Draw detailed hand analysis for object detection mode."""
        handedness = hand_data.get('handedness', 'Unknown')
        object_detection = hand_data.get('object_detection', {})
        
        # Calculate text position
        x, y = position
        text_y_start = y - 10 - (hand_index * 120)  # Stack multiple hands vertically
        
        # Draw hand label
        is_holding = object_detection.get('is_holding_object', False)
        status_text = f"{handedness} Hand: {'HOLDING OBJECT' if is_holding else 'EMPTY'}"
        status_color = (0, 255, 0) if is_holding else (0, 0, 255)
        
        cv2.putText(frame, status_text, (x, text_y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Draw grip details
        grip_type = object_detection.get('grip_type', 'unknown')
        confidence = object_detection.get('grip_confidence', 0.0)
        
        grip_text = f"Grip: {grip_type.replace('_', ' ').title()}"
        cv2.putText(frame, grip_text, (x, text_y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, conf_text, (x, text_y_start + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw finger curl analysis
        finger_curls = object_detection.get('finger_curls', {})
        curl_y = text_y_start + 70
        
        for finger, curl in finger_curls.items():
            curl_text = f"{finger.title()}: {curl:.2f}"
            curl_color = (0, 255, 255) if curl > 0.5 else (255, 255, 255)  # Yellow for curled fingers
            cv2.putText(frame, curl_text, (x, curl_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, curl_color, 1)
            curl_y += 15
        
        return frame
    
    def _draw_hand_objects_summary(self, frame: np.ndarray, hands_data: List[Dict]) -> np.ndarray:
        """Draw summary statistics for hand object detection."""
        total_hands = len(hands_data)
        hands_holding = sum(1 for hand in hands_data 
                           if hand.get('object_detection', {}).get('is_holding_object', False))
        
        # Count grip types
        grip_counts = {}
        for hand in hands_data:
            grip_type = hand.get('object_detection', {}).get('grip_type', 'unknown')
            grip_counts[grip_type] = grip_counts.get(grip_type, 0) + 1
        
        # Draw summary box
        summary_x = 10
        summary_y = frame.shape[0] - 150
        
        # Background rectangle
        cv2.rectangle(frame, (summary_x, summary_y), (summary_x + 300, summary_y + 140),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (summary_x, summary_y), (summary_x + 300, summary_y + 140),
                     (255, 255, 255), 2)
        
        # Summary text
        title_text = "Hand Object Detection Summary"
        cv2.putText(frame, title_text, (summary_x + 10, summary_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        hands_text = f"Total Hands: {total_hands}"
        cv2.putText(frame, hands_text, (summary_x + 10, summary_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        objects_text = f"Hands Holding Objects: {hands_holding}"
        objects_color = (0, 255, 0) if hands_holding > 0 else (255, 255, 255)
        cv2.putText(frame, objects_text, (summary_x + 10, summary_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, objects_color, 1)
        
        # Grip type breakdown
        grip_y = summary_y + 80
        for grip_type, count in grip_counts.items():
            if count > 0:
                grip_text = f"{grip_type.replace('_', ' ').title()}: {count}"
                cv2.putText(frame, grip_text, (summary_x + 10, grip_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                grip_y += 15
        
        # Instructions
        instructions = [
            "Try holding different objects:",
            "- Cups, utensils (loose grip)",
            "- Small items like chips (pinch)",
            "- Empty hands (open)"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = summary_y - 20 - (i * 15)
            cv2.putText(frame, instruction, (summary_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def _draw_volume_indicators(self, frame: np.ndarray, index_tip: Tuple[float, float], 
                               thumb_tip: Tuple[float, float], volume_result: Dict) -> np.ndarray:
        """Draw volume control indicators on frame."""
        # Convert to pixel coordinates
        index_x, index_y = denormalize_coordinates(index_tip[0], index_tip[1],
                                                 frame.shape[1], frame.shape[0])
        thumb_x, thumb_y = denormalize_coordinates(thumb_tip[0], thumb_tip[1],
                                                 frame.shape[1], frame.shape[0])
        
        # Draw fingertip indicators
        color = (0, 255, 0) if volume_result['distance_detected'] else (255, 255, 0)
        cv2.circle(frame, (index_x, index_y), 12, color, 3)  # Index finger
        cv2.circle(frame, (thumb_x, thumb_y), 12, (255, 0, 0), 3)  # Thumb
        
        # Draw line between fingers
        if volume_result['distance_detected']:
            line_color = (0, 255, 0) if volume_result['volume_changed'] else (255, 255, 0)
            cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), line_color, 3)
            
            # Draw distance text
            distance = volume_result.get('finger_distance', 0)
            normalized_ratio = volume_result.get('normalized_ratio', None)
            mid_x = (index_x + thumb_x) // 2
            mid_y = (index_y + thumb_y) // 2
            
            if normalized_ratio is not None:
                # Show normalized ratio for scale-invariant detection
                distance_text = f"N:{normalized_ratio:.2f}"
            else:
                # Show raw distance for fallback method
                distance_text = f"D:{distance:.2f}"
                
            cv2.putText(frame, distance_text, (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw volume bar
        frame = self._draw_volume_bar(frame)
        
        return frame
    
    def _draw_volume_bar(self, frame: np.ndarray) -> np.ndarray:
        """Draw volume level bar."""
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = frame.shape[0] - 40
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # Volume level
        volume_width = int((self.volume_controller.current_volume / 100) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + volume_width, bar_y + bar_height),
                     (0, 255, 0), -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (255, 255, 255), 2)
        
        # Volume text
        volume_text = f"Volume: {self.volume_controller.current_volume}%"
        cv2.putText(frame, volume_text, (bar_x, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add comfort zone indicator
        comfort_text = "Comfort Zone: Pinch to mute, spread comfortably for max"
        cv2.putText(frame, comfort_text, (bar_x, bar_y - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def _draw_feeding_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw feeding gesture overlay."""
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Change background color
        overlay_color = (0, 100, 200)  # Blue tint
        overlay = cv2.addWeighted(overlay, 0.3, np.full_like(overlay, overlay_color), 0.7, 0)
        
        # Add feeding emoji or text
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        
        # Draw feeding indicator
        cv2.circle(overlay, (center_x, center_y), 100, (0, 255, 0), 3)
        cv2.putText(overlay, "FEEDING DETECTED!", (center_x - 100, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Blend with original frame
        alpha = 0.7
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return frame
    
    def _overlay_plot(self, frame: np.ndarray, plot_img: np.ndarray) -> np.ndarray:
        """Overlay accuracy plot on frame."""
        # Position plot in top-right corner
        plot_x = frame.shape[1] - plot_img.shape[1] - 10
        plot_y = 10
        
        # Create ROI
        roi = frame[plot_y:plot_y + plot_img.shape[0], plot_x:plot_x + plot_img.shape[1]]
        
        # Blend plot with frame
        alpha = 0.8
        blended = cv2.addWeighted(roi, 1 - alpha, plot_img, alpha, 0)
        
        # Place back in frame
        frame[plot_y:plot_y + plot_img.shape[0], plot_x:plot_x + plot_img.shape[1]] = blended
        
        return frame
    
    def _add_mode_indicator(self, frame: np.ndarray) -> np.ndarray:
        """Add current mode indicator to frame."""
        mode_text = f"Mode: {self.mode_names[self.current_mode]}"
        cv2.putText(frame, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add mode instructions
        instructions = {
            'normal': 'Press V for Volume Control, G for Gesture Test, H for Hand Objects',
            'volume': 'Spread fingers to control volume (pinch to mute)',
            'gesture_test': 'Watch distance plot and feeding overlay',
            'hand_objects': 'Show hands to camera - try holding different objects'
        }
        
        instruction_text = instructions.get(self.current_mode, '')
        cv2.putText(frame, instruction_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def _add_performance_metrics(self, frame: np.ndarray) -> np.ndarray:
        """Add performance metrics to frame."""
        if self.frame_times:
            avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def handle_keyboard_input(self, key: int) -> bool:
        """
        Handle keyboard input for mode switching.
        
        Args:
            key: Key code from cv2.waitKey()
        
        Returns:
            True to continue, False to quit
        """
        if key == ord('v'):
            self.current_mode = 'volume'
            print("Switched to Volume Control Mode")
        elif key == ord('g'):
            self.current_mode = 'gesture_test'
            print("Switched to Gesture Accuracy Test Mode")
        elif key == ord('h'):
            self.current_mode = 'hand_objects'
            print("Switched to Hand Object Detection Mode")
        elif key == ord('n'):
            self.current_mode = 'normal'
            print("Switched to Normal Mode")
        elif key == ord('q'):
            return False
        
        return True
    
    def get_mode_info(self) -> Dict:
        """Get information about current mode and system state."""
        return {
            'current_mode': self.current_mode,
            'mode_name': self.mode_names[self.current_mode],
            'volume': self.volume_controller.current_volume,
            'avg_fps': 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0,
            'distance_samples': len(self.accuracy_visualizer.distance_history)
        } 