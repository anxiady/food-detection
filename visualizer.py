"""
Video visualization and annotation module.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import config
from utils import denormalize_coordinates, draw_text_with_background


class Visualizer:
    """
    Handles drawing annotations, keypoints, and gesture indicators on video frames.
    """
    
    def __init__(self):
        """Initialize visualizer with drawing settings."""
        self.frame_width = 0
        self.frame_height = 0
        
    def set_frame_dimensions(self, width: int, height: int):
        """Set frame dimensions for coordinate conversion."""
        self.frame_width = width
        self.frame_height = height
    
    def draw_frame(self, frame: np.ndarray, detection_data: Dict, 
                  gesture_data: Dict, statistics: Dict, food_data: Dict = None, food_only_mode: bool = False) -> np.ndarray:
        """
        Draw all visualizations on the frame.
        
        Args:
            frame: Input frame
            detection_data: Keypoint detection results
            gesture_data: Gesture analysis results
            statistics: Performance statistics
        
        Returns:
            Annotated frame
        """
        # Set frame dimensions if not set
        if self.frame_width == 0 or self.frame_height == 0:
            self.set_frame_dimensions(frame.shape[1], frame.shape[0])
        
        # Create a copy of the frame
        annotated_frame = frame.copy()
        
        # Draw keypoints and bounding boxes
        if config.SHOW_KEYPOINTS:
            annotated_frame = self._draw_keypoints(annotated_frame, detection_data)
        
        if config.SHOW_BOUNDING_BOXES:
            annotated_frame = self._draw_bounding_boxes(annotated_frame, detection_data)
        
        # Draw object detection indicators
        annotated_frame = self._draw_object_detection(annotated_frame, detection_data)
        
        # Draw food detection results
        if food_data:
            annotated_frame = self._draw_food_detections(annotated_frame, food_data)
        
        # Draw gesture-specific visualizations
        annotated_frame = self._draw_gesture_indicators(annotated_frame, gesture_data)
        
        # Draw proximity line
        if gesture_data.get('hand_position') and gesture_data.get('mouth_position'):
            annotated_frame = self._draw_proximity_line(
                annotated_frame, 
                gesture_data['hand_position'], 
                gesture_data['mouth_position'],
                gesture_data.get('proximity_distance')
            )
        
        # Draw statistics and information
        annotated_frame = self._draw_statistics(annotated_frame, statistics, gesture_data, detection_data, food_data, food_only_mode)
        
        return annotated_frame
    
    def _draw_keypoints(self, frame: np.ndarray, detection_data: Dict) -> np.ndarray:
        """
        Draw hand and face keypoints on the frame.
        
        Args:
            frame: Input frame
            detection_data: Detection results
        
        Returns:
            Frame with keypoints drawn
        """
        # Draw hand keypoints
        hands_data = detection_data.get('hands', [])
        for hand_data in hands_data:
            keypoints = hand_data.get('keypoints', {})
            handedness = hand_data.get('handedness', 'Unknown')
            object_detection = hand_data.get('object_detection', {})
            
            # Determine color based on object detection
            is_holding = object_detection.get('is_holding_object', False)
            keypoint_color = (0, 255, 0) if is_holding else config.COLORS['hand']  # Green if holding, default if not
            
            for keypoint_name, coords in keypoints.items():
                if coords:
                    # Convert normalized coordinates to pixel coordinates
                    x, y = denormalize_coordinates(coords[0], coords[1], 
                                                 self.frame_width, self.frame_height)
                    
                    # Draw keypoint
                    cv2.circle(frame, (x, y), config.CIRCLE_RADIUS, 
                              keypoint_color, -1)
                    
                    # Draw keypoint label
                    label = f"{keypoint_name}"
                    cv2.putText(frame, label, (x + 10, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                               config.COLORS['text'], 1)
            
            # Draw handedness label
            if keypoints.get('wrist'):
                wrist_x, wrist_y = denormalize_coordinates(
                    keypoints['wrist'][0], keypoints['wrist'][1],
                    self.frame_width, self.frame_height
                )
                cv2.putText(frame, handedness, (wrist_x - 20, wrist_y - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLORS['hand'], 2)
        
        # Draw face keypoints
        face_data = detection_data.get('face')
        if face_data:
            keypoints = face_data.get('keypoints', {})
            
            for keypoint_name, coords in keypoints.items():
                if coords:
                    # Convert normalized coordinates to pixel coordinates
                    x, y = denormalize_coordinates(coords[0], coords[1],
                                                 self.frame_width, self.frame_height)
                    
                    # Draw keypoint
                    cv2.circle(frame, (x, y), config.CIRCLE_RADIUS,
                              config.COLORS['face'], -1)
                    
                    # Draw keypoint label
                    label = f"{keypoint_name}"
                    cv2.putText(frame, label, (x + 10, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                               config.COLORS['text'], 1)
        
        return frame
    
    def _draw_object_detection(self, frame: np.ndarray, detection_data: Dict) -> np.ndarray:
        """
        Draw object detection indicators for hands.
        
        Args:
            frame: Input frame
            detection_data: Detection results
        
        Returns:
            Frame with object detection indicators
        """
        hands_data = detection_data.get('hands', [])
        
        for hand_data in hands_data:
            object_detection = hand_data.get('object_detection', {})
            handedness = hand_data.get('handedness', 'Unknown')
            
            if object_detection.get('is_holding_object'):
                # Get hand bounding box for positioning
                bbox = hand_data.get('bounding_box')
                if bbox:
                    min_x, min_y, max_x, max_y = bbox
                    
                    # Convert to pixel coordinates
                    x1, y1 = denormalize_coordinates(min_x, min_y, self.frame_width, self.frame_height)
                    x2, y2 = denormalize_coordinates(max_x, max_y, self.frame_width, self.frame_height)
                    
                    # Draw object detection indicator
                    indicator_color = (0, 255, 255)  # Yellow for object detection
                    cv2.rectangle(frame, (x1, y1), (x2, y2), indicator_color, 3)
                    
                    # Draw object detection text
                    grip_type = object_detection.get('grip_type', 'unknown')
                    confidence = object_detection.get('grip_confidence', 0.0)
                    
                    text = f"{handedness}: {grip_type} ({confidence:.2f})"
                    text_position = (x1, y1 - 10)
                    
                    draw_text_with_background(
                        frame, text, text_position,
                        font_scale=0.5, thickness=1,
                        text_color=(255, 255, 255),
                        bg_color=indicator_color
                    )
                    
                    # Draw object icon
                    icon_text = "📦" if grip_type in ['tight_grip', 'loose_grip'] else "✋"
                    icon_position = (x1 + 5, y1 + 20)
                    
                    # Note: OpenCV doesn't support emoji directly, so we'll use text
                    cv2.putText(frame, "OBJECT", icon_position,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, indicator_color, 2)
        
        return frame
    
    def _draw_bounding_boxes(self, frame: np.ndarray, detection_data: Dict) -> np.ndarray:
        """
        Draw bounding boxes around detected hands and face.
        
        Args:
            frame: Input frame
            detection_data: Detection results
        
        Returns:
            Frame with bounding boxes drawn
        """
        # Draw hand bounding boxes
        hands_data = detection_data.get('hands', [])
        for hand_data in hands_data:
            bbox = hand_data.get('bounding_box')
            handedness = hand_data.get('handedness', 'Unknown')
            object_detection = hand_data.get('object_detection', {})
            
            if bbox:
                min_x, min_y, max_x, max_y = bbox
                
                # Convert to pixel coordinates
                x1, y1 = denormalize_coordinates(min_x, min_y,
                                               self.frame_width, self.frame_height)
                x2, y2 = denormalize_coordinates(max_x, max_y,
                                               self.frame_width, self.frame_height)
                
                # Determine color based on object detection
                is_holding = object_detection.get('is_holding_object', False)
                box_color = (0, 255, 0) if is_holding else config.COLORS['hand']  # Green if holding, default if not
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                             box_color, config.LINE_THICKNESS)
                
                # Draw hand label with object status
                grip_type = object_detection.get('grip_type', 'unknown')
                label = f"{handedness}: {grip_type.replace('_', ' ').title()}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Draw face bounding box
        face_data = detection_data.get('face')
        if face_data:
            bbox = face_data.get('bounding_box')
            if bbox:
                min_x, min_y, max_x, max_y = bbox
                
                # Convert to pixel coordinates
                x1, y1 = denormalize_coordinates(min_x, min_y,
                                               self.frame_width, self.frame_height)
                x2, y2 = denormalize_coordinates(max_x, max_y,
                                               self.frame_width, self.frame_height)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                             config.COLORS['face'], config.LINE_THICKNESS)
        
        return frame
    
    def _draw_food_detections(self, frame: np.ndarray, food_data: Dict) -> np.ndarray:
        """
        Draw food detection results on the frame.
        
        Args:
            frame: Input frame
            food_data: Food detection results
        
        Returns:
            Frame with food detection visualizations
        """
        detections = food_data.get('detections', [])
        food_detected = food_data.get('food_detected', False)
        stable_detection = food_data.get('stable_detection', False)
        method = food_data.get('method', 'unknown')
        
        # Draw main food detection indicator
        if food_detected:
            self._draw_food_indicator(frame, stable_detection, method)
        
        # Draw individual food detections
        for detection in detections:
            self._draw_single_food_detection(frame, detection)
        
        return frame
    
    def _draw_food_indicator(self, frame: np.ndarray, stable: bool, method: str):
        """
        Draw main food detection indicator.
        
        Args:
            frame: Input frame
            stable: Whether detection is stable
            method: Detection method used
        """
        # Position in top-center
        center_x = self.frame_width // 2
        center_y = 80
        
        # Choose colors based on stability
        if stable:
            bg_color = (0, 255, 0)  # Green for stable detection
            text_color = (0, 0, 0)  # Black text
            status_text = "FOOD DETECTED (STABLE)"
        else:
            bg_color = (0, 165, 255)  # Orange for unstable detection
            text_color = (255, 255, 255)  # White text
            status_text = "FOOD DETECTED"
        
        # Draw main indicator
        draw_text_with_background(
            frame, status_text, (center_x - 100, center_y),
            font_scale=0.8, thickness=2,
            text_color=text_color, bg_color=bg_color
        )
        
        # Draw method indicator
        method_text = f"Method: {method.upper()}"
        draw_text_with_background(
            frame, method_text, (center_x - 60, center_y + 30),
            font_scale=0.5, thickness=1,
            text_color=text_color, bg_color=bg_color
        )
    
    def _draw_single_food_detection(self, frame: np.ndarray, detection: Dict):
        """
        Draw a single food detection with bounding box and label.
        
        Args:
            frame: Input frame
            detection: Single detection result
        """
        bbox = detection.get('bbox')
        class_name = detection.get('class_name', 'food')
        confidence = detection.get('confidence', 0.0)
        
        if bbox:
            x1, y1, x2, y2 = bbox
            
            # Choose color based on confidence
            if confidence > 0.8:
                box_color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.5:
                box_color = (0, 255, 255)  # Yellow for medium confidence
            else:
                box_color = (0, 165, 255)  # Orange for low confidence
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate text size for background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1), box_color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            # For detections without bounding box (like HuggingFace classification)
            center_x = detection.get('center', (self.frame_width // 2, self.frame_height // 2))[0]
            center_y = detection.get('center', (self.frame_width // 2, self.frame_height // 2))[1]
            
            # Draw classification result
            label = f"{class_name}: {confidence:.2f}"
            draw_text_with_background(
                frame, label, (center_x - 80, center_y),
                font_scale=0.7, thickness=2,
                text_color=(255, 255, 255), bg_color=(0, 255, 0)
            )
    
    def _draw_gesture_indicators(self, frame: np.ndarray, gesture_data: Dict) -> np.ndarray:
        """
        Draw gesture-specific visual indicators.
        
        Args:
            frame: Input frame
            gesture_data: Gesture analysis results
        
        Returns:
            Frame with gesture indicators drawn
        """
        if gesture_data.get('feeding_detected'):
            # Draw feeding detection indicator
            self._draw_feeding_indicator(frame, gesture_data)
        
        return frame
    
    def _draw_feeding_indicator(self, frame: np.ndarray, gesture_data: Dict):
        """
        Draw feeding gesture detection indicator.
        
        Args:
            frame: Input frame
            gesture_data: Gesture analysis results
        """
        # Draw a prominent indicator when feeding is detected
        center_x = self.frame_width // 2
        center_y = 50
        
        # Draw background circle
        cv2.circle(frame, (center_x, center_y), 30,
                  config.COLORS['feeding_detected'], -1)
        
        # Draw text
        text = "FEEDING"
        draw_text_with_background(frame, text, (center_x - 40, center_y + 10),
                                 font_scale=0.8, thickness=2,
                                 text_color=config.COLORS['text'],
                                 bg_color=config.COLORS['feeding_detected'])
        
        # Draw duration if available
        duration = gesture_data.get('feeding_duration', 0)
        if duration > 0:
            duration_text = f"{duration:.1f}s"
            draw_text_with_background(frame, duration_text, (center_x - 20, center_y + 40),
                                     font_scale=0.6, thickness=1,
                                     text_color=config.COLORS['text'],
                                     bg_color=config.COLORS['feeding_detected'])
    
    def _draw_proximity_line(self, frame: np.ndarray, hand_pos: Tuple[float, float],
                           mouth_pos: Tuple[float, float], distance: Optional[float]) -> np.ndarray:
        """
        Draw line between hand and mouth with distance information.
        
        Args:
            frame: Input frame
            hand_pos: Hand position (normalized)
            mouth_pos: Mouth position (normalized)
            distance: Distance between hand and mouth
        
        Returns:
            Frame with proximity line drawn
        """
        # Convert to pixel coordinates
        hand_x, hand_y = denormalize_coordinates(hand_pos[0], hand_pos[1],
                                               self.frame_width, self.frame_height)
        mouth_x, mouth_y = denormalize_coordinates(mouth_pos[0], mouth_pos[1],
                                                 self.frame_width, self.frame_height)
        
        # Determine line color based on proximity
        if distance and distance <= config.PROXIMITY_THRESHOLD:
            line_color = config.COLORS['feeding_detected']  # Red for proximate
            thickness = 3
        else:
            line_color = config.COLORS['proximity_line']  # Yellow for distant
            thickness = 2
        
        # Draw line
        cv2.line(frame, (hand_x, hand_y), (mouth_x, mouth_y),
                line_color, thickness)
        
        # Draw distance text if enabled
        if config.SHOW_DISTANCE and distance is not None:
            mid_x = (hand_x + mouth_x) // 2
            mid_y = (hand_y + mouth_y) // 2
            
            distance_text = f"{distance:.2f}"
            draw_text_with_background(frame, distance_text, (mid_x, mid_y),
                                     font_scale=0.4, thickness=1,
                                     text_color=config.COLORS['text'],
                                     bg_color=line_color)
        
        # Add chest-mounted camera indicator
        if config.CHEST_MOUNTED_MODE:
            cv2.putText(frame, "CHEST-MOUNTED", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return frame
    
    def _draw_statistics(self, frame: np.ndarray, statistics: Dict, 
                        gesture_data: Dict, detection_data: Dict = None, food_data: Dict = None, food_only_mode: bool = False) -> np.ndarray:
        """
        Draw performance statistics and gesture information.
        
        Args:
            frame: Input frame
            statistics: Performance statistics
            gesture_data: Gesture analysis results
        
        Returns:
            Frame with statistics drawn
        """
        # Draw FPS
        if config.SHOW_FPS:
            fps = statistics.get('fps', 0)
            fps_text = f"FPS: {fps:.1f}"
            draw_text_with_background(frame, fps_text, (10, 30),
                                     font_scale=config.TEXT_SCALE,
                                     thickness=config.TEXT_THICKNESS,
                                     text_color=config.COLORS['text'],
                                     bg_color=config.COLORS['background'])
        
        # Draw object detection count (only if not in food-only mode)
        if not food_only_mode and detection_data and detection_data.get('hands'):
            hands_data = detection_data.get('hands', [])
            hands_holding_objects = sum(1 for hand in hands_data 
                                      if hand.get('object_detection', {}).get('is_holding_object', False))
            objects_text = f"Objects: {hands_holding_objects}/{len(hands_data)}"
            draw_text_with_background(frame, objects_text, (10, 60),
                                     font_scale=config.TEXT_SCALE,
                                     thickness=config.TEXT_THICKNESS,
                                     text_color=config.COLORS['text'],
                                     bg_color=config.COLORS['background'])
        
        # Draw feeding count (only if not in food-only mode)
        if not food_only_mode:
            feeding_count = statistics.get('feeding_count', 0)
            count_text = f"Feedings: {feeding_count}"
            y_position = 90 if not food_only_mode else 60  # Adjust position based on mode
            draw_text_with_background(frame, count_text, (10, y_position),
                                     font_scale=config.TEXT_SCALE,
                                     thickness=config.TEXT_THICKNESS,
                                     text_color=config.COLORS['text'],
                                     bg_color=config.COLORS['background'])
        
        # Draw gesture state (only if not in food-only mode)
        if not food_only_mode:
            gesture_state = gesture_data.get('gesture_state', 'none')
            state_text = f"State: {gesture_state}"
            draw_text_with_background(frame, state_text, (10, 120),
                                     font_scale=config.TEXT_SCALE,
                                     thickness=config.TEXT_THICKNESS,
                                     text_color=config.COLORS['text'],
                                     bg_color=config.COLORS['background'])
        
        # Draw confidence (only if not in food-only mode)
        if not food_only_mode:
            confidence = gesture_data.get('confidence', 0)
            if confidence > 0:
                conf_text = f"Conf: {confidence:.2f}"
                draw_text_with_background(frame, conf_text, (10, 150),
                                         font_scale=config.TEXT_SCALE,
                                         thickness=config.TEXT_THICKNESS,
                                         text_color=config.COLORS['text'],
                                         bg_color=config.COLORS['background'])
        
        # Draw proximity status (only if not in food-only mode)
        if not food_only_mode and gesture_data.get('proximity_distance') is not None:
            distance = gesture_data['proximity_distance']
            if distance <= config.PROXIMITY_THRESHOLD:
                proximity_text = "PROXIMATE"
                color = config.COLORS['feeding_detected']
            else:
                proximity_text = "DISTANT"
                color = config.COLORS['proximity_line']
            
            draw_text_with_background(frame, proximity_text, (10, 180),
                                     font_scale=config.TEXT_SCALE,
                                     thickness=config.TEXT_THICKNESS,
                                     text_color=config.COLORS['text'],
                                     bg_color=color)
        
        # Draw food detection statistics (only if not in food-only mode)
        if food_data and not food_only_mode:
            food_detected = food_data.get('food_detected', False)
            stable_detection = food_data.get('stable_detection', False)
            detection_count = food_data.get('detection_count', 0)
            method = food_data.get('method', 'unknown')
            
            # Food detection status
            if food_detected:
                if stable_detection:
                    food_text = f"FOOD: STABLE ({detection_count})"
                    food_color = (0, 255, 0)  # Green
                else:
                    food_text = f"FOOD: DETECTED ({detection_count})"
                    food_color = (0, 165, 255)  # Orange
            else:
                food_text = "FOOD: NONE"
                food_color = (0, 0, 255)  # Red
            
            draw_text_with_background(frame, food_text, (10, 200),
                                     font_scale=config.TEXT_SCALE,
                                     thickness=config.TEXT_THICKNESS,
                                     text_color=config.COLORS['text'],
                                     bg_color=food_color)
            
            # Detection method
            method_text = f"Method: {method.upper()}"
            draw_text_with_background(frame, method_text, (10, 230),
                                     font_scale=config.TEXT_SCALE - 0.1,
                                     thickness=1,
                                     text_color=config.COLORS['text'],
                                     bg_color=config.COLORS['background'])
        
        # Draw ROI information
        if hasattr(config, 'ROI_ENABLED') and config.ROI_ENABLED:
            roi_text = f"ROI: {config.ROI_REGION[2]*100:.0f}% x {config.ROI_REGION[3]*100:.0f}%"
            if hasattr(config, 'OBFUSCATION_ENABLED') and config.OBFUSCATION_ENABLED:
                roi_text += f" | {config.OBFUSCATION_TYPE.upper()}"
            
            draw_text_with_background(frame, roi_text, (10, 260),
                                     font_scale=config.TEXT_SCALE - 0.1,
                                     thickness=1,
                                     text_color=config.COLORS['text'],
                                     bg_color=(0, 255, 255))  # Cyan for ROI info
        
        return frame
    
    def draw_object_detections(self, frame: np.ndarray, object_data: Dict) -> np.ndarray:
        """
        Draw all object detections on the frame.
        
        Args:
            frame: Input frame
            object_data: Object detection results
            
        Returns:
            Annotated frame with object detections
        """
        if not object_data.get('detections'):
            return frame
        
        annotated_frame = frame.copy()
        
        # Draw each detection
        for detection in object_data['detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Choose color based on object type
            if 'food' in class_name.lower() or class_name in ['apple', 'banana', 'orange', 'pizza', 'cake', 'donut']:
                color = (0, 255, 0)  # Green for food
            elif class_name == 'person':
                color = (255, 0, 0)  # Blue for person
            else:
                color = (0, 255, 255)  # Yellow for other objects
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw depth-filtered objects if available (only when depth estimation is enabled)
        if object_data.get('near_objects'):
            for detection in object_data['near_objects']:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                proximity = detection.get('proximity', 'near')
                score = detection.get('proximity_score', 0.0)
                
                # Green box for near objects (depth estimation mode)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw proximity label with white background box
                label = f"{proximity}: {score:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - text_height - 15), (x1 + text_width + 5, y1 - 5), (255, 255, 255), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black text on white background
        
        if object_data.get('far_objects'):
            for detection in object_data['far_objects']:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                proximity = detection.get('proximity', 'far')
                score = detection.get('proximity_score', 0.0)
                
                # Green box for far objects (depth estimation mode)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw proximity label with white background box
                label = f"{proximity}: {score:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - text_height - 15), (x1 + text_width + 5, y1 - 5), (255, 255, 255), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black text on white background
        
        return annotated_frame
    
    def draw_debug_info(self, frame: np.ndarray, debug_data: Dict) -> np.ndarray:
        """
        Draw additional debug information when debug mode is enabled.
        
        Args:
            frame: Input frame
            debug_data: Debug information
        
        Returns:
            Frame with debug information drawn
        """
        if not config.DEBUG_MODE:
            return frame
        
        # Draw debug information in bottom-right corner
        y_offset = self.frame_height - 30
        
        for i, (key, value) in enumerate(debug_data.items()):
            text = f"{key}: {value}"
            x_pos = self.frame_width - 200
            y_pos = y_offset - (i * 25)
            
            draw_text_with_background(frame, text, (x_pos, y_pos),
                                     font_scale=0.4, thickness=1,
                                     text_color=config.COLORS['text'],
                                     bg_color=(0, 0, 0))
        
        return frame 