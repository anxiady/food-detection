"""
Proximity-based depth estimation module using synthetic radial gradient mask.
Replaces heavy depth models with fast proximity scoring based on position.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import config


class ProximityDepthEstimator:
    """
    Fast proximity-based depth estimation using synthetic radial gradient mask.
    Much faster than neural network depth models.
    """
    
    def __init__(self):
        """Initialize proximity depth estimator."""
        self.enabled = config.DEPTH_ESTIMATION_ENABLED
        self.depth_mask = None
        self.frame_width = None
        self.frame_height = None
        
        # Proximity thresholds (binary: near/far only) - high threshold for near
        self.near_threshold = 0.75  # Only objects with score >= 0.75 are "near", rest are "far"
        
        # For compatibility with debug display
        self.min_depth = 0.0
        self.max_depth = 1.0
        
        # Performance tracking
        self.frame_count = 0
        self.total_estimations = 0
        self.start_time = None
        
    def set_frame_dimensions(self, width: int, height: int):
        """Set frame dimensions and generate depth mask."""
        self.frame_width = width
        self.frame_height = height
        self.depth_mask = self._generate_depth_mask(width, height)
        
    def _generate_depth_mask(self, width: int, height: int) -> np.ndarray:
        """
        Generate synthetic depth mask with radial gradient.
        
        Args:
            width: Frame width
            height: Frame height
            
        Returns:
            Depth mask with values 0.0 (far) to 1.0 (near)
        """
        # Create coordinate grids
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        
        # Center point at bottom-center of frame
        center_x = width // 2
        center_y = height  # Bottom of frame
        
        # Calculate distance from center
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        
        # Normalize distance (0 to 1)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
        normalized_distance = distance / max_distance
        
        # Create depth mask (1.0 = near, 0.0 = far)
        depth_mask = 1.0 - normalized_distance
        
        # Ensure values are in [0, 1] range
        depth_mask = np.clip(depth_mask, 0.0, 1.0)
        
        return depth_mask
    
    def get_proximity_label(self, bbox: Tuple[int, int, int, int], 
                          depth_mask: np.ndarray, 
                          frame_shape: Tuple[int, int]) -> Tuple[str, float]:
        """
        Get proximity label for a bounding box.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            depth_mask: Synthetic depth mask
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Tuple of (label, score)
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_shape
        
        # Clamp coordinates to frame bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        # NEW LOGIC: Check if object touches top edge
        if y1 <= 5:  # Object touches top edge (clipped)
            return "no-depth", 0.0  # Skip depth estimation
        
        # NEW LOGIC: Use only bottom of object for depth estimation
        # Calculate bottom center point
        bottom_center_x = (x1 + x2) // 2
        bottom_center_y = y2 - 1  # Bottom edge of object
        
        # Get depth score from bottom center point only
        score = depth_mask[bottom_center_y, bottom_center_x]
        
        # Determine proximity label based on score (binary: near/far only)
        if score >= self.near_threshold:
            return "near", score
        else:
            return "far", score
    
    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth using synthetic proximity mask.
        
        Args:
            frame: Input frame
            
        Returns:
            Synthetic depth map
        """
        if not self.enabled:
            return None
        
        # Generate depth mask if not exists
        if self.depth_mask is None:
            h, w = frame.shape[:2]
            self.set_frame_dimensions(w, h)
        
        self.total_estimations += 1
        return self.depth_mask
    
    def filter_objects_by_depth(self, detections: List[Dict], depth_map: np.ndarray) -> Dict:
        """
        Filter objects based on proximity scoring.
        
        Args:
            detections: List of object detections
            depth_map: Synthetic depth mask
            
        Returns:
            Dictionary with filtered objects
        """
        if depth_map is None or not detections:
            return {
                'near_objects': [],
                'far_objects': [],
                'depth_available': False
            }
        
        near_objects = []
        far_objects = []
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Get proximity label and score
            label, score = self.get_proximity_label(bbox, depth_map, depth_map.shape)
            
            # Add proximity information to detection
            detection_with_proximity = detection.copy()
            detection_with_proximity['proximity'] = label
            detection_with_proximity['proximity_score'] = score
            
            # Categorize based on proximity (binary: near/far only)
            if label == 'no-depth':
                # Skip objects that touch top edge (no depth estimation)
                continue
            elif label == 'near':
                near_objects.append(detection_with_proximity)
            else:  # 'far'
                far_objects.append(detection_with_proximity)
        
        return {
            'near_objects': near_objects,
            'far_objects': far_objects,
            'depth_available': True
        }
    
    def draw_depth_overlay(self, frame: np.ndarray, depth_map: np.ndarray, 
                          far_objects: List[Dict]) -> np.ndarray:
        """
        Draw proximity-based depth overlay on frame.
        
        Args:
            frame: Input frame
            depth_map: Synthetic depth mask
            far_objects: List of far objects
            
        Returns:
            Frame with proximity overlay
        """
        if not config.DEPTH_SHOW_OVERLAY or depth_map is None:
            return frame
        
        # Create depth heatmap
        depth_heatmap = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original frame
        overlay_frame = cv2.addWeighted(frame, 0.7, depth_heatmap, 0.3, 0)
        
        # Draw far objects with green color and proximity labels only
        for detection in far_objects:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            proximity = detection.get('proximity', 'far')
            score = detection.get('proximity_score', 0.0)
            
            # Use green color for all depth estimation boxes
            color = (0, 255, 0)  # Green for all proximity levels
            
            # Draw bounding box
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw proximity label with white background box
            label = f"{proximity}: {score:.2f}"
            
            # Get text size for background box
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw white background box
            cv2.rectangle(overlay_frame, (x1, y1 - text_height - 15), (x1 + text_width + 5, y1 - 5), (255, 255, 255), -1)
            
            # Draw text on white background
            cv2.putText(overlay_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black text on white background
        
        return overlay_frame
    
    def get_statistics(self) -> Dict:
        """Get proximity depth estimation statistics."""
        if self.start_time is None:
            self.start_time = self.frame_count
        
        current_time = self.frame_count
        elapsed_time = max(1, current_time - self.start_time)
        
        return {
            'total_frames': self.frame_count,
            'fps': self.frame_count / elapsed_time if elapsed_time > 0 else 0,
            'total_estimations': self.total_estimations,
            'estimation_rate': self.total_estimations / self.frame_count if self.frame_count > 0 else 0,
            'model_loaded': True,
            'model_name': 'proximity_synthetic',
            'device': 'cpu',
            'proximity_range': f"binary: {self.near_threshold:.2f}"
        }
    
    def toggle_depth(self):
        """Toggle proximity depth estimation on/off."""
        self.enabled = not self.enabled
    
    def release(self):
        """Release resources."""
        self.depth_mask = None
        self.frame_width = None
        self.frame_height = None
