"""
Simple focus implementation - fixed box in lower center with grey background.
Much simpler than the complex ROI system.
"""

import cv2
import numpy as np
import config


class SimpleFocus:
    """
    Simple focus implementation with fixed box in lower center.
    """
    
    def __init__(self):
        """Initialize simple focus processor."""
        self.enabled = config.SIMPLE_FOCUS_ENABLED
        self.box_ratio = config.SIMPLE_FOCUS_BOX_RATIO  # Fraction of frame size
        self.grey_value = config.SIMPLE_FOCUS_GREY_VALUE  # Grey color (0-255)
        
        # Frame dimensions (set when processing)
        self.frame_width = 0
        self.frame_height = 0
        
    def set_frame_dimensions(self, width: int, height: int):
        """Set frame dimensions."""
        self.frame_width = width
        self.frame_height = height
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame with simple focus - fixed box in lower center.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with focus box
        """
        if not self.enabled:
            return frame
            
        if self.frame_width == 0 or self.frame_height == 0:
            self.set_frame_dimensions(frame.shape[1], frame.shape[0])
        
        # Create a copy of the frame
        processed_frame = frame.copy()
        
        # Calculate box dimensions
        box_width = int(self.frame_width * self.box_ratio)
        box_height = int(self.frame_height * self.box_ratio * 1.2)  # Increase height by 20%
        
        # Calculate box position (lower center, touching bottom)
        box_x = (self.frame_width - box_width) // 2
        box_y = self.frame_height - box_height  # Touch the bottom of the frame
        
        # Create grey background
        processed_frame[:] = self.grey_value
        
        # Copy the focus box area from original frame
        processed_frame[box_y:box_y+box_height, box_x:box_x+box_width] = \
            frame[box_y:box_y+box_height, box_x:box_x+box_width]
        
        return processed_frame
    
    def draw_focus_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw focus box overlay on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with focus box overlay
        """
        if not self.enabled:
            return frame
            
        # Calculate box dimensions
        box_width = int(self.frame_width * self.box_ratio)
        box_height = int(self.frame_height * self.box_ratio * 1.2)  # Increase height by 20%
        
        # Calculate box position (lower center, touching bottom)
        box_x = (self.frame_width - box_width) // 2
        box_y = self.frame_height - box_height  # Touch the bottom of the frame
        
        # Draw focus box border
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     (0, 255, 0), 3)  # Green border
        
        # Draw focus label (removed confusing percentage display)
        label = "FOCUS"
        cv2.putText(frame, label, (box_x, box_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def toggle_focus(self):
        """Toggle focus on/off."""
        self.enabled = not self.enabled
        
    def get_focus_info(self) -> dict:
        """Get focus information."""
        return {
            'enabled': self.enabled,
            'box_ratio': self.box_ratio,
            'grey_value': self.grey_value,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height
        }
