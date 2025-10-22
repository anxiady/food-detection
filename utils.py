"""
Utility functions for the gesture recognition system.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict
import time
from collections import deque


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
    
    Returns:
        Euclidean distance between the points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_angle(point1: Tuple[float, float], point2: Tuple[float, float], point3: Tuple[float, float]) -> float:
    """
    Calculate angle between three points (point2 is the vertex).
    
    Args:
        point1: First point (x, y)
        point2: Vertex point (x, y)
        point3: Third point (x, y)
    
    Returns:
        Angle in degrees
    """
    # Calculate vectors
    vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
    
    # Calculate dot product
    dot_product = np.dot(vector1, vector2)
    
    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Calculate cosine of angle
    cos_angle = dot_product / (magnitude1 * magnitude2)
    
    # Clamp to valid range
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def normalize_coordinates(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    """
    Normalize coordinates to 0-1 range.
    
    Args:
        x: X coordinate
        y: Y coordinate
        width: Image width
        height: Image height
    
    Returns:
        Normalized coordinates (x_norm, y_norm)
    """
    return x / width, y / height


def denormalize_coordinates(x_norm: float, y_norm: float, width: int, height: int) -> Tuple[int, int]:
    """
    Convert normalized coordinates back to pixel coordinates.
    
    Args:
        x_norm: Normalized X coordinate (0-1)
        y_norm: Normalized Y coordinate (0-1)
        width: Image width
        height: Image height
    
    Returns:
        Pixel coordinates (x, y)
    """
    return int(x_norm * width), int(y_norm * height)


def smooth_coordinates(coordinates: deque, alpha: float = 0.7) -> Optional[Tuple[float, float]]:
    """
    Apply exponential smoothing to a sequence of coordinates.
    
    Args:
        coordinates: Deque of coordinate tuples
        alpha: Smoothing factor (0-1)
    
    Returns:
        Smoothed coordinates or None if no coordinates available
    """
    if not coordinates:
        return None
    
    if len(coordinates) == 1:
        return coordinates[0]
    
    # Apply exponential smoothing
    smoothed_x = coordinates[0][0]
    smoothed_y = coordinates[0][1]
    
    # Convert deque to list for iteration (skip first element)
    coord_list = list(coordinates)
    for x, y in coord_list[1:]:
        smoothed_x = alpha * x + (1 - alpha) * smoothed_x
        smoothed_y = alpha * y + (1 - alpha) * smoothed_y
    
    return smoothed_x, smoothed_y


def get_dominant_hand(hands_data: List[dict]) -> Optional[dict]:
    """
    Determine the dominant hand based on confidence and position.
    
    Args:
        hands_data: List of detected hands with keypoints and confidence
    
    Returns:
        Dominant hand data or None if no hands detected
    """
    if not hands_data:
        return None
    
    if len(hands_data) == 1:
        return hands_data[0]
    
    # For multiple hands, choose the one with higher confidence
    # or the one closer to the center of the frame
    best_hand = None
    best_score = -1
    
    for hand in hands_data:
        # Calculate score based on confidence and position
        confidence = hand.get('confidence', 0)
        wrist_pos = hand.get('keypoints', {}).get('wrist', (0.5, 0.5))
        
        # Prefer hands closer to center (assuming chest-mounted camera)
        center_distance = calculate_distance(wrist_pos, (0.5, 0.5))
        score = confidence * (1 - center_distance)
        
        if score > best_score:
            best_score = score
            best_hand = hand
    
    return best_hand


def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate current FPS.
    
    Args:
        start_time: Start time of processing
        frame_count: Number of processed frames
    
    Returns:
        Current FPS
    """
    elapsed_time = time.time() - start_time
    return frame_count / elapsed_time if elapsed_time > 0 else 0


def draw_text_with_background(img: np.ndarray, text: str, position: Tuple[int, int], 
                             font_scale: float = 0.6, thickness: int = 2,
                             text_color: Tuple[int, int, int] = (255, 255, 255),
                             bg_color: Tuple[int, int, int] = (0, 0, 0)) -> None:
    """
    Draw text with background for better visibility.
    
    Args:
        img: Image to draw on
        text: Text to draw
        position: Position (x, y) for text
        font_scale: Font scale
        thickness: Text thickness
        text_color: Text color (B, G, R)
        bg_color: Background color (B, G, R)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    x, y = position
    cv2.rectangle(img, (x, y - text_height - baseline), 
                  (x + text_width, y + baseline), bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, position, font, font_scale, text_color, thickness)


def create_timestamp() -> str:
    """
    Create a timestamp string for logging and file naming.
    
    Returns:
        Timestamp string in format YYYY-MM-DD_HH-MM-SS
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def validate_camera(camera_index: int) -> bool:
    """
    Validate if camera is available and working.
    
    Args:
        camera_index: Camera device index
    
    Returns:
        True if camera is available, False otherwise
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False
    
    ret, frame = cap.read()
    cap.release()
    return ret

def find_continuity_camera() -> Optional[int]:
    """
    Find the Continuity Camera device index.
    
    Returns:
        Camera index if found, None otherwise
    """
    # Check common camera indices for Continuity Camera
    for camera_index in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            try:
                # Get camera properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Try to read a frame
                ret, frame = cap.read()
                cap.release()
                
                if ret and frame is not None:
                    # Continuity Camera typically has high resolution but lower FPS
                    # Built-in cameras usually have 30fps, Continuity Camera often has 1fps or lower
                    if width >= 1920 and height >= 1080:
                        # Look for camera with very low FPS (indicating Continuity Camera)
                        if fps < 5:  # Continuity Camera often has 1fps
                            print(f"Found Continuity Camera at index {camera_index}")
                            print(f"Resolution: {width}x{height}, FPS: {fps}")
                            return camera_index
                        # If it's not camera 0 and has high resolution, it might be Continuity Camera
                        elif camera_index > 0:
                            print(f"Found potential external camera at index {camera_index}")
                            print(f"Resolution: {width}x{height}, FPS: {fps}")
                            return camera_index
            except:
                cap.release()
                continue
    
    return None

def list_available_cameras() -> List[Dict]:
    """
    List all available cameras with their properties.
    
    Returns:
        List of camera information dictionaries
    """
    cameras = []
    
    for camera_index in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            try:
                # Get camera properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Try to read a frame
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    camera_info = {
                        'index': camera_index,
                        'width': int(width),
                        'height': int(height),
                        'fps': fps,
                        'working': True
                    }
                    cameras.append(camera_info)
                    print(f"Camera {camera_index}: {width}x{height} @ {fps}fps")
                
            except Exception as e:
                print(f"Error reading camera {camera_index}: {e}")
            finally:
                cap.release()
    
    return cameras


def resize_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        target_width: Target width
        target_height: Target height
    
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    # Calculate scaling factor
    scale = min(target_width / width, target_height / height)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    return frame


def calculate_face_center(face_keypoints: dict) -> Optional[Tuple[float, float]]:
    """
    Calculate the center point of the face based on keypoints.
    
    Args:
        face_keypoints: Dictionary of face keypoints
    
    Returns:
        Face center coordinates or None if insufficient data
    """
    required_points = ['mouth_center', 'nose_tip']
    
    if not all(point in face_keypoints for point in required_points):
        return None
    
    mouth_center = face_keypoints['mouth_center']
    nose_tip = face_keypoints['nose_tip']
    
    # Calculate center between mouth and nose
    center_x = (mouth_center[0] + nose_tip[0]) / 2
    center_y = (mouth_center[1] + nose_tip[1]) / 2
    
    return center_x, center_y 