"""
Configuration settings for the gesture recognition system.
"""

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15

# MediaPipe settings
HANDS_MAX_NUM = 2  # Maximum number of hands to detect
FACE_DETECTION_CONFIDENCE = 0.5
HAND_DETECTION_CONFIDENCE = 0.5

# Gesture detection settings
FEEDING_THRESHOLD_DURATION = 0.5  # seconds
PROXIMITY_THRESHOLD = 0.12  # normalized distance (0-1) - adjusted for chest-mounted camera
BUFFER_SIZE = 30  # number of frames to buffer
CONFIDENCE_THRESHOLD = 0.7  # minimum confidence for keypoint detection

# Chest-mounted camera specific settings
CHEST_MOUNTED_MODE = True  # Enable chest-mounted camera optimizations
HAND_ENTRY_ZONE_LOWER = 0.3  # Lower zone threshold (hands should be below this Y coordinate)
HAND_ENTRY_ZONE_SIDES = 0.2  # Side zone threshold (hands should be outside X < 0.2 or X > 0.8)

# Keypoint indices for MediaPipe
HAND_KEYPOINTS = {
    'wrist': 0,
    'thumb_tip': 4,
    'thumb_ip': 3,
    'thumb_mcp': 2,
    'index_tip': 8,
    'index_pip': 7,
    'index_mcp': 6,
    'middle_tip': 12,
    'middle_pip': 11,
    'middle_mcp': 10,
    'ring_tip': 16,
    'ring_pip': 15,
    'ring_mcp': 14,
    'pinky_tip': 20,
    'pinky_pip': 19,
    'pinky_mcp': 18,
    'palm_center': 9  # Use index finger pip as palm center approximation
}

FACE_KEYPOINTS = {
    'chin': 152,  # Chin center
    'mouth_left': 61,  # Left corner of mouth
    'mouth_right': 291,  # Right corner of mouth
    'mouth_center': 13,  # Center of mouth
    'nose_tip': 1  # Nose tip
}

# Visual settings
COLORS = {
    'hand': (0, 255, 0),  # Green
    'face': (255, 0, 0),  # Blue
    'feeding_detected': (0, 0, 255),  # Red
    'proximity_line': (255, 255, 0),  # Yellow
    'text': (255, 255, 255),  # White
    'background': (0, 0, 0)  # Black
}

# Drawing settings
LINE_THICKNESS = 2
CIRCLE_RADIUS = 5
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2

# Performance settings
SKIP_FRAMES = 1  # Process every Nth frame for performance
ENABLE_SMOOTHING = True
SMOOTHING_ALPHA = 0.7  # Exponential smoothing factor

# Debug settings
DEBUG_MODE = False
SHOW_FPS = True
SHOW_DISTANCE = True
SHOW_KEYPOINTS = True
SHOW_BOUNDING_BOXES = True

# Output settings
SAVE_VIDEO = False
OUTPUT_PATH = "output.mp4"
OUTPUT_FPS = 30

# Food detection settings
FOOD_DETECTION_CONFIDENCE = 0.5  # Confidence threshold for food detection
FOOD_NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold
DETECTION_BUFFER_SIZE = 15  # Frames to buffer for stable detection
MIN_STABLE_FRAMES = 5  # Minimum frames needed for stable detection
STABLE_DETECTION_THRESHOLD = 0.6  # Ratio of frames that must contain food for stable detection
MIN_FOOD_AREA = 1000  # Minimum area for food detection (basic method)
FOOD_DETECTION_METHOD = 'yolo'  # Options: 'yolo', 'huggingface', 'basic'

# Testing module settings
TESTING_ENABLED = True
VOLUME_CONTROL_ENABLED = True
GESTURE_ACCURACY_TEST_ENABLED = True
HAND_OBJECT_DETECTION_ENABLED = True
PLOT_WIDTH = 300
PLOT_HEIGHT = 200
OVERLAY_ALPHA = 0.8

# Simple Focus settings
SIMPLE_FOCUS_ENABLED = False  # Enable/disable simple focus
SIMPLE_FOCUS_BOX_RATIO = 0.6  # Box size as fraction of frame (0-1) - increased by 50%
SIMPLE_FOCUS_GREY_VALUE = 128  # Grey color value (0-255)
SIMPLE_FOCUS_SHOW_OVERLAY = True  # Show focus box overlay

# Depth Estimation settings
DEPTH_ESTIMATION_ENABLED = False  # Enable/disable depth estimation
DEPTH_MIN_DISTANCE = 0.3  # Minimum depth threshold (normalized 0-1)
DEPTH_MAX_DISTANCE = 0.7  # Maximum depth threshold (normalized 0-1)
DEPTH_DISTANCE_THRESHOLD = 0.5  # Default depth threshold
DEPTH_SHOW_HEATMAP = False  # Show depth heatmap overlay
DEPTH_SHOW_OVERLAY = True  # Show depth-based detection overlay 