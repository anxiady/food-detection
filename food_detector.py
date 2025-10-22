"""
Food detection module for identifying food items in video frames.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import time
from collections import deque
import config


class FoodDetector:
    """
    Detects food items in video frames using multiple detection approaches.
    """
    
    def __init__(self, detection_method: str = 'yolo'):
        """
        Initialize food detector.
        
        Args:
            detection_method: Detection method to use ('yolo', 'huggingface', 'opencv')
        """
        self.detection_method = detection_method
        self.model = None
        self.class_names = []
        self.confidence_threshold = config.FOOD_DETECTION_CONFIDENCE
        self.nms_threshold = config.FOOD_NMS_THRESHOLD
        
        # Detection history for stability
        self.detection_history = deque(maxlen=config.DETECTION_BUFFER_SIZE)
        self.food_detected_buffer = deque(maxlen=config.DETECTION_BUFFER_SIZE)
        
        # Performance tracking
        self.frame_count = 0
        self.total_detections = 0
        self.start_time = time.time()
        
        # Initialize the selected detection method
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the selected detection method."""
        try:
            if self.detection_method == 'yolo':
                self._initialize_yolo()
            elif self.detection_method == 'huggingface':
                self._initialize_huggingface()
            elif self.detection_method == 'opencv':
                self._initialize_opencv()
            else:
                raise ValueError(f"Unsupported detection method: {self.detection_method}")
                
            print(f"Food detector initialized with method: {self.detection_method}")
            
        except Exception as e:
            print(f"Error initializing {self.detection_method} detector: {e}")
            print("Falling back to basic color-based detection...")
            self.detection_method = 'basic'
            self._initialize_basic_detector()
    
    def _initialize_yolo(self):
        """Initialize YOLOv8 food detection model."""
        try:
            # Try to import ultralytics
            from ultralytics import YOLO
            
            # Try to load a custom food model or use COCO model as fallback
            try:
                # Try custom food model first
                self.model = YOLO('yolov8n.pt')  # Start with base model
                print("Using YOLOv8 base model (includes some food classes)")
            except:
                # Fallback to downloading standard model
                self.model = YOLO('yolov8n.pt')
                print("Downloaded and loaded YOLOv8 nano model")
            
            # COCO class names that include food items
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'
            ]
            
            # Food-related class indices in COCO
            self.food_classes = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]  # bottle through cake
            
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
    
    def _initialize_huggingface(self):
        """Initialize Hugging Face food detection model."""
        try:
            from transformers import pipeline
            
            # Use food-specific model from Hugging Face
            self.model = pipeline(
                "image-classification",
                model="Jacques7103/Food-Recognition",
                trust_remote_code=True
            )
            print("Loaded Hugging Face food recognition model")
            
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers torch")
    
    def _initialize_opencv(self):
        """Initialize OpenCV DNN with pre-trained model."""
        try:
            # This would load a custom food detection model
            # For now, we'll use a placeholder approach
            print("OpenCV DNN food detection not implemented yet")
            raise NotImplementedError("OpenCV food detection needs custom model")
            
        except Exception as e:
            raise e
    
    def _initialize_basic_detector(self):
        """Initialize basic color/texture based food detection as fallback."""
        print("Using basic color-based food detection")
        self.detection_method = 'basic'
        
        # Define food-like color ranges in HSV
        self.food_color_ranges = {
            'fruits': [
                ([0, 50, 50], [10, 255, 255]),    # Red fruits (apples, strawberries)
                ([160, 50, 50], [180, 255, 255]), # Red fruits (continued range)
                ([10, 50, 50], [25, 255, 255]),   # Orange fruits (oranges, carrots)
                ([25, 50, 50], [35, 255, 255]),   # Yellow fruits (bananas, lemons)
                ([35, 50, 50], [85, 255, 255]),   # Green fruits/vegetables
            ],
            'bread': [
                ([15, 30, 80], [30, 180, 220]),   # Brown/tan colors (bread, pastries)
            ],
            'cooked_food': [
                ([0, 30, 50], [15, 150, 200]),    # Cooked meat colors
                ([15, 40, 60], [25, 200, 200]),   # Golden/fried food colors
            ]
        }
    
    def detect_food(self, frame: np.ndarray) -> Dict:
        """
        Detect food items in the given frame.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Dictionary containing detection results
        """
        self.frame_count += 1
        
        if self.detection_method == 'yolo':
            return self._detect_food_yolo(frame)
        elif self.detection_method == 'huggingface':
            return self._detect_food_huggingface(frame)
        elif self.detection_method == 'opencv':
            return self._detect_food_opencv(frame)
        else:  # basic
            return self._detect_food_basic(frame)
    
    def _detect_food_yolo(self, frame: np.ndarray) -> Dict:
        """Detect food using YOLO model."""
        try:
            # Run YOLO inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            food_detected = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a food-related class
                        if class_id in self.food_classes:
                            food_detected = True
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            detection = {
                                'class_id': class_id,
                                'class_name': self.class_names[class_id],
                                'confidence': confidence,
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                'area': (x2 - x1) * (y2 - y1)
                            }
                            detections.append(detection)
            
            # Update detection history
            self._update_detection_history(food_detected, detections)
            
            return {
                'food_detected': food_detected,
                'stable_detection': self._get_stable_detection(),
                'detections': detections,
                'detection_count': len(detections),
                'method': 'yolo',
                'frame_count': self.frame_count
            }
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return self._empty_detection_result()
    
    def _detect_food_huggingface(self, frame: np.ndarray) -> Dict:
        """Detect food using Hugging Face model."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(rgb_frame)
            
            food_detected = False
            detections = []
            
            # Process results
            for result in results:
                confidence = result['score']
                if confidence > self.confidence_threshold:
                    food_detected = True
                    detection = {
                        'class_name': result['label'],
                        'confidence': confidence,
                        'bbox': None,  # HF classification doesn't provide bbox
                        'center': (frame.shape[1] // 2, frame.shape[0] // 2),  # Center of frame
                        'area': frame.shape[0] * frame.shape[1]  # Full frame
                    }
                    detections.append(detection)
            
            # Update detection history
            self._update_detection_history(food_detected, detections)
            
            return {
                'food_detected': food_detected,
                'stable_detection': self._get_stable_detection(),
                'detections': detections,
                'detection_count': len(detections),
                'method': 'huggingface',
                'frame_count': self.frame_count
            }
            
        except Exception as e:
            print(f"Hugging Face detection error: {e}")
            return self._empty_detection_result()
    
    def _detect_food_opencv(self, frame: np.ndarray) -> Dict:
        """Detect food using OpenCV DNN."""
        # Placeholder for OpenCV DNN implementation
        return self._empty_detection_result()
    
    def _detect_food_basic(self, frame: np.ndarray) -> Dict:
        """Basic food detection using color analysis."""
        try:
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            food_detected = False
            detections = []
            
            # Check for food-like colors
            for food_type, color_ranges in self.food_color_ranges.items():
                for lower, upper in color_ranges:
                    lower = np.array(lower)
                    upper = np.array(upper)
                    
                    # Create mask for this color range
                    mask = cv2.inRange(hsv, lower, upper)
                    
                    # Find contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        
                        # Filter by minimum area
                        if area > config.MIN_FOOD_AREA:
                            food_detected = True
                            
                            # Get bounding box
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            detection = {
                                'class_name': f'{food_type}_colored_object',
                                'confidence': min(0.8, area / 10000),  # Rough confidence based on area
                                'bbox': (x, y, x + w, y + h),
                                'center': (x + w // 2, y + h // 2),
                                'area': area
                            }
                            detections.append(detection)
            
            # Update detection history
            self._update_detection_history(food_detected, detections)
            
            return {
                'food_detected': food_detected,
                'stable_detection': self._get_stable_detection(),
                'detections': detections,
                'detection_count': len(detections),
                'method': 'basic',
                'frame_count': self.frame_count
            }
            
        except Exception as e:
            print(f"Basic detection error: {e}")
            return self._empty_detection_result()
    
    def _update_detection_history(self, food_detected: bool, detections: List[Dict]):
        """Update detection history for stability analysis."""
        self.food_detected_buffer.append(food_detected)
        self.detection_history.append({
            'timestamp': time.time(),
            'food_detected': food_detected,
            'detection_count': len(detections),
            'detections': detections
        })
        
        if food_detected:
            self.total_detections += 1
    
    def _get_stable_detection(self) -> bool:
        """Get stable detection result based on recent history."""
        if len(self.food_detected_buffer) < config.MIN_STABLE_FRAMES:
            return False
        
        # Require a certain percentage of recent frames to have food
        recent_detections = list(self.food_detected_buffer)[-config.MIN_STABLE_FRAMES:]
        detection_ratio = sum(recent_detections) / len(recent_detections)
        
        return detection_ratio >= config.STABLE_DETECTION_THRESHOLD
    
    def _empty_detection_result(self) -> Dict:
        """Return empty detection result."""
        return {
            'food_detected': False,
            'stable_detection': False,
            'detections': [],
            'detection_count': 0,
            'method': self.detection_method,
            'frame_count': self.frame_count
        }
    
    def get_statistics(self) -> Dict:
        """Get food detection statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        return {
            'total_frames': self.frame_count,
            'fps': self.frame_count / elapsed_time if elapsed_time > 0 else 0,
            'total_detections': self.total_detections,
            'detection_rate': self.total_detections / self.frame_count if self.frame_count > 0 else 0,
            'current_stable_detection': self._get_stable_detection(),
            'detection_method': self.detection_method,
            'buffer_size': len(self.detection_history)
        }
    
    def reset(self):
        """Reset detection history and statistics."""
        self.detection_history.clear()
        self.food_detected_buffer.clear()
        self.frame_count = 0
        self.total_detections = 0
        self.start_time = time.time()
        print("Food detector reset")
    
    def release(self):
        """Release any resources used by the detector."""
        if hasattr(self.model, 'close'):
            self.model.close()
        print(f"Food detector ({self.detection_method}) released")
