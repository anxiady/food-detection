"""
Object detection module for identifying all objects in video frames.
Uses YOLO to detect all COCO classes, not just food items.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import time
from collections import deque
import config


class ObjectDetector:
    """
    Detects all objects in video frames using YOLO.
    """
    
    def __init__(self, detection_method: str = 'yolo'):
        """
        Initialize object detector.
        
        Args:
            detection_method: Detection method to use ('yolo', 'huggingface', 'opencv')
        """
        self.detection_method = detection_method
        self.model = None
        self.class_names = []
        self.confidence_threshold = config.FOOD_DETECTION_CONFIDENCE
        self.nms_threshold = config.FOOD_NMS_THRESHOLD
        
        # Objects to ignore (filter out from detection)
        self.ignored_classes = ['laptop', 'person', 'chair']
        
        # Detection history for stability
        self.detection_history = deque(maxlen=config.DETECTION_BUFFER_SIZE)
        self.objects_detected_buffer = deque(maxlen=config.DETECTION_BUFFER_SIZE)
        
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
                raise ValueError(f"Unknown detection method: {self.detection_method}")
                
        except Exception as e:
            print(f"Error initializing {self.detection_method} detector: {e}")
            raise
    
    def _initialize_yolo(self):
        """Initialize YOLO detector for all objects."""
        try:
            from ultralytics import YOLO
            
            # Try to load existing model first
            try:
                self.model = YOLO('yolov8n.pt')
                print("Loaded existing YOLOv8 nano model")
            except:
                # Fallback to downloading standard model
                self.model = YOLO('yolov8n.pt')
                print("Downloaded and loaded YOLOv8 nano model")
            
            # All COCO class names (80 classes)
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush'
            ]
            
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
    
    def _initialize_huggingface(self):
        """Initialize Hugging Face object detection model."""
        try:
            from transformers import pipeline
            
            # Use general object detection model
            self.model = pipeline(
                "object-detection",
                model="facebook/detr-resnet-50",
                trust_remote_code=True
            )
            print("Loaded Hugging Face DETR model for object detection")
            
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")
    
    def _initialize_opencv(self):
        """Initialize OpenCV-based object detection."""
        try:
            # Load OpenCV DNN model
            model_path = 'yolov8n.onnx'
            config_path = 'yolov8n.cfg'
            
            if not os.path.exists(model_path):
                print("OpenCV model not found, using basic detection")
                self.model = None
            else:
                self.model = cv2.dnn.readNet(model_path, config_path)
                print("Loaded OpenCV DNN model")
                
        except Exception as e:
            print(f"OpenCV model loading failed: {e}")
            self.model = None
    
    def detect_objects(self, frame: np.ndarray) -> Dict:
        """
        Detect all objects in the frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing detection results
        """
        self.frame_count += 1
        
        if self.model is None:
            return {
                'objects_detected': False,
                'detections': [],
                'detection_count': 0,
                'method': 'none',
                'stable_detection': False
            }
        
        try:
            if self.detection_method == 'yolo':
                return self._detect_objects_yolo(frame)
            elif self.detection_method == 'huggingface':
                return self._detect_objects_huggingface(frame)
            elif self.detection_method == 'opencv':
                return self._detect_objects_opencv(frame)
            else:
                return {
                    'objects_detected': False,
                    'detections': [],
                    'detection_count': 0,
                    'method': 'unknown',
                    'stable_detection': False
                }
                
        except Exception as e:
            print(f"Object detection error: {e}")
            return {
                'objects_detected': False,
                'detections': [],
                'detection_count': 0,
                'method': 'error',
                'stable_detection': False
            }
    
    def _detect_objects_yolo(self, frame: np.ndarray) -> Dict:
        """Detect objects using YOLO."""
        try:
            # Run YOLO inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter by confidence and ignored classes
                        if confidence >= self.confidence_threshold:
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                            
                            # Skip ignored classes
                            if class_name.lower() in self.ignored_classes:
                                continue
                            
                            detection = {
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'class_name': class_name
                            }
                            detections.append(detection)
            
            # Apply NMS
            if len(detections) > 1:
                detections = self._apply_nms(detections)
            
            # Update detection history
            self.detection_history.append(len(detections))
            self.objects_detected_buffer.append(len(detections) > 0)
            
            # Check for stable detection
            stable_detection = self._check_stable_detection()
            
            return {
                'objects_detected': len(detections) > 0,
                'detections': detections,
                'detection_count': len(detections),
                'method': 'yolo',
                'stable_detection': stable_detection
            }
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return {
                'objects_detected': False,
                'detections': [],
                'detection_count': 0,
                'method': 'yolo_error',
                'stable_detection': False
            }
    
    def _detect_objects_huggingface(self, frame: np.ndarray) -> Dict:
        """Detect objects using Hugging Face model."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(rgb_frame)
            
            detections = []
            for result in results:
                if result['score'] >= self.confidence_threshold:
                    class_name = result['label'].lower()
                    
                    # Skip ignored classes
                    if class_name in self.ignored_classes:
                        continue
                    
                    bbox = result['box']
                    detection = {
                        'bbox': (int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])),
                        'confidence': float(result['score']),
                        'class_name': result['label']
                    }
                    detections.append(detection)
            
            # Update detection history
            self.detection_history.append(len(detections))
            self.objects_detected_buffer.append(len(detections) > 0)
            
            # Check for stable detection
            stable_detection = self._check_stable_detection()
            
            return {
                'objects_detected': len(detections) > 0,
                'detections': detections,
                'detection_count': len(detections),
                'method': 'huggingface',
                'stable_detection': stable_detection
            }
            
        except Exception as e:
            print(f"Hugging Face detection error: {e}")
            return {
                'objects_detected': False,
                'detections': [],
                'detection_count': 0,
                'method': 'huggingface_error',
                'stable_detection': False
            }
    
    def _detect_objects_opencv(self, frame: np.ndarray) -> Dict:
        """Detect objects using OpenCV DNN."""
        try:
            # This would implement OpenCV DNN detection
            # For now, return empty results
            return {
                'objects_detected': False,
                'detections': [],
                'detection_count': 0,
                'method': 'opencv',
                'stable_detection': False
            }
            
        except Exception as e:
            print(f"OpenCV detection error: {e}")
            return {
                'objects_detected': False,
                'detections': [],
                'detection_count': 0,
                'method': 'opencv_error',
                'stable_detection': False
            }
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to detections."""
        if len(detections) <= 1:
            return detections
        
        # Extract bounding boxes and scores
        boxes = []
        scores = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            boxes.append([x1, y1, x2, y2])
            scores.append(det['confidence'])
        
        # Apply NMS
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            self.confidence_threshold, 
            self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return []
    
    def _check_stable_detection(self) -> bool:
        """Check if detection is stable over recent frames."""
        if len(self.objects_detected_buffer) < config.DETECTION_BUFFER_SIZE:
            return False
        
        # Check if objects have been detected consistently
        recent_detections = list(self.objects_detected_buffer)
        detection_rate = sum(recent_detections) / len(recent_detections)
        
        return detection_rate >= config.STABLE_DETECTION_THRESHOLD
    
    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        return {
            'total_frames': self.frame_count,
            'fps': self.frame_count / elapsed_time if elapsed_time > 0 else 0,
            'total_detections': self.total_detections,
            'detection_rate': self.total_detections / self.frame_count if self.frame_count > 0 else 0,
            'method': self.detection_method,
            'model_loaded': self.model is not None
        }
    
    def release(self):
        """Release resources."""
        if self.model is not None:
            # YOLO models don't need explicit cleanup
            pass
