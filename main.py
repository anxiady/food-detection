#!/usr/bin/env python3
"""
Main application for real-time gesture recognition using MediaPipe.
"""

import cv2
import argparse
import sys
import time
from typing import Optional
import config
from keypoint_tracker import KeypointTracker
from gesture_detector import GestureDetector
from food_detector import FoodDetector
from object_detector import ObjectDetector
from visualizer import Visualizer
from testing_module import TestingModule
from hand_object_detector import HandObjectDetector
from simple_focus import SimpleFocus
from depth_estimator import DepthEstimator
from proximity_depth_estimator import ProximityDepthEstimator
from utils import validate_camera, resize_frame, create_timestamp


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time gesture recognition using MediaPipe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Use default camera (0)
  python main.py --camera 1                # Use camera 1
  python main.py --threshold 0.3           # Set feeding threshold to 0.3 seconds
  python main.py --buffer_size 60          # Use 60-frame buffer
  python main.py --debug                   # Enable debug mode
  python main.py --save_video output.mp4   # Save processed video
        """
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=config.CAMERA_INDEX,
        help=f'Camera device index (default: {config.CAMERA_INDEX})'
    )
    
    parser.add_argument(
        '--skip-camera-selection',
        action='store_true',
        help='Skip interactive camera selection'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=config.FEEDING_THRESHOLD_DURATION,
        help=f'Feeding gesture duration threshold in seconds (default: {config.FEEDING_THRESHOLD_DURATION})'
    )
    
    parser.add_argument(
        '--buffer_size', '-b',
        type=int,
        default=config.BUFFER_SIZE,
        help=f'Number of frames to buffer (default: {config.BUFFER_SIZE})'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode with additional visualizations'
    )
    
    parser.add_argument(
        '--save_video', '-s',
        action='store_true',
        help='Save processed video to file'
    )
    
    parser.add_argument(
        '--output_path', '-o',
        type=str,
        default=config.OUTPUT_PATH,
        help=f'Output video path (default: {config.OUTPUT_PATH})'
    )
    
    parser.add_argument(
        '--width', '-W',
        type=int,
        default=config.FRAME_WIDTH,
        help=f'Frame width (default: {config.FRAME_WIDTH})'
    )
    
    parser.add_argument(
        '--height', '-H',
        type=int,
        default=config.FRAME_HEIGHT,
        help=f'Frame height (default: {config.FRAME_HEIGHT})'
    )
    
    parser.add_argument(
        '--continuity', '-C',
        action='store_true',
        help='Auto-detect and use Continuity Camera from iPhone'
    )
    
    parser.add_argument(
        '--list-cameras', '-l',
        action='store_true',
        help='List all available cameras and their properties'
    )
    
    parser.add_argument(
        '--food-method', '-fm',
        type=str,
        default=config.FOOD_DETECTION_METHOD,
        choices=['yolo', 'huggingface', 'basic'],
        help=f'Food detection method (default: {config.FOOD_DETECTION_METHOD})'
    )
    
    parser.add_argument(
        '--food-only', '-fo',
        action='store_true',
        help='Focus only on food detection, disable gesture detection'
    )
    
    parser.add_argument(
        '--focus', '-f',
        action='store_true',
        help='Enable simple focus (fixed box in lower center)'
    )
    
    parser.add_argument(
        '--focus-size', '-fs',
        type=float,
        default=config.SIMPLE_FOCUS_BOX_RATIO,
        help=f'Focus box size as fraction (0-1) (default: {config.SIMPLE_FOCUS_BOX_RATIO})'
    )
    
    parser.add_argument(
        '--grey-value', '-gv',
        type=int,
        default=config.SIMPLE_FOCUS_GREY_VALUE,
        help=f'Grey background value (0-255) (default: {config.SIMPLE_FOCUS_GREY_VALUE})'
    )
    
    parser.add_argument(
        '--depth', '-dp',
        action='store_true',
        help='Enable depth estimation with MiDaS'
    )
    
    parser.add_argument(
        '--depth-min', '-dmin',
        type=float,
        default=config.DEPTH_MIN_DISTANCE,
        help=f'Minimum depth threshold (default: {config.DEPTH_MIN_DISTANCE})'
    )
    
    parser.add_argument(
        '--depth-max', '-dmax',
        type=float,
        default=config.DEPTH_MAX_DISTANCE,
        help=f'Maximum depth threshold (default: {config.DEPTH_MAX_DISTANCE})'
    )
    
    parser.add_argument(
        '--depth-heatmap', '-dh',
        action='store_true',
        help='Show depth heatmap overlay'
    )
    
    
    return parser.parse_args()


def update_config_from_args(args):
    """Update configuration based on command line arguments."""
    config.CAMERA_INDEX = args.camera
    config.FEEDING_THRESHOLD_DURATION = args.threshold
    config.BUFFER_SIZE = args.buffer_size
    config.DEBUG_MODE = args.debug
    config.SAVE_VIDEO = args.save_video
    config.OUTPUT_PATH = args.output_path
    config.FRAME_WIDTH = args.width
    config.FRAME_HEIGHT = args.height
    config.FOOD_DETECTION_METHOD = args.food_method
    
    # Simple focus settings
    config.SIMPLE_FOCUS_ENABLED = args.focus
    config.SIMPLE_FOCUS_BOX_RATIO = args.focus_size
    config.SIMPLE_FOCUS_GREY_VALUE = args.grey_value
    
    # Depth estimation settings
    config.DEPTH_ESTIMATION_ENABLED = args.depth
    config.DEPTH_MIN_DISTANCE = args.depth_min
    config.DEPTH_MAX_DISTANCE = args.depth_max
    config.DEPTH_SHOW_HEATMAP = args.depth_heatmap


def initialize_camera(camera_index: int) -> Optional[cv2.VideoCapture]:
    """
    Initialize camera with error handling.
    
    Args:
        camera_index: Camera device index
    
    Returns:
        VideoCapture object or None if failed
    """
    print(f"Initializing camera {camera_index}...")
    
    if not validate_camera(camera_index):
        print(f"Error: Camera {camera_index} is not available.")
        print("Available cameras:")
        for i in range(5):  # Check first 5 camera indices
            if validate_camera(i):
                print(f"  Camera {i}: Available")
        return None
    
    cap = cv2.VideoCapture(camera_index)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return None
    
    print(f"Camera {camera_index} initialized successfully")
    print(f"Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
    print(f"FPS: {config.FPS}")
    
    return cap


def initialize_video_writer(output_path: str, frame_width: int, frame_height: int) -> Optional[cv2.VideoWriter]:
    """
    Initialize video writer for saving processed video.
    
    Args:
        output_path: Output video file path
        frame_width: Frame width
        frame_height: Frame height
    
    Returns:
        VideoWriter object or None if failed
    """
    if not config.SAVE_VIDEO:
        return None
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, config.OUTPUT_FPS, (frame_width, frame_height))
    
    if not writer.isOpened():
        print(f"Error: Could not create video writer for {output_path}")
        return None
    
    print(f"Video writer initialized: {output_path}")
    return writer


def print_controls():
    """Print keyboard controls."""
    print("\nControls:")
    print("  'q' or 'ESC': Quit")
    print("  'r': Reset gesture detection")
    print("  'd': Toggle debug mode")
    print("  's': Toggle statistics display")
    print("  'k': Toggle keypoint display")
    print("  'b': Toggle bounding box display")
    print("  'f': Toggle FPS display")
    print("  'c': Switch to Continuity Camera")
    print("  'h': Show this help")
    print("\nFocus Controls:")
    print("  'f': Toggle simple focus")
    print("  'g': Toggle focus overlay")
    print("\nDepth Controls:")
    print("  'd': Toggle depth estimation")
    print("  't': Toggle depth overlay")
    print("\nTesting Module Controls:")
    print("  'v': Volume Control Mode")
    print("  'g': Gesture Accuracy Test Mode")
    print("  'h': Hand Object Detection Mode")
    print("  'n': Normal Mode")
    print()


def handle_keyboard_input(key: int, gesture_detector: GestureDetector, testing_module: TestingModule, simple_focus: SimpleFocus = None, depth_estimator: DepthEstimator = None, cap: cv2.VideoCapture = None):
    """
    Handle keyboard input for interactive controls.
    
    Args:
        key: Key code from cv2.waitKey()
        gesture_detector: Gesture detector instance
        testing_module: Testing module instance
        cap: VideoCapture instance for camera switching
    """
    if key in [ord('q'), 27]:  # 'q' or ESC
        return False
    
    elif key == ord('r'):
        gesture_detector.reset()
        print("Gesture detection reset")
    
    elif key == ord('d'):
        config.DEBUG_MODE = not config.DEBUG_MODE
        print(f"Debug mode: {'ON' if config.DEBUG_MODE else 'OFF'}")
    
    elif key == ord('s'):
        config.SHOW_DISTANCE = not config.SHOW_DISTANCE
        print(f"Distance display: {'ON' if config.SHOW_DISTANCE else 'OFF'}")
    
    elif key == ord('k'):
        config.SHOW_KEYPOINTS = not config.SHOW_KEYPOINTS
        print(f"Keypoint display: {'ON' if config.SHOW_KEYPOINTS else 'OFF'}")
    
    elif key == ord('b'):
        config.SHOW_BOUNDING_BOXES = not config.SHOW_BOUNDING_BOXES
        print(f"Bounding box display: {'ON' if config.SHOW_BOUNDING_BOXES else 'OFF'}")
    
    elif key == ord('f'):
        config.SHOW_FPS = not config.SHOW_FPS
        print(f"FPS display: {'ON' if config.SHOW_FPS else 'OFF'}")
    
    elif key == ord('c'):
        if cap is not None:
            # This will be handled in the main loop since we need to update the cap variable
            return 'switch_camera'
    
    elif key == ord('h'):
        print_controls()
    
    # Handle simple focus controls
    if simple_focus:
        if key == ord('f'):  # Toggle focus
            simple_focus.toggle_focus()
            print(f"Simple focus: {'ON' if simple_focus.enabled else 'OFF'}")
        
        elif key == ord('g'):  # Toggle focus overlay
            config.SIMPLE_FOCUS_SHOW_OVERLAY = not config.SIMPLE_FOCUS_SHOW_OVERLAY
            print(f"Focus overlay: {'ON' if config.SIMPLE_FOCUS_SHOW_OVERLAY else 'OFF'}")
    
    # Handle depth controls
    if depth_estimator:
        if key == ord('d'):  # Toggle depth estimation
            depth_estimator.toggle_depth()
            print(f"Depth estimation: {'ON' if depth_estimator.enabled else 'OFF'}")
        
        elif key == ord('t'):  # Toggle depth overlay
            config.DEPTH_SHOW_OVERLAY = not config.DEPTH_SHOW_OVERLAY
            print(f"Depth overlay: {'ON' if config.DEPTH_SHOW_OVERLAY else 'OFF'}")
    
    # Handle testing module keyboard input
    if not testing_module.handle_keyboard_input(key):
        return False
    
    return True

def switch_to_continuity_camera(cap: cv2.VideoCapture) -> cv2.VideoCapture:
    """
    Switch to Continuity Camera if available.
    
    Args:
        cap: Current VideoCapture instance
    
    Returns:
        New VideoCapture instance (either Continuity Camera or original)
    """
    print("\n=== Switching to Continuity Camera ===")
    from utils import find_continuity_camera
    
    # Find Continuity Camera
    continuity_camera = find_continuity_camera()
    
    if continuity_camera is not None:
        print(f"Found Continuity Camera at index {continuity_camera}")
        
        # Release current camera
        cap.release()
        
        # Initialize Continuity Camera
        new_cap = cv2.VideoCapture(continuity_camera)
        if new_cap.isOpened():
            # Set high resolution for Continuity Camera
            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            # Get actual properties
            width = new_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = new_cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Continuity Camera initialized successfully!")
            print(f"Resolution: {width}x{height}")
            print(f"FPS: {fps}")
            
            # Update configuration
            config.CAMERA_INDEX = continuity_camera
            config.FRAME_WIDTH = int(width)
            config.FRAME_HEIGHT = int(height)
            
            return new_cap
            
        else:
            print("Failed to initialize Continuity Camera")
            # Reinitialize default camera
            new_cap = cv2.VideoCapture(0)
            print("Reverted to default camera")
            return new_cap
    else:
        print("Continuity Camera not found. Make sure:")
        print("1. iPhone is connected and Continuity Camera is enabled")
        print("2. Both devices are signed in to the same iCloud account")
        print("3. Continuity Camera is enabled in System Settings")
        print("4. Open Camera app on iPhone → '...' → 'Continuity Camera'")
        return cap


def main():
    """Main application function."""
    print("=== Real-Time Food Recognition System ===")
    if config.CHEST_MOUNTED_MODE:
        print("Detecting food and feeding gestures from chest-mounted upward-facing camera view")
        print("Optimized for wearable camera setup with hand entry from bottom/sides")
    else:
        print("Detecting food and feeding gestures from front-facing camera view")
    print()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle list cameras option
    if args.list_cameras:
        print("\n=== Available Cameras ===")
        from utils import list_available_cameras
        cameras = list_available_cameras()
        if not cameras:
            print("No cameras found!")
        else:
            print(f"\nFound {len(cameras)} camera(s):")
            for camera in cameras:
                print(f"  Camera {camera['index']}: {camera['width']}x{camera['height']} @ {camera['fps']}fps")
        return
    
    # Ask user about Continuity Camera if not specified and camera selection not skipped
    if not args.continuity and not args.skip_camera_selection:
        print("\n=== Camera Selection ===")
        
        # First, check what cameras are available
        from utils import list_available_cameras, find_continuity_camera
        cameras = list_available_cameras()
        continuity_camera = find_continuity_camera()
        
        print("Available options:")
        print("1. Use default camera (built-in)")
        
        if continuity_camera is not None:
            print(f"2. Use Continuity Camera (iPhone) - Camera {continuity_camera}")
        else:
            print("2. Use Continuity Camera (iPhone) - Not detected")
            
        print("3. List available cameras")
        
        if len(cameras) > 1:
            print("4. Choose specific camera")
        
        try:
            choice = input("\nEnter your choice or press Enter for default: ").strip()
            
            if choice == "2":
                if continuity_camera is not None:
                    args.continuity = True
                    print(f"Selected: Continuity Camera (Camera {continuity_camera})")
                else:
                    print("Continuity Camera not detected. Please check setup:")
                    print("1. iPhone is connected and Continuity Camera is enabled")
                    print("2. Both devices are signed in to the same iCloud account")
                    print("3. Continuity Camera is enabled in System Settings")
                    print("4. Open Camera app on iPhone → '...' → 'Continuity Camera'")
                    print("Falling back to default camera...")
                    
            elif choice == "3":
                print("\n=== Available Cameras ===")
                if not cameras:
                    print("No cameras found!")
                else:
                    print(f"\nFound {len(cameras)} camera(s):")
                    for camera in cameras:
                        camera_type = "Continuity Camera" if camera['index'] == continuity_camera else "Built-in" if camera['index'] == 0 else "External"
                        print(f"  Camera {camera['index']}: {camera['width']}x{camera['height']} @ {camera['fps']}fps ({camera_type})")
                return
                
            elif choice == "4" and len(cameras) > 1:
                print("\n=== Choose Camera ===")
                for camera in cameras:
                    camera_type = "Continuity Camera" if camera['index'] == continuity_camera else "Built-in" if camera['index'] == 0 else "External"
                    print(f"  {camera['index']}: {camera['width']}x{camera['height']} @ {camera['fps']}fps ({camera_type})")
                
                try:
                    camera_choice = input(f"Enter camera number (0-{len(cameras)-1}): ").strip()
                    camera_index = int(camera_choice)
                    if 0 <= camera_index < len(cameras):
                        args.camera = camera_index
                        print(f"Selected: Camera {camera_index}")
                    else:
                        print("Invalid camera number. Using default camera.")
                except ValueError:
                    print("Invalid input. Using default camera.")
                    
            else:
                print("Selected: Default camera")
        except KeyboardInterrupt:
            print("\nUsing default camera")
    
    # Handle Continuity Camera detection if requested
    if args.continuity:
        print("\n=== Looking for Continuity Camera ===")
        from utils import find_continuity_camera
        continuity_camera = find_continuity_camera()
        if continuity_camera is not None:
            print(f"Found Continuity Camera at index {continuity_camera}")
            args.camera = continuity_camera
        else:
            print("Continuity Camera not found. Make sure:")
            print("1. iPhone is connected and Continuity Camera is enabled")
            print("2. Both devices are signed in to the same iCloud account")
            print("3. Continuity Camera is enabled in System Settings")
            print("4. Open Camera app on iPhone → '...' → 'Continuity Camera'")
            print("Falling back to default camera...")
    
    update_config_from_args(args)
    
    # Initialize camera
    cap = initialize_camera(args.camera)
    if cap is None:
        sys.exit(1)
    
    # Initialize components
    print("Initializing components...")
    tracker = KeypointTracker()
    detector = GestureDetector()
    food_detector = FoodDetector(detection_method=config.FOOD_DETECTION_METHOD)
    object_detector = ObjectDetector(detection_method=config.FOOD_DETECTION_METHOD)
    visualizer = Visualizer()
    testing_module = TestingModule()
    hand_object_detector = HandObjectDetector()
    simple_focus = SimpleFocus()
    depth_estimator = ProximityDepthEstimator()
    
    # Set frame dimensions for depth map resizing
    depth_estimator.set_frame_dimensions(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    
    # Check if food-only mode
    food_only_mode = args.food_only
    if food_only_mode:
        print("Running in FOOD-ONLY mode (gesture detection disabled)")
    else:
        print("Running in FULL mode (food + gesture detection)")
    
    # Initialize video writer if needed
    video_writer = initialize_video_writer(args.output_path, args.width, args.height)
    
    print("System initialized successfully!")
    print_controls()
    
    # Main processing loop
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Resize frame if needed
            frame = resize_frame(frame, args.width, args.height)
            
            # Apply simple focus processing if enabled
            processed_frame = simple_focus.process_frame(frame)
            
            # Process object detection (all objects, not just food) on processed frame
            object_data = object_detector.detect_objects(processed_frame)
            
            # Process food detection for food-specific analysis
            food_data = food_detector.detect_food(processed_frame)
            
            # Process depth estimation if enabled (extra layer)
            depth_map = None
            far_objects = []
            if depth_estimator.enabled:
                depth_map = depth_estimator.estimate_depth(frame)  # Use original frame for depth
                if depth_map is not None and object_data.get('detections'):
                    # Filter all object detections by depth
                    depth_result = depth_estimator.filter_objects_by_depth(
                        object_data['detections'], depth_map
                    )
                    far_objects = depth_result['far_objects']
                    # Update object_data with depth information
                    object_data['depth_available'] = depth_result['depth_available']
                    object_data['near_objects'] = depth_result['near_objects']
                    object_data['far_objects'] = far_objects
            
            # Process keypoints and gestures (only if not food-only mode)
            if not food_only_mode:
                detection_data = tracker.process_frame(processed_frame)
                
                # Detect hand objects
                if detection_data.get('hands'):
                    detection_data['hands'] = hand_object_detector.detect_hand_objects(detection_data['hands'])
                
                gesture_data = detector.detect_gestures(detection_data)
                statistics = detector.get_statistics()
            else:
                # In food-only mode, create empty data structures
                detection_data = {'hands': [], 'face': None}
                gesture_data = {}
                statistics = food_detector.get_statistics()
            
            # Apply testing module processing
            testing_processed_frame = testing_module.process_frame(processed_frame, detection_data, gesture_data)
            
            # Visualize results (only in normal mode)
            if testing_module.current_mode == 'normal':
                annotated_frame = visualizer.draw_frame(testing_processed_frame, detection_data, gesture_data, statistics, food_data, food_only_mode)
                # Draw all object detections
                annotated_frame = visualizer.draw_object_detections(annotated_frame, object_data)
            else:
                annotated_frame = testing_processed_frame
            
            # Draw focus overlay if enabled
            if config.SIMPLE_FOCUS_SHOW_OVERLAY and simple_focus.enabled:
                annotated_frame = simple_focus.draw_focus_overlay(annotated_frame)
            
            # Draw depth overlay if enabled (purple boxes for far objects)
            if config.DEPTH_SHOW_OVERLAY and depth_estimator.enabled and far_objects:
                annotated_frame = depth_estimator.draw_depth_overlay(annotated_frame, depth_map, far_objects)
            
            # Add debug information if enabled
            if config.DEBUG_MODE:
                if not food_only_mode:
                    proximity_distance = gesture_data.get('proximity_distance')
                    distance_str = f"{proximity_distance:.3f}" if proximity_distance is not None else "N/A"
                    debug_data = {
                        'Frame': frame_count,
                        'Hands': len(detection_data.get('hands', [])),
                        'Face': 'Detected' if detection_data.get('face') else 'None',
                        'Distance': distance_str,
                        'Buffer': len(detector.distance_buffer),
                        'Food': 'YES' if food_data.get('food_detected') else 'NO',
                        'Method': food_data.get('method', 'unknown')
                    }
                else:
                    debug_data = {
                        'Frame': frame_count,
                        'Food': 'YES' if food_data.get('food_detected') else 'NO',
                        'Objects': object_data.get('detection_count', 0),
                        'Stable': 'YES' if food_data.get('stable_detection') else 'NO',
                        'Count': food_data.get('detection_count', 0),
                        'Method': food_data.get('method', 'unknown')
                    }
                    
                    # Add depth information if available (using object data)
                    if depth_estimator.enabled and object_data.get('depth_available'):
                        near_count = len(object_data.get('near_objects', []))
                        far_count = len(object_data.get('far_objects', []))
                        debug_data.update({
                            'Depth': 'YES',
                            'Model': depth_estimator.model_name or 'unknown',
                            'Near': near_count,
                            'Far': far_count,
                            'Range': f"{depth_estimator.min_depth:.2f}-{depth_estimator.max_depth:.2f}"
                        })
                    elif depth_estimator.enabled:
                        debug_data.update({
                            'Depth': 'NO',
                            'Model': depth_estimator.model_name or 'unknown'
                        })
                    else:
                        debug_data.update({
                            'Depth': 'OFF',
                            'Model': 'disabled'
                        })
                annotated_frame = visualizer.draw_debug_info(annotated_frame, debug_data)
            
            # Save video if enabled
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Display frame
            cv2.imshow('Gesture Recognition', annotated_frame)
            
            # Handle keyboard input with FPS control
            delay = int(1000 / config.FPS)  # Convert FPS to milliseconds delay
            key = cv2.waitKey(delay) & 0xFF
            result = handle_keyboard_input(key, detector, testing_module, simple_focus, depth_estimator, cap)
            
            if result == 'switch_camera':
                cap = switch_to_continuity_camera(cap)
                # Reinitialize components with new camera
                tracker = KeypointTracker()
                detector = GestureDetector()
                food_detector = FoodDetector(detection_method=config.FOOD_DETECTION_METHOD)
                visualizer = Visualizer()
                testing_module = TestingModule()
                hand_object_detector = HandObjectDetector()
                simple_focus = SimpleFocus()
                depth_estimator = ProximityDepthEstimator()
                depth_estimator.set_frame_dimensions(config.FRAME_WIDTH, config.FRAME_HEIGHT)
                print("Components reinitialized for new camera")
            elif result is False:
                break
            
            frame_count += 1
            
            # Print periodic status
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                if food_only_mode:
                    food_stats = food_detector.get_statistics()
                    print(f"Processed {frame_count} frames, FPS: {fps:.1f}, "
                          f"Food Detections: {food_stats.get('total_detections', 0)}, "
                          f"Detection Rate: {food_stats.get('detection_rate', 0):.1%}")
                else:
                    print(f"Processed {frame_count} frames, FPS: {fps:.1f}, "
                          f"Feedings: {statistics.get('feeding_count', 0)}, "
                          f"Food Detections: {food_data.get('detection_count', 0)}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("Cleaning up...")
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        if not food_only_mode:
            tracker.release()
        food_detector.release()
        
        # Print final statistics
        if food_only_mode:
            final_stats = food_detector.get_statistics()
            print(f"\nFinal Food Detection Statistics:")
            print(f"  Total frames processed: {final_stats.get('total_frames', 0)}")
            print(f"  Average FPS: {final_stats.get('fps', 0):.1f}")
            print(f"  Food detections: {final_stats.get('total_detections', 0)}")
            print(f"  Detection rate: {final_stats.get('detection_rate', 0):.1%}")
            print(f"  Detection method: {final_stats.get('detection_method', 'unknown')}")
        else:
            final_stats = detector.get_statistics()
            food_stats = food_detector.get_statistics()
            print(f"\nFinal Statistics:")
            print(f"  Total frames processed: {final_stats.get('total_frames', 0)}")
            print(f"  Average FPS: {final_stats.get('fps', 0):.1f}")
            print(f"  Feeding gestures detected: {final_stats.get('feeding_count', 0)}")
            print(f"  Average hand-mouth distance: {final_stats.get('average_distance', 0):.3f}")
            print(f"  Food detections: {food_stats.get('total_detections', 0)}")
            print(f"  Food detection rate: {food_stats.get('detection_rate', 0):.1%}")
        
        if config.SAVE_VIDEO:
            print(f"  Video saved to: {args.output_path}")
        
        print("Application terminated successfully.")


if __name__ == "__main__":
    main() 