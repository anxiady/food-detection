# Food Recognition System

Real-time food detection using computer vision.

## Features

- **Food Detection**: Detects food items in video streams
- **Depth Estimation**: Shows proximity of detected objects
- **Object Filtering**: Ignores laptop, person, chair objects
- **Real-time Processing**: Optimized for live video analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Application
```bash
python main.py
```

### iPhone Continuity Camera (MacBook)
The application will automatically detect and prompt for Continuity Camera when available.

## Controls

- `d`: Toggle depth estimation
- `f`: Toggle focus mode
- `q`: Quit

## Requirements

- Python 3.8+
- OpenCV
- YOLOv8 model (included)
- MediaPipe (for gesture detection)

## Files

- `main.py`: Main application
- `object_detector.py`: Object detection
- `proximity_depth_estimator.py`: Depth estimation
- `visualizer.py`: Display
- `config.py`: Settings