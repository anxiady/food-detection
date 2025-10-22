# Quick Start Guide

Get the gesture recognition system running in minutes!

## Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- Internet connection (for downloading dependencies)

## Installation

### Option 1: Automatic Setup (Recommended)

```bash
# Run the setup script
python setup.py
```

This will:
- Check Python version
- Install all dependencies
- Create necessary directories
- Run installation tests
- Make scripts executable

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Test installation
python test_installation.py
```

## Running the System

### Basic Usage

```bash
# Start with default settings
python main.py
```

### Advanced Usage

```bash
# Use different camera
python main.py --camera 1

# Adjust sensitivity
python main.py --threshold 0.3

# Enable debug mode
python main.py --debug

# Save processed video
python main.py --save_video

# Full example
python main.py --camera 0 --threshold 0.5 --debug --save_video output.mp4
```

## Controls

While the application is running:

### Basic Controls
- **'q' or 'ESC'**: Quit
- **'r'**: Reset gesture detection
- **'d'**: Toggle debug mode
- **'s'**: Toggle distance display
- **'k'**: Toggle keypoint display
- **'b'**: Toggle bounding boxes
- **'f'**: Toggle FPS display
- **'h'**: Show help

### Testing Module Controls
- **'v'**: Volume Control Mode - Control system volume with comfortable finger distance
- **'g'**: Gesture Accuracy Test Mode - Real-time distance plotting and validation
- **'h'**: Hand Object Detection Mode - Detailed analysis of hand grip types and object holding
- **'n'**: Normal Mode - Standard gesture recognition display

### Volume Control Tips
- **Pinch fingers together** (very close) = **Mute (0% volume)**
- **Spread fingers comfortably** (not stretched) = **Maximum volume (100%)**
- **Fine-tune** by adjusting the distance between index and thumb
- **Watch the comfort zone indicator** for guidance

### Hand Object Detection Tips
- **Press 'h'** to enter Hand Object Detection Mode
- **Show both hands** to the camera for analysis
- **Try different objects**:
  - **Cups, utensils** = Loose grip detection
  - **Small items (chips, nuts)** = Small item pinch detection
  - **Empty hands** = Open hand detection
- **Watch the detailed analysis** showing finger curl measurements
- **Green boxes** indicate hands holding objects
- **Red boxes** indicate empty hands

## Testing

### Test Installation
```bash
python test_installation.py
```

### Run Examples
```bash
python example_usage.py
```

### Test Testing Module
```bash
python test_testing_module.py
```

## Troubleshooting

### Camera Issues
- Try different camera indices: `--camera 0`, `--camera 1`, etc.
- Check camera permissions
- Ensure camera is not in use by another application

### Performance Issues
- Reduce frame resolution: `--width 320 --height 240`
- Disable debug mode
- Reduce buffer size: `--buffer_size 15`

### Installation Issues
- Update pip: `pip install --upgrade pip`
- Install system dependencies (Ubuntu/Debian):
  ```bash
  sudo apt-get install python3-opencv
  ```
- On macOS with Homebrew:
  ```bash
  brew install opencv
  ```

## Configuration

Edit `config.py` to customize:
- Detection thresholds
- Visual appearance
- Performance settings
- Camera parameters

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Try the examples** in `example_usage.py`
3. **Customize settings** in `config.py`
4. **Extend the system** for your specific needs

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run `python test_installation.py` to diagnose problems
3. Check the full README.md for detailed information
4. Ensure all dependencies are properly installed

Happy gesture recognition! 🎉 