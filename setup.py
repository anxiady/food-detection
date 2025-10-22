#!/usr/bin/env python3
"""
Setup script for the gesture recognition system.
"""

import os
import sys
import subprocess
import platform


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python {sys.version.split()[0]} detected")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = ["output", "logs", "data"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")


def make_executable():
    """Make scripts executable on Unix-like systems."""
    if platform.system() != "Windows":
        scripts = ["main.py", "test_installation.py", "example_usage.py"]
        
        for script in scripts:
            if os.path.exists(script):
                try:
                    os.chmod(script, 0o755)
                    print(f"✓ Made {script} executable")
                except Exception as e:
                    print(f"Warning: Could not make {script} executable: {e}")


def run_tests():
    """Run installation tests."""
    print("\nRunning installation tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_installation.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ All tests passed!")
            return True
        else:
            print("✗ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Could not run tests: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. Test the system:")
    print("   python test_installation.py")
    print("\n2. Run the main application:")
    print("   python main.py")
    print("\n3. Try examples:")
    print("   python example_usage.py")
    print("\n4. View help:")
    print("   python main.py --help")
    print("\nFor more information, see README.md")


def main():
    """Main setup function."""
    print("=== Gesture Recognition System Setup ===")
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\nPlease install dependencies manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Make scripts executable
    make_executable()
    
    # Run tests
    if run_tests():
        print_next_steps()
    else:
        print("\nSetup completed with warnings.")
        print("Please run 'python test_installation.py' to check your installation.")
        print("If tests fail, you may need to install additional system dependencies.")


if __name__ == "__main__":
    main() 