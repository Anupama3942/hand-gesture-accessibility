#!/bin/bash
echo "Fixing Python development environment on Fedora..."

# Install required system packages
sudo dnf install -y python3-devel redhat-rpm-config gcc-c++ make
sudo dnf install -y portaudio-devel alsa-lib-devel libevdev-devel

# Try to install system packages first
sudo dnf install -y python3-pyaudio python3-evdev 2>/dev/null || echo "System packages not available, will use pip"

# Install Python packages
pip install --upgrade pip

# Install core packages
pip install opencv-python mediapipe tensorflow numpy pyautogui pygetwindow pyttsx3 pynput flask speechrecognition

# Try to install problematic packages with proper headers
if ! python -c "import pyaudio" 2>/dev/null; then
    echo "Attempting to install pyaudio..."
    pip install pyaudio
fi

if ! python -c "import evdev" 2>/dev/null; then
    echo "Attempting to install evdev..."
    pip install evdev
fi

echo "Testing imports..."
python -c "
import sys
print('Python version:', sys.version)

success = []
failed = []

packages = ['cv2', 'mediapipe', 'tensorflow', 'numpy', 'pyautogui', 
            'pygetwindow', 'pyttsx3', 'pynput', 'flask', 'speech_recognition']

for pkg in packages:
    try:
        __import__(pkg.replace('_', ''))
        success.append(pkg)
    except ImportError:
        failed.append(pkg)

print(f'✓ Successfully imported: {success}')
if failed:
    print(f'✗ Failed to import: {failed}')
"

echo "Setup complete!"
# Save this script as setup_fedora_fix.sh and run it with:  
# bash setup_fedora_fix.sh
# This script is intended for Fedora-based systems to fix common Python package installation issues.
# It installs necessary development tools and libraries, then attempts to install and verify key Python packages.
# Adjust package names and installation commands as needed for your specific distribution.
# Note: Some packages may still require manual intervention if they fail to install.
# Always run such scripts in a virtual environment to avoid conflicts with system packages.
# Remember to check for any additional dependencies specific to your project.
# Always review and understand scripts before executing them on your system.
# This script assumes you have sudo privileges on the system.
# For any issues, refer to the official documentation of the packages being installed.


# End of script
