# Hand Gesture Accessibility System

A powerful, AI-driven hands-free computer control system that enables users to operate their computer using hand gestures and voice commands. Designed for accessibility and convenience, this system provides a comprehensive interface for mouse control, media playback, and system navigation.

## 🌟 Key Features

### ✋ Hand Gesture Control
- **Advanced Tracking**: Utilizes MediaPipe for real-time, high-precision hand tracking.
- **30+ Supported Gestures**:
  - **Mouse**: Move cursor, Left/Right Click, Double Click, Drag & Drop.
  - **Scrolling**: Scroll Up/Down.
  - **Zoom**: Zoom In/Out.
  - **Browser**: Back, Forward, New Tab, Close Tab, Switch Tabs.
  - **Media**: Play/Pause, Volume Up/Down, Mute, Next/Prev Track.
  - **System**: Copy, Paste, Cut, Undo, Redo, Save, Switch App, Desktop, Task View.

### 🎙️ Voice Feedback
- **Audio Feedback**: Provides spoken confirmation for recognized gestures (via `pyttsx3`).
- **Status Announcements**: alerts for system start/stop and critical errors.

### 🖥️ Web Dashboard
- **Live Monitoring**: View real-time camera feed with gesture and hand tracking overlays.
- **Performance Stats**: Monitor FPS, latency, and memory usage.
- **Gesture Training**: Built-in interface to record and train new custom gestures.
- **Settings Management**: Adjust sensitivity, speed, and other parameters on the fly.

### 🛡️ Robust & Secure
- **Authentication**: Optional login system for the web interface.
- **Fallbacks**: Gracefully handles missing optional dependencies (e.g., runs without voice feedback if `pyttsx3` is missing).
- **Emergency Stop**: Quick access to stop control if needed.

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Microphone (optional, for voice commands)

### Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd hand-gesture-accessibility
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have issues with TensorFlow, `requirements_no_tf.txt` is available for lighter setups, though some features may be limited.*

## 🚀 Usage

1.  **Start the Application**
    ```bash
    python run_app.py
    ```

2.  **Access the Dashboard**
    Open your web browser and navigate to:
    ```
    http://localhost:5000
    ```

3.  **Operation**
    - **Cursor Mode**: Use the "Index Finger Up" gesture (or configured gesture) to move the mouse.
    - **Gestures**: Perform hand signs to execute commands. The system will announce the recognized gesture.
    - **Training**: Go to the "Training" tab in the web UI to teach the system new gestures.

## ⚙️ Configuration

You can configure the system via the **Settings** page in the web dashboard or by editing `config.py`.

**Key Settings:**
- **Sensitivity**: Adjust how responsive the cursor is to hand movement.
- **Cursor Speed**: Set the base speed for mouse movement.
- **Target FPS**: Cap the frame rate to manage CPU usage.
- **Auth**: Enable/Disable password protection for the web interface.

## 📂 Project Structure

- `run_app.py`: Entry point for the application.
- `accessibility_controller.py`: Core logic for gesture recognition and system control.
- `accessibility_web.py`: Flask web server for the dashboard and API.
- `config.py`: Configuration management and validation.
- `training_ui.py`: Logic specific to the training interface.
- `models/`: Stores trained gesture recognition models.
- `logs/`: System logs for debugging.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[License Name] - See the LICENSE file for details.