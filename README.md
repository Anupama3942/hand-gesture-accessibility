# Hand Gesture Accessibility System

Clean workflow for running, collecting gesture samples, and training models.

## Requirements

- Windows + webcam
- Python 3.13.x

## Setup (one-time)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run App (recommended)

Always run with the venv interpreter:

```powershell
.\venv\Scripts\python.exe run_app.py
```

Open: `http://127.0.0.1:5000`

## Train Gestures (Web UI)

1. Open `http://127.0.0.1:5000/training`
2. Select one gesture
3. Click **Start Training**
4. Keep hand visible and collect samples
5. Click **Train Model**
6. Click **Save Model**

## Alternative Collection Script

```powershell
.\venv\Scripts\python.exe collect_gesture_data.py
```

Use `c` to capture samples for the selected gesture.

## Common Issues

- If app fails with missing packages, you are likely using system Python. Use:
  - `.\venv\Scripts\python.exe run_app.py`
- If camera opens but hand is not detected, ensure hand is centered, well lit, and fully visible.
- If FPS is low, lower camera resolution in `settings.json`.

## Main Files

- `run_app.py` - app entrypoint
- `accessibility_controller.py` - gesture + camera pipeline
- `accessibility_web.py` - web server and training APIs
- `templates/training_ui.html` - training interface
- `collect_gesture_data.py` - optional CLI sample collector