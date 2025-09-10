from flask import Flask, render_template, Response, jsonify, request, session
from accessibility_controller import AccessibilityController
import threading
import json
import cv2
import numpy as np
import time
import os
import logging
import re
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'accessibility-control-system-secret-key-2024')
controller = AccessibilityController()
current_gesture = "none"
is_running = False

# Global variable to store the controller thread
controller_thread = None

# Authentication decorator
def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_enabled = os.environ.get('REQUIRE_AUTH', 'false').lower() == 'true'
        if auth_enabled and not session.get('authenticated'):
            return jsonify({"status": "error", "message": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated

# Input validation functions
def validate_gesture_name(gesture):
    """Validate gesture name format"""
    if not isinstance(gesture, str):
        return False
    if gesture not in controller.gestures:
        return False
    if not re.match(r'^[a-zA-Z0-9_]+$', gesture):
        return False
    return True

def validate_settings(settings):
    """Validate settings input"""
    if not isinstance(settings, dict):
        return False
    
    allowed_keys = {
        'sensitivity', 'voice_feedback', 'cursor_speed', 'scroll_speed',
        'zoom_sensitivity', 'gesture_hold_time', 'double_click_speed',
        'gesture_confidence_threshold', 'voice_control', 'auto_calibration',
        'dark_mode', 'audio_feedback', 'haptic_feedback'
    }
    
    for key in settings:
        if key not in allowed_keys:
            return False
    
    return True

def validate_command(command):
    """Validate command input"""
    valid_commands = {
        'emergency_stop', 'toggle_cursor', 'toggle_voice', 'help'
    }
    return command in valid_commands

# Error handling decorator
def handle_errors(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}")
            return jsonify({"status": "error", "message": "Internal server error"}), 500
    return decorated

def run_controller():
    """Run the controller in a separate thread"""
    global current_gesture, is_running
    is_running = True
    try:
        controller.start_processing()
    except Exception as e:
        logger.error(f"Controller error: {e}")
        is_running = False
    finally:
        is_running = False

@app.route('/')
def index():
    """Main page route"""
    return render_template('accessibility_control.html')

@app.route('/training')
def training_ui():
    """Training UI route"""
    return render_template('training_ui.html')

@app.route('/login', methods=['POST'])
def login():
    """Handle user login"""
    if os.environ.get('REQUIRE_AUTH', 'false').lower() != 'true':
        return jsonify({"status": "success", "message": "Authentication not required"})
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        username = data.get('username')
        password = data.get('password')
        
        # Simple authentication - in production, use proper authentication
        expected_username = os.environ.get('AUTH_USERNAME', 'admin')
        expected_password = os.environ.get('AUTH_PASSWORD', 'password')
        
        if username == expected_username and password == expected_password:
            session['authenticated'] = True
            return jsonify({"status": "success", "message": "Login successful"})
        else:
            return jsonify({"status": "error", "message": "Invalid credentials"}), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"status": "error", "message": "Login failed"}), 500

@app.route('/logout', methods=['POST'])
def logout():
    """Handle user logout"""
    session.pop('authenticated', None)
    return jsonify({"status": "success", "message": "Logged out"})

@app.route('/video_feed')
@handle_errors
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            try:
                frame = controller.get_current_frame()
                if frame is None:
                    # Create a placeholder frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera not active", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, "Press Start to begin", (180, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Add status information
                status_text = f"Mode: {'Cursor' if controller.cursor_mode else 'Gesture'} | Voice: {'On' if controller.voice_enabled else 'Off'}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add current gesture
                gesture_text = f"Gesture: {controller.current_gesture}"
                cv2.putText(frame, gesture_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Add hand detection status
                if controller.current_results and controller.current_results.multi_hand_landmarks:
                    hands_text = f"Hands: {len(controller.current_results.multi_hand_landmarks)}"
                    cv2.putText(frame, hands_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                    
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
                       
            except Exception as e:
                logger.error(f"Video feed error: {e}")
                time.sleep(0.1)
            
            time.sleep(0.033)  # ~30 FPS

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/settings', methods=['GET', 'POST'])
@requires_auth
@handle_errors
def handle_settings():
    """Handle settings GET/POST requests"""
    if request.method == 'POST':
        try:
            new_settings = request.get_json()
            if not new_settings:
                return jsonify({"status": "error", "message": "No settings provided"}), 400
            
            if not validate_settings(new_settings):
                return jsonify({"status": "error", "message": "Invalid settings format"}), 400
            
            # Validate settings before applying
            validated_settings = controller.validate_settings(new_settings)
            controller.settings.update(validated_settings)
            controller.save_settings()
            return jsonify({"status": "success", "message": "Settings updated"})
            
        except Exception as e:
            logger.error(f"Settings update error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify(controller.settings)

@app.route('/gesture')
@handle_errors
def get_current_gesture():
    """Get current gesture and system status"""
    hand_detected = controller.current_results and controller.current_results.multi_hand_landmarks
    hand_count = len(controller.current_results.multi_hand_landmarks) if hand_detected else 0
    
    return jsonify({
        "gesture": controller.current_gesture,
        "cursor_mode": controller.cursor_mode,
        "voice_enabled": controller.voice_enabled,
        "running": is_running,
        "hand_detected": hand_detected,
        "hand_count": hand_count
    })

@app.route('/execute', methods=['POST'])
@requires_auth
@handle_errors
def execute_command():
    """Execute system commands"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
            
        command = data.get('command')
        if not command or not validate_command(command):
            return jsonify({"status": "error", "message": "Invalid command"}), 400

        if command == "emergency_stop":
            controller.emergency_stop()
            return jsonify({"status": "success", "message": "Emergency stop executed"})
            
        elif command == "toggle_cursor":
            controller.cursor_mode = not controller.cursor_mode
            mode = "enabled" if controller.cursor_mode else "disabled"
            return jsonify({"status": "success", "message": f"Cursor mode {mode}"})
            
        elif command == "toggle_voice":
            # Since voice recognition is disabled, just toggle feedback
            controller.settings["voice_feedback"] = not controller.settings.get("voice_feedback", True)
            controller.save_settings()
            feedback_status = "enabled" if controller.settings["voice_feedback"] else "disabled"
            return jsonify({"status": "success", "message": f"Voice feedback {feedback_status}"})
            
        elif command == "help":
            controller.show_help()
            return jsonify({"status": "success", "message": "Help displayed"})
            
        else:
            return jsonify({"status": "error", "message": f"Unknown command: {command}"}), 400
            
    except Exception as e:
        logger.error(f"Command execution error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/start', methods=['POST'])
@requires_auth
@handle_errors
def start_controller():
    """Start the accessibility controller"""
    global controller_thread, is_running
    
    try:
        if not is_running:
            controller_thread = threading.Thread(target=run_controller, daemon=True)
            controller_thread.start()
            # Wait a moment to ensure controller started
            time.sleep(0.5)
            if is_running:
                return jsonify({"status": "success", "message": "Controller started"})
            else:
                return jsonify({"status": "error", "message": "Controller failed to start"}), 500
        else:
            return jsonify({"status": "warning", "message": "Controller already running"})
    except Exception as e:
        logger.error(f"Controller start error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stop_training', methods=['POST'])
@requires_auth
@handle_errors
def stop_training():
    """Stop the training process"""
    try:
        controller.is_training = False
        controller.training_gesture = None
        return jsonify({"status": "success", "message": "Training stopped"})
    except Exception as e:
        logger.error(f"Stop training error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/status')
@handle_errors
def get_status():
    """Get system status"""
    try:
        # Ensure boolean, not object
        has_results = controller.current_results is not None
        hand_detected = bool(has_results and controller.current_results.multi_hand_landmarks)
        hand_count = len(controller.current_results.multi_hand_landmarks) if hand_detected else 0

        status_data = {
            "gesture": controller.current_gesture,
            "cursor_mode": controller.cursor_mode,
            "voice_enabled": controller.voice_enabled,
            "running": is_running,
            "hand_detected": hand_detected,   # ✅ now boolean
            "hand_count": hand_count,
            "settings": controller.settings
        }

        # Add landmarks if available
        if hand_detected:
            status_data["landmarks"] = []
            for hand_landmarks in controller.current_results.multi_hand_landmarks:
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    landmark_dict = {
                        'x': float(landmark.x),
                        'y': float(landmark.y),
                        'z': float(landmark.z)
                    }
                    if hasattr(landmark, 'visibility'):
                        landmark_dict['visibility'] = float(landmark.visibility)
                    hand_data.append(landmark_dict)
                status_data["landmarks"].append(hand_data)

        return jsonify(status_data)

    except Exception as e:
        logger.error(f"Status retrieval error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# Training API endpoints
@app.route('/start_training', methods=['POST'])
@requires_auth
@handle_errors
def start_training():
    """Start the training process for a specific gesture"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
            
        gesture = data.get('gesture')
        if not gesture or not validate_gesture_name(gesture):
            return jsonify({"status": "error", "message": "Invalid gesture name"}), 400
            
        success = controller.start_training(gesture)
        
        if success:
            return jsonify({"status": "success", "message": f"Started training for {gesture}"})
        else:
            return jsonify({"status": "error", "message": f"Failed to start training for {gesture}"}), 500
            
    except Exception as e:
        logger.error(f"Start training error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/capture_sample', methods=['POST'])
@requires_auth
@handle_errors
def capture_sample():
    """Capture a training sample for the current gesture"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
            
        gesture = data.get('gesture')
        if not gesture or not validate_gesture_name(gesture):
            return jsonify({"status": "error", "message": "Invalid gesture name"}), 400
            
        sample_count = controller.capture_training_sample(gesture)
        
        return jsonify({
            "status": "success", 
            "message": f"Sample captured for {gesture}",
            "sample_count": sample_count
        })
            
    except Exception as e:
        logger.error(f"Capture sample error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/train_model', methods=['POST'])
@requires_auth
@handle_errors
def train_model():
    """Train the model with collected samples"""
    try:
        result = controller.train_model()
        
        if result["status"] == "success":
            return jsonify({
                "status": "success", 
                "message": "Model trained successfully",
                "accuracy": result["accuracy"],
                "val_accuracy": result["val_accuracy"],
                "loss": result["loss"]
            })
        else:
            return jsonify({"status": "error", "message": result["message"]}), 400
            
    except Exception as e:
        logger.error(f"Train model error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/save_model', methods=['POST'])
@requires_auth
@handle_errors
def save_model():
    """Save the trained model to file"""
    try:
        success = controller.save_model()
        
        if success:
            return jsonify({"status": "success", "message": "Model saved successfully"})
        else:
            return jsonify({"status": "error", "message": "Failed to save model"}), 500
            
    except Exception as e:
        logger.error(f"Save model error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/training_status', methods=['GET'])
@handle_errors
def get_training_status():
    """Get training status and statistics"""
    try:
        status = controller.get_training_status()
        return jsonify({
            "status": "success",
            "current_gesture": status["current_gesture"],
            "samples_collected": status["samples_collected"],
            "total_samples": status["total_samples"],
            "is_training": status["is_training"],
            "gestures_trained": status["gestures_trained"],
            "format_version": status["format_version"]
        })
    except Exception as e:
        logger.error(f"Training status error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/reset_training', methods=['POST'])
@requires_auth
@handle_errors
def reset_training():
    """Reset training data for a specific gesture or all gestures"""
    try:
        data = request.get_json()
        gesture = data.get('gesture', None)  # If None, reset all
        
        if gesture and not validate_gesture_name(gesture):
            return jsonify({"status": "error", "message": "Invalid gesture name"}), 400
        
        success = controller.reset_training_data(gesture)
        
        if success:
            if gesture:
                message = f"Training data reset for {gesture}"
            else:
                message = "All training data reset"
                
            return jsonify({"status": "success", "message": message})
        else:
            return jsonify({"status": "error", "message": "Failed to reset training data"}), 500
            
    except Exception as e:
        logger.error(f"Reset training error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/export_training', methods=['POST'])
@requires_auth
@handle_errors
def export_training_data():
    """Export training data"""
    try:
        data = request.get_json()
        format = data.get('format', 'pkl') if data else 'pkl'
        
        if format not in ['pkl', 'json']:
            return jsonify({"status": "error", "message": "Invalid format. Use 'pkl' or 'json'"}), 400
        
        export_path = controller.export_training_data(format)
        
        if export_path:
            return jsonify({
                "status": "success", 
                "message": "Training data exported successfully",
                "export_path": export_path
            })
        else:
            return jsonify({"status": "error", "message": "Failed to export training data"}), 500
            
    except Exception as e:
        logger.error(f"Export training error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Calibration endpoints
@app.route('/calibrate', methods=['POST'])
@requires_auth
@handle_errors
def calibrate_gesture():
    """Calibrate a specific gesture"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
            
        gesture = data.get('gesture')
        if not gesture or not validate_gesture_name(gesture):
            return jsonify({"status": "error", "message": "Invalid gesture name"}), 400
            
        # This would typically capture calibration data
        # For now, just return success
        return jsonify({"status": "success", "message": f"Calibration started for {gesture}"})
            
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/calibrate_reset', methods=['POST'])
@requires_auth
@handle_errors
def reset_calibration():
    """Reset calibration data"""
    try:
        # Reset calibration to default values
        default_settings = {
            "sensitivity": 1.0,
            "cursor_speed": 2.0,
            "scroll_speed": 40,
            "gesture_confidence_threshold": 0.7,
            "gesture_hold_time": 1.0
        }
        
        controller.settings.update(default_settings)
        controller.save_settings()
        
        return jsonify({"status": "success", "message": "Calibration reset to default values"})
            
    except Exception as e:
        logger.error(f"Reset calibration error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/performance')
@handle_errors
def get_performance():
    """Get performance statistics"""
    try:
        stats = controller.get_performance_stats()
        return jsonify({
            "status": "success",
            "fps": stats["fps"],
            "frame_count": stats["frame_count"]
        })
    except Exception as e:
        logger.error(f"Performance stats error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({"status": "error", "message": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.errorhandler(401)
def unauthorized(error):
    """Handle 401 errors"""
    return jsonify({"status": "error", "message": "Authentication required"}), 401

if __name__ == '__main__':
    logger.info("Starting Accessibility Control Web Interface...")
    logger.info("Open http://localhost:5000 in your browser")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('training_data', exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)