# accessibility_web.py
from flask import Flask, render_template, Response, jsonify, request, session
from accessibility_controller import AccessibilityController
import threading
import json
import cv2
import numpy as np
import time
import os
from typing import Dict, Any, Optional, Tuple
from functools import wraps
import atexit

from logging_config import logger, log_exceptions
from config import config_manager, SystemConfig

app = Flask(__name__)
app.secret_key = config_manager.get_config().secret_key

app.config.update(
    DEBUG=False,
    ENV='production'
)

controller = AccessibilityController()
current_gesture = "none"
is_running = False
controller_thread: Optional[threading.Thread] = None

# Enhanced authentication decorator
def requires_auth(f):
    @wraps(f)
    @log_exceptions
    def decorated(*args, **kwargs):
        config = config_manager.get_config()
        if config.require_auth and not session.get('authenticated'):
            return jsonify({
                "status": "error", 
                "message": "Authentication required",
                "code": "AUTH_REQUIRED"
            }), 401
        
        # Validate session for authenticated users
        if session.get('authenticated') and not _validate_session():
            session.clear()
            return jsonify({
                "status": "error",
                "message": "Session expired",
                "code": "SESSION_EXPIRED"
            }), 401
            
        return f(*args, **kwargs)
    return decorated

def _validate_session() -> bool:
    """Validate session integrity"""
    try:
        # Add session validation logic here
        return session.get('authenticated', False)
    except Exception:
        return False

# Enhanced input validation
def validate_gesture_name(gesture: str) -> bool:
    """Validate gesture name format"""
    if not isinstance(gesture, str) or not gesture:
        return False
    if gesture not in controller.gestures:
        return False
    # Additional validation
    return all(c.isalnum() or c == '_' for c in gesture)

def validate_settings(settings: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate settings input with detailed error messages"""
    if not isinstance(settings, dict):
        return False, "Settings must be a dictionary"
    
    # Use config validation
    try:
        test_config = SystemConfig.from_dict(settings)
        if test_config.validate():
            return True, None
        else:
            return False, "Settings validation failed"
    except Exception as e:
        return False, f"Invalid settings format: {str(e)}"

def validate_command(command: str) -> Tuple[bool, Optional[str]]:
    """Validate command input"""
    valid_commands = {
        'emergency_stop', 'toggle_cursor', 'toggle_voice', 'help'
    }
    if command not in valid_commands:
        return False, f"Invalid command. Must be one of: {', '.join(valid_commands)}"
    return True, None

# Enhanced error handling
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 Not Found: {request.path}")
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "code": "NOT_FOUND"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    logger.warning(f"405 Method Not Allowed: {request.method} {request.path}")
    return jsonify({
        "status": "error",
        "message": "Method not allowed",
        "code": "METHOD_NOT_ALLOWED"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 Internal Server Error: {str(error)}")
    return jsonify({
        "status": "error",
        "message": "Internal server error",
        "code": "INTERNAL_ERROR"
    }), 500

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({
        "status": "error",
        "message": "Authentication required",
        "code": "UNAUTHORIZED"
    }), 401

@app.route('/')
@log_exceptions
def index():
    """Main page route"""
    return render_template('accessibility_control.html')

@app.route('/training')
@log_exceptions
def training_ui():
    """Training UI route"""
    return render_template('training_ui.html')

@app.route('/login', methods=['POST'])
@log_exceptions
def login():
    """Handle user login"""
    config = config_manager.get_config()
    
    if not config.require_auth:
        return jsonify({
            "status": "success", 
            "message": "Authentication not required"
        })
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error", 
                "message": "No data provided",
                "code": "NO_DATA"
            }), 400
        
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                "status": "error", 
                "message": "Username and password required",
                "code": "MISSING_CREDENTIALS"
            }), 400
        
        if username == config.auth_username and password == config.auth_password:
            session['authenticated'] = True
            session['login_time'] = time.time()
            logger.info(f"User {username} logged in successfully")
            return jsonify({
                "status": "success", 
                "message": "Login successful"
            })
        else:
            logger.warning(f"Failed login attempt for username: {username}")
            return jsonify({
                "status": "error", 
                "message": "Invalid credentials",
                "code": "INVALID_CREDENTIALS"
            }), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({
            "status": "error", 
            "message": "Login failed",
            "code": "LOGIN_ERROR"
        }), 500

@app.route('/logout', methods=['POST'])
@log_exceptions
def logout():
    """Handle user logout"""
    session.pop('authenticated', None)
    session.pop('login_time', None)
    logger.info("User logged out")
    return jsonify({
        "status": "success", 
        "message": "Logged out"
    })

@app.route('/video_feed')
@log_exceptions
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
                if controller.current_results and hasattr(controller.current_results, 'multi_hand_landmarks') and controller.current_results.multi_hand_landmarks:
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
@log_exceptions
def handle_settings():
    """Handle settings GET/POST requests"""
    if request.method == 'POST':
        try:
            new_settings = request.get_json()
            if not new_settings:
                return jsonify({
                    "status": "error",
                    "message": "No settings provided",
                    "code": "NO_SETTINGS"
                }), 400
            
            # Validate settings
            is_valid, error_msg = validate_settings(new_settings)
            if not is_valid:
                return jsonify({
                    "status": "error",
                    "message": error_msg,
                    "code": "INVALID_SETTINGS"
                }), 400
            
            # Update configuration
            if config_manager.update_config(new_settings):
                return jsonify({
                    "status": "success",
                    "message": "Settings updated successfully"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Failed to update settings",
                    "code": "UPDATE_FAILED"
                }), 500
                
        except Exception as e:
            logger.error(f"Settings update error: {e}")
            return jsonify({
                "status": "error",
                "message": "Internal server error",
                "code": "INTERNAL_ERROR"
            }), 500

    # GET request - return current settings
    try:
        return jsonify({
            "status": "success",
            "data": config_manager.get_config().to_dict()
        })
    except Exception as e:
        logger.error(f"Settings retrieval error: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve settings",
            "code": "RETRIEVAL_ERROR"
        }), 500

@app.route('/gesture')
@log_exceptions
def get_current_gesture():
    """Get current gesture and system status"""
    try:
        hand_detected = (
            controller.current_results and 
            hasattr(controller.current_results, 'multi_hand_landmarks') and 
            controller.current_results.multi_hand_landmarks
        )
        hand_count = len(controller.current_results.multi_hand_landmarks) if hand_detected else 0
        
        return jsonify({
            "gesture": controller.current_gesture,
            "cursor_mode": controller.cursor_mode,
            "voice_enabled": controller.voice_enabled,
            "running": is_running,
            "hand_detected": hand_detected,
            "hand_count": hand_count
        })
    except Exception as e:
        logger.error(f"Gesture status error: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to get gesture status",
            "code": "GESTURE_ERROR"
        }), 500

@app.route('/execute', methods=['POST'])
@requires_auth
@log_exceptions
def execute_command():
    """Execute system commands"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided",
                "code": "NO_DATA"
            }), 400
            
        command = data.get('command')
        if not command:
            return jsonify({
                "status": "error",
                "message": "No command provided",
                "code": "NO_COMMAND"
            }), 400

        # Validate command
        is_valid, error_msg = validate_command(command)
        if not is_valid:
            return jsonify({
                "status": "error",
                "message": error_msg,
                "code": "INVALID_COMMAND"
            }), 400

        if command == "emergency_stop":
            controller.emergency_stop()
            return jsonify({
                "status": "success", 
                "message": "Emergency stop executed"
            })
            
        elif command == "toggle_cursor":
            controller.cursor_mode = not controller.cursor_mode
            mode = "enabled" if controller.cursor_mode else "disabled"
            return jsonify({
                "status": "success", 
                "message": f"Cursor mode {mode}"
            })
            
        elif command == "toggle_voice":
            # Since voice recognition is disabled, just toggle feedback
            config = config_manager.get_config()
            new_voice_feedback = not config.voice_feedback
            config_manager.update_config({"voice_feedback": new_voice_feedback})
            feedback_status = "enabled" if new_voice_feedback else "disabled"
            return jsonify({
                "status": "success", 
                "message": f"Voice feedback {feedback_status}"
            })
            
        elif command == "help":
            controller.show_help()
            return jsonify({
                "status": "success", 
                "message": "Help displayed"
            })
            
        else:
            return jsonify({
                "status": "error",
                "message": f"Unknown command: {command}",
                "code": "UNKNOWN_COMMAND"
            }), 400
            
    except Exception as e:
        logger.error(f"Command execution error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route('/start', methods=['POST'])
@requires_auth
@log_exceptions
def start_controller():
    """Start the accessibility controller"""
    global controller_thread, is_running
    
    try:
        if not is_running:
            # Reinitialize controller if needed
            if hasattr(controller, 'hands') and (not controller.hands or controller.hands._graph is None):
                logger.info("Reinitializing MediaPipe for controller restart...")
                controller._initialize_mediapipe()
            
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

@app.route('/stop', methods=['POST'])
@requires_auth
@log_exceptions
def stop_controller():
    """Stop the accessibility controller and suggest restart"""
    global is_running
    
    try:
        if is_running:
            success = controller.stop_processing()
            is_running = not success
            
            if success:
                return jsonify({
                    "status": "success", 
                    "message": "Controller stopped. Please refresh the page to restart.",
                    "requires_restart": True
                })
            else:
                return jsonify({
                    "status": "error", 
                    "message": "Controller failed to stop"
                }), 500
        else:
            return jsonify({
                "status": "warning", 
                "message": "Controller not running"
            })
    except Exception as e:
        logger.error(f"Controller stop error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500

@app.route('/stop_training', methods=['POST'])
@requires_auth
@log_exceptions
def stop_training():
    """Stop the training process"""
    try:
        controller.is_training = False
        controller.training_gesture = None
        return jsonify({
            "status": "success", 
            "message": "Training stopped"
        })
    except Exception as e:
        logger.error(f"Stop training error: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to stop training",
            "code": "TRAINING_STOP_ERROR"
        }), 500

@app.route('/status')
@log_exceptions
def get_status():
    """Get system status with enhanced error handling"""
    try:
        status_data = controller.get_status()
        
        # Ensure the running status is included
        response_data = {
            "status": "success",
            "data": {
                **status_data,
                "running": is_running  # Make sure this global variable is included
            },
            "timestamp": time.time()
        }
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Status retrieval error: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve status",
            "code": "STATUS_ERROR",
            "running": is_running  # Fallback to global variable
        }), 500

# Training API endpoints
@app.route('/start_training', methods=['POST'])
@requires_auth
@log_exceptions
def start_training():
    """Start the training process for a specific gesture"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided",
                "code": "NO_DATA"
            }), 400
            
        gesture = data.get('gesture')
        if not gesture or not validate_gesture_name(gesture):
            return jsonify({
                "status": "error",
                "message": "Invalid gesture name",
                "code": "INVALID_GESTURE"
            }), 400
            
        success = controller.start_training(gesture)
        
        if success:
            return jsonify({
                "status": "success", 
                "message": f"Started training for {gesture}"
            })
        else:
            return jsonify({
                "status": "error", 
                "message": f"Failed to start training for {gesture}",
                "code": "TRAINING_START_ERROR"
            }), 500
            
    except Exception as e:
        logger.error(f"Start training error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route('/capture_sample', methods=['POST'])
@requires_auth
@log_exceptions
def capture_sample():
    """Capture a training sample for the current gesture"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided",
                "code": "NO_DATA"
            }), 400
            
        gesture = data.get('gesture')
        if not gesture or not validate_gesture_name(gesture):
            return jsonify({
                "status": "error",
                "message": "Invalid gesture name",
                "code": "INVALID_GESTURE"
            }), 400
            
        sample_count = controller.capture_training_sample(gesture)
        
        return jsonify({
            "status": "success", 
            "message": f"Sample captured for {gesture}",
            "sample_count": sample_count
        })
            
    except Exception as e:
        logger.error(f"Capture sample error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route('/train_model', methods=['POST'])
@requires_auth
@log_exceptions
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
            return jsonify({
                "status": "error", 
                "message": result["message"],
                "code": "TRAINING_ERROR"
            }), 400
            
    except Exception as e:
        logger.error(f"Train model error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route('/save_model', methods=['POST'])
@requires_auth
@log_exceptions
def save_model():
    """Save the trained model to file"""
    try:
        success = controller.save_model()
        
        if success:
            return jsonify({
                "status": "success", 
                "message": "Model saved successfully"
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Failed to save model",
                "code": "SAVE_ERROR"
            }), 500
            
    except Exception as e:
        logger.error(f"Save model error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route('/training_status', methods=['GET'])
@log_exceptions
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
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route('/reset_training', methods=['POST'])
@requires_auth
@log_exceptions
def reset_training():
    """Reset training data for a specific gesture or all gestures"""
    try:
        data = request.get_json()
        gesture = data.get('gesture', None)  # If None, reset all
        
        if gesture and not validate_gesture_name(gesture):
            return jsonify({
                "status": "error",
                "message": "Invalid gesture name",
                "code": "INVALID_GESTURE"
            }), 400
        
        success = controller.reset_training_data(gesture)
        
        if success:
            if gesture:
                message = f"Training data reset for {gesture}"
            else:
                message = "All training data reset"
                
            return jsonify({
                "status": "success", 
                "message": message
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Failed to reset training data",
                "code": "RESET_ERROR"
            }), 500
            
    except Exception as e:
        logger.error(f"Reset training error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route('/export_training', methods=['POST'])
@requires_auth
@log_exceptions
def export_training_data():
    """Export training data"""
    try:
        data = request.get_json()
        format = data.get('format', 'pkl') if data else 'pkl'
        
        if format not in ['pkl', 'json']:
            return jsonify({
                "status": "error",
                "message": "Invalid format. Use 'pkl' or 'json'",
                "code": "INVALID_FORMAT"
            }), 400
        
        export_path = controller.export_training_data(format)
        
        if export_path:
            return jsonify({
                "status": "success", 
                "message": "Training data exported successfully",
                "export_path": export_path
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Failed to export training data",
                "code": "EXPORT_ERROR"
            }), 500
            
    except Exception as e:
        logger.error(f"Export training error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "code": "INTERNAL_ERROR"
        }), 500

# Calibration endpoints
@app.route('/calibrate', methods=['POST'])
@requires_auth
@log_exceptions
def calibrate_gesture():
    """Calibrate a specific gesture"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided",
                "code": "NO_DATA"
            }), 400
            
        gesture = data.get('gesture')
        if not gesture or not validate_gesture_name(gesture):
            return jsonify({
                "status": "error",
                "message": "Invalid gesture name",
                "code": "INVALID_GESTURE"
            }), 400
            
        # This would typically capture calibration data
        # For now, just return success
        return jsonify({
            "status": "success", 
            "message": f"Calibration started for {gesture}"
        })
            
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route('/calibrate_reset', methods=['POST'])
@requires_auth
@log_exceptions
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
        
        config_manager.update_config(default_settings)
        
        return jsonify({
            "status": "success", 
            "message": "Calibration reset to default values"
        })
            
    except Exception as e:
        logger.error(f"Reset calibration error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route('/performance')
@log_exceptions
def get_performance():
    """Get performance statistics"""
    try:
        stats = controller.get_performance_stats()
        return jsonify({
            "status": "success",
            "fps": stats["fps"],
            "frame_count": stats["frame_count"],
            "memory_usage": stats.get("memory_usage", 0)
        })
    except Exception as e:
        logger.error(f"Performance stats error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "code": "INTERNAL_ERROR"
        }), 500

@app.route('/shutdown', methods=['POST'])
@requires_auth
@log_exceptions
def shutdown_server():
    """Graceful server shutdown"""
    try:
        logger.info("Received shutdown request")
        
        # Stop controller first
        global is_running
        if is_running:
            controller.stop_processing()
            is_running = False
        
        # Shutdown Flask
        def shutdown():
            time.sleep(1)
            os._exit(0)
        
        threading.Thread(target=shutdown).start()
        
        return jsonify({
            "status": "success",
            "message": "Server shutting down gracefully"
        })
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to shutdown server",
            "code": "SHUTDOWN_ERROR"
        }), 500

def run_controller():
    """Run the controller in a separate thread with proper error handling"""
    global is_running
    is_running = True
    
    try:
        # Use restart instead of start to ensure fresh MediaPipe instance
        if controller.restart_processing():
            logger.info("Controller started successfully")
        else:
            logger.error("Failed to start controller")
            is_running = False
    except Exception as e:
        logger.error(f"Controller startup error: {e}")
        is_running = False
    finally:
        if not is_running:
            controller.cleanup_resources()

# Add proper shutdown hook
@atexit.register
def cleanup_on_exit():
    """Cleanup resources on application exit"""
    logger.info("Application shutting down, cleaning up resources...")
    controller.stop_processing()
    controller.cleanup_resources()
    logger.info("Cleanup completed")

if __name__ == '__main__':
    try:
        logger.info("Starting Accessibility Control Web Interface...")
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('training_data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Validate configuration
        if not config_manager.get_config().validate():
            logger.warning("Configuration validation failed, using defaults")
        
        # Use updated Flask configuration
        app.run(
            host='0.0.0.0', 
            port=5000, 
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise