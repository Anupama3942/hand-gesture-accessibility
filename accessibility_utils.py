import json
import os
import logging
from datetime import datetime
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccessibilityLogger:
    def __init__(self):
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accessibility_log.json")
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'accessibility_system.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def log_event(self, event_type, details):
        """Log system events"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details
        }
        
        # Add to log file
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(event)
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
        
        # Also log to console and file
        logging.info(f"{event_type}: {details}")

class CalibrationSystem:
    def __init__(self):
        self.calibration_data = {}
        self.calibration_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_data.json")
        self.load_calibration()
    
    def load_calibration(self):
        """Load calibration data from file"""
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    self.calibration_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            self.calibration_data = {}
    
    def save_calibration(self):
        """Save calibration data to file"""
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
    
    def calibrate_gesture(self, gesture_name, landmarks_sequence):
        """Calibrate a specific gesture"""
        if gesture_name not in self.calibration_data:
            self.calibration_data[gesture_name] = []
        
        self.calibration_data[gesture_name].append({
            "timestamp": datetime.now().isoformat(),
            "landmarks": landmarks_sequence.tolist() if hasattr(landmarks_sequence, 'tolist') else landmarks_sequence
        })
        
        # Keep only last 5 calibrations per gesture
        if len(self.calibration_data[gesture_name]) > 5:
            self.calibration_data[gesture_name] = self.calibration_data[gesture_name][-5:]
        
        self.save_calibration()
    
    def get_calibration_stats(self, gesture_name):
        """Get calibration statistics for a gesture"""
        if gesture_name not in self.calibration_data:
            return None
        
        calibrations = self.calibration_data[gesture_name]
        if not calibrations:
            return None
        
        return {
            "count": len(calibrations),
            "last_calibration": calibrations[-1]["timestamp"]
        }

class PerformanceMonitor:
    def __init__(self):
        self.fps_history = []
        self.latency_history = []
        self.max_history_size = 100
    
    def update_fps(self, fps):
        """Update FPS monitoring"""
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_history_size:
            self.fps_history.pop(0)
    
    def update_latency(self, latency_ms):
        """Update latency monitoring"""
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > self.max_history_size:
            self.latency_history.pop(0)
    
    def get_stats(self):
        """Get performance statistics"""
        if not self.fps_history:
            return {"fps": 0, "latency_ms": 0}
        
        return {
            "fps": sum(self.fps_history) / len(self.fps_history),
            "latency_ms": sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0,
            "fps_min": min(self.fps_history),
            "fps_max": max(self.fps_history)
        }

class AccessibilityPresets:
    def __init__(self):
        self.presets_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accessibility_presets.json")
        self.presets = self.load_presets()
    
    def load_presets(self):
        """Load accessibility presets"""
        default_presets = {
            "default": {
                "sensitivity": 1.0,
                "cursor_speed": 2.0,
                "voice_feedback": True
            },
            "high_precision": {
                "sensitivity": 0.7,
                "cursor_speed": 1.5,
                "voice_feedback": True
            },
            "gaming": {
                "sensitivity": 1.3,
                "cursor_speed": 3.0,
                "voice_feedback": False
            },
            "presentation": {
                "sensitivity": 1.1,
                "cursor_speed": 2.5,
                "voice_feedback": True
            }
        }
        
        try:
            if os.path.exists(self.presets_file):
                with open(self.presets_file, 'r') as f:
                    loaded = json.load(f)
                    return {**default_presets, **loaded}
        except Exception as e:
            logger.error(f"Error loading presets: {e}")
        
        return default_presets
    
    def save_presets(self):
        """Save presets to file"""
        try:
            with open(self.presets_file, 'w') as f:
                json.dump(self.presets, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save presets: {e}")
    
    def apply_preset(self, preset_name, settings):
        """Apply a preset to settings"""
        if preset_name in self.presets:
            settings.update(self.presets[ preset_name])
            return True
        return False
    
    def create_preset(self, name, settings):
        """Create a new preset"""
        self.presets[name] = settings
        self.save_presets()
    
    def delete_preset(self, name):
        """Delete a preset"""
        if name in self.presets and name not in ["default", "high_precision", "gaming", "presentation"]:
            del self.presets[name]
            self.save_presets()
            return True
        return False