# config.py
import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from logging_config import logger

@dataclass
class SystemConfig:
    # General settings
    sensitivity: float = 1.0
    voice_feedback: bool = True
    cursor_speed: float = 2.0
    scroll_speed: int = 40
    zoom_sensitivity: float = 1.2
    gesture_hold_time: float = 1.0
    double_click_speed: float = 0.3
    gesture_confidence_threshold: float = 0.7
    training_samples: int = 30
    
    # Security settings
    require_auth: bool = False
    auth_username: str = "admin"
    auth_password: str = "password"
    secret_key: str = "default-secret-key-change-in-production"
    
    # Performance settings
    camera_width: int = 1280
    camera_height: int = 720
    target_fps: int = 30
    max_history_size: int = 100
    
    # Path settings
    model_path: str = "models/accessibility_gesture_model.h5"
    training_data_path: str = "training_data/training_samples.pkl"
    settings_path: str = "settings.json"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate configuration values"""
        validations = [
            (0.1 <= self.sensitivity <= 3.0, "Sensitivity must be between 0.1 and 3.0"),
            (0.5 <= self.cursor_speed <= 5.0, "Cursor speed must be between 0.5 and 5.0"),
            (10 <= self.scroll_speed <= 100, "Scroll speed must be between 10 and 100"),
            (0.5 <= self.gesture_hold_time <= 3.0, "Gesture hold time must be between 0.5 and 3.0 seconds"),
            (0.3 <= self.gesture_confidence_threshold <= 0.95, "Confidence threshold must be between 0.3 and 0.95"),
            (self.training_samples >= 5, "At least 5 training samples required")
        ]
        
        for condition, message in validations:
            if not condition:
                logger.warning(f"Configuration validation failed: {message}")
                return False
        return True

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.config = SystemConfig()
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config.settings_path):
                with open(self.config.settings_path, 'r') as f:
                    data = json.load(f)
                    self.config = SystemConfig.from_dict(data)
                
                if self.config.validate():
                    logger.info("Configuration loaded successfully")
                    return True
                else:
                    logger.warning("Configuration validation failed, using defaults")
                    self.config = SystemConfig()
            return False
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config.settings_path), exist_ok=True)
            with open(self.config.settings_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_config(self) -> SystemConfig:
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update configuration with validation"""
        try:
            updated_config = SystemConfig.from_dict({**self.config.to_dict(), **new_config})
            if updated_config.validate():
                self.config = updated_config
                return self.save_config()
            return False
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False

# Global config instance
config_manager = ConfigManager()