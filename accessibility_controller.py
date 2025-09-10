import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyautogui
import pyttsx3
import time
import threading
from pynput import keyboard, mouse
from pynput.keyboard import Controller as KeyController, Key
from pynput.mouse import Controller as MouseController, Button
import os
import json
import platform
import subprocess
from datetime import datetime
import pickle
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccessibilityController:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Gesture recognition model
        self.gestures = np.array([
            'none', 'cursor_mode', 'click', 'right_click', 'double_click', 'drag',
            'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out', 'back',
            'forward', 'volume_up', 'volume_down', 'mute', 'play_pause',
            'next_track', 'prev_track', 'copy', 'paste', 'cut',
            'undo', 'redo', 'save', 'new_tab', 'close_tab',
            'switch_app', 'desktop', 'task_view', 'help'
        ])
        
        self.model = None
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'accessibility_gesture_model.h5')
        self.initialize_model()
        
        # Control state
        self.current_gesture = "none"
        self.previous_gesture = "none"
        self.gesture_start_time = 0
        self.gesture_hold_threshold = 1.0  # seconds
        
        # Processing state
        self.running = False
        self.current_frame = None
        self.current_results = None
        self.cap = None
        
        # Mouse control
        self.mouse_controller = MouseController()
        self.key_controller = KeyController()
        self.cursor_mode = False
        self.dragging = False
        
        # Voice feedback (text-to-speech only) - optional
        self.voice_engine = None
        self.voice_enabled = False
        try:
            self.voice_engine = pyttsx3.init()
            self.voice_engine.setProperty('rate', 150)
            self.voice_engine.setProperty('volume', 0.8)
            self.voice_enabled = True
            logger.info("Voice engine initialized successfully")
        except Exception as e:
            logger.warning(f"Voice engine initialization failed (optional feature): {e}")
            self.voice_engine = None
            self.voice_enabled = False
        
        # User settings
        self.settings = self.load_settings()
        
        # Gesture sequence for complex commands
        self.gesture_sequence = []
        self.sequence_timeout = 2.0  # seconds
        self.last_gesture_time = time.time()
        
        # Media control state
        self.media_keys = {
            'play_pause': Key.media_play_pause,
            'next_track': Key.media_next,
            'prev_track': Key.media_previous,
            'volume_up': Key.media_volume_up,
            'volume_down': Key.media_volume_down,
            'mute': Key.media_volume_mute
        }
        
        # Browser control keys
        self.browser_keys = {
            'back': ('browser_back', 'chrome_back'),
            'forward': ('browser_forward', 'chrome_forward'),
            'new_tab': (Key.ctrl, 't'),
            'close_tab': (Key.ctrl, 'w')
        }
        
        # Training state
        self.is_training = False
        self.training_gesture = None
        self.training_data = {}
        self.training_samples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data', 'training_samples.pkl')
        self.load_training_data()
        
        # Performance monitoring
        self.performance_stats = {
            'fps': 0,
            'last_update': time.time(),
            'frame_count': 0
        }
        
    def get_status_for_json(self):
        """Get status data that can be serialized to JSON"""
        hand_detected = self.current_results and self.current_results.multi_hand_landmarks
        hand_count = len(self.current_results.multi_hand_landmarks) if hand_detected else 0
        
        # Convert any numpy arrays to lists for JSON serialization
        settings_serializable = {}
        for key, value in self.settings.items():
            if isinstance(value, (np.integer, np.floating)):
                settings_serializable[key] = float(value)
            elif isinstance(value, np.ndarray):
                settings_serializable[key] = value.tolist()
            else:
                settings_serializable[key] = value
        
        return {
            "running": self.running,
            "cursor_mode": self.cursor_mode,
            "voice_enabled": self.voice_enabled,
            "current_gesture": self.current_gesture,
            "hand_detected": hand_detected,
            "hand_count": hand_count,
            "settings": settings_serializable
        }

    # Standardized Training Data Methods
    def load_training_data(self):
        """Load training data from file using standardized format"""
        training_dir = os.path.dirname(self.training_samples_path)
        os.makedirs(training_dir, exist_ok=True)
        
        if os.path.exists(self.training_samples_path):
            try:
                with open(self.training_samples_path, 'rb') as f:
                    data = pickle.load(f)
                    # Convert old format to new standardized format if needed
                    if isinstance(data, dict) and all(isinstance(samples, list) for samples in data.values()):
                        self.training_data = data
                        logger.info(f"Loaded {sum(len(samples) for samples in self.training_data.values())} training samples")
                    else:
                        logger.warning("Training data format is invalid, initializing empty data")
                        self.training_data = {}
            except Exception as e:
                logger.error(f"Error loading training data: {e}")
                # Initialize empty training data instead of crashing
                self.training_data = {}
        else:
            self.training_data = {}
            
    def save_training_data(self):
        """Save training data to file using standardized format"""
        try:
            training_dir = os.path.dirname(self.training_samples_path)
            os.makedirs(training_dir, exist_ok=True)
            
            # Ensure data is in correct format before saving
            validated_data = {}
            for gesture, samples in self.training_data.items():
                if gesture in self.gestures and isinstance(samples, list):
                    validated_data[gesture] = samples
            
            with open(self.training_samples_path, 'wb') as f:
                pickle.dump(validated_data, f)
            logger.info("Training data saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
            return False
    
    def validate_training_sample(self, keypoints):
        """Validate a training sample meets requirements"""
        if not isinstance(keypoints, np.ndarray):
            return False
        if keypoints.shape != (63,):  # 21 landmarks * 3 coordinates
            return False
        if np.all(keypoints == 0) or not np.any(keypoints):
            return False
        if np.any(np.isnan(keypoints)) or np.any(np.isinf(keypoints)):
            return False
        return True
    
    def start_training(self, gesture):
        """Start training mode for a specific gesture"""
        if not self.validate_gesture_name(gesture):
            logger.warning(f"Invalid gesture name: {gesture}")
            return False

        self.is_training = True
        self.training_gesture = gesture

        # Initialize training data for this gesture if it doesn't exist
        if gesture not in self.training_data:
            self.training_data[gesture] = []

        logger.info(f"Started training for gesture: {gesture}")
        self.speak(f"Training {gesture.replace('_', ' ')}")
        return True
    
    def capture_training_sample(self, gesture):
        """Capture a training sample for the current gesture"""
        if not self.is_training or gesture != self.training_gesture:
            logger.warning("Not in training mode or gesture mismatch")
            return 0
        
        if self.current_results and self.current_results.multi_hand_landmarks:
            try:
                # Extract keypoints from the current frame
                keypoints = self.extract_keypoints(self.current_results)
                
                # Validate sample
                if self.validate_training_sample(keypoints):
                    # Add to training data with standardized format
                    sample = {
                        'keypoints': keypoints,
                        'timestamp': datetime.now().isoformat(),
                        'gesture': gesture,
                        'version': '1.0'  # Standardized format version
                    }
                    
                    self.training_data[gesture].append(sample)
                    
                    # Save training data
                    self.save_training_data()
                    
                    sample_count = len(self.training_data[gesture])
                    logger.info(f"Captured sample {sample_count} for {gesture}")
                    
                    return sample_count
                else:
                    logger.warning("Invalid sample (validation failed)")
                    return 0
                
            except Exception as e:
                logger.error(f"Error capturing training sample: {e}")
                return 0
        else:
            logger.warning("No hand landmarks detected")
            return 0
    
    def train_model(self):
        """Train the model with collected samples using standardized format"""
        try:
            # Check if we have enough training data
            total_samples = sum(len(samples) for samples in self.training_data.values())
            if total_samples < 20:
                return {
                    "status": "error",
                    "message": f"Not enough training data. Need at least 20 samples, have {total_samples}"
                }
            
            # Prepare training data using standardized format
            X_train = []
            y_train = []
            
            for i, gesture in enumerate(self.gestures):
                if gesture in self.training_data and self.training_data[gesture]:
                    for sample in self.training_data[gesture]:
                        # Ensure we have valid samples in standardized format
                        if (self.validate_training_sample(sample['keypoints']) and 
                            sample.get('gesture') == gesture):
                            X_train.append(sample['keypoints'])
                            y_train.append(i)
            
            if not X_train:
                return {
                    "status": "error",
                    "message": "No valid training data available"
                }
            
            # Convert to numpy arrays
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Convert labels to categorical
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(self.gestures))
            
            # Split data (80% train, 20% validation)
            split_idx = int(0.8 * len(X_train))
            X_val, y_val = X_train[split_idx:], y_train[split_idx:]
            X_train, y_train = X_train[:split_idx], y_train[:split_idx]
            
            logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
            
            # Create a new model if none exists
            if self.model is None:
                self.model = self.create_model()
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
                ]
            )
            
            # Get training results
            final_accuracy = history.history['categorical_accuracy'][-1]
            final_val_accuracy = history.history['val_categorical_accuracy'][-1]
            final_loss = history.history['loss'][-1]
            
            logger.info(f"Training completed. Accuracy: {final_accuracy:.2%}, Validation Accuracy: {final_val_accuracy:.2%}")
            
            return {
                "status": "success",
                "accuracy": float(final_accuracy),
                "val_accuracy": float(final_val_accuracy),
                "loss": float(final_loss)
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def save_model(self):
        """Save the trained model to file"""
        try:
            if self.model is not None:
                model_dir = os.path.dirname(self.model_path)
                os.makedirs(model_dir, exist_ok=True)
                self.model.save(self.model_path)
                logger.info("Model saved successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def validate_gesture_name(self, gesture):
        """Validate gesture name format"""
        if not isinstance(gesture, str):
            return False
        if gesture not in self.gestures:
            return False
        # Allow only alphanumeric and underscore characters
        if not re.match(r'^[a-zA-Z0-9_]+$', gesture):
            return False
        return True
    
    def get_training_status(self):
        """Get training status and statistics with standardized format"""
        total_samples = sum(len(samples) for samples in self.training_data.values())
        current_samples = len(self.training_data[self.training_gesture]) if self.training_gesture in self.training_data else 0
        
        return {
            "current_gesture": self.training_gesture,
            "samples_collected": current_samples,
            "total_samples": total_samples,
            "is_training": self.is_training,
            "gestures_trained": [g for g in self.gestures if g in self.training_data and self.training_data[g]],
            "format_version": "1.0"
        }
    
    def reset_training_data(self, gesture=None):
        """Reset training data for a specific gesture or all gestures"""
        try:
            if gesture:
                if self.validate_gesture_name(gesture) and gesture in self.training_data:
                    self.training_data[gesture] = []
                    logger.info(f"Reset training data for {gesture}")
                else:
                    return False
            else:
                self.training_data = {}
                logger.info("Reset all training data")
            
            # Save the changes
            return self.save_training_data()
            
        except Exception as e:
            logger.error(f"Error resetting training data: {e}")
            return False
    
    def export_training_data(self, format='pkl'):
        """Export training data in standardized format"""
        try:
            export_data = {
                'metadata': {
                    'export_date': datetime.now().isoformat(),
                    'total_samples': sum(len(samples) for samples in self.training_data.values()),
                    'format_version': '1.0',
                    'gestures': list(self.training_data.keys())
                },
                'samples': self.training_data
            }
            
            if format == 'pkl':
                export_path = os.path.join(os.path.dirname(self.training_samples_path), 'training_export.pkl')
                with open(export_path, 'wb') as f:
                    pickle.dump(export_data, f)
            elif format == 'json':
                export_path = os.path.join(os.path.dirname(self.training_samples_path), 'training_export.json')
                # Convert numpy arrays to lists for JSON serialization
                json_data = export_data.copy()
                for gesture, samples in json_data['samples'].items():
                    for i, sample in enumerate(samples):
                        if isinstance(sample['keypoints'], np.ndarray):
                            json_data['samples'][gesture][i]['keypoints'] = sample['keypoints'].tolist()
                
                with open(export_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Training data exported to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            return None
    
    # Existing methods remain the same but with improved error handling
    def initialize_model(self):
        """Load or create the gesture recognition model"""
        model_dir = os.path.dirname(self.model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        if os.path.exists(self.model_path):
            try:
                logger.info("Loading existing model...")
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.info("Creating new model...")
                self.model = self.create_model()
                self.create_dummy_training_data()
        else:
            logger.info("Creating new accessibility model...")
            self.model = self.create_model()
            self.create_dummy_training_data()
    
    def create_model(self):
        """Create a new model for gesture recognition"""
        try:
            # Simple Dense model for single-frame gesture recognition
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(len(self.gestures), activation='softmax')
            ])
            
            model.compile(optimizer='adam', 
                         loss='categorical_crossentropy', 
                         metrics=['categorical_accuracy'])
            return model
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return None
    
    def create_dummy_training_data(self):
        """Create minimal training data so model can function"""
        if self.model is None:
            return
            
        try:
            # Create dummy data for each gesture
            dummy_data = []
            dummy_labels = []
            
            for i, gesture in enumerate(self.gestures):
                # Create 5 dummy samples per gesture
                for _ in range(5):
                    # Create realistic hand landmark data
                    if gesture == "none":
                        # Random noise for "none" gesture
                        dummy_sample = np.random.normal(0.5, 0.2, 63)
                    else:
                        # Create more structured data for actual gestures
                        dummy_sample = np.random.normal(0.5, 0.1, 63)
                    
                    dummy_data.append(dummy_sample)
                    dummy_labels.append(i)
            
            X_train = np.array(dummy_data)
            y_train = tf.keras.utils.to_categorical(dummy_labels, num_classes=len(self.gestures))
            
            # Quick training with dummy data
            logger.info("Training model with dummy data...")
            self.model.fit(X_train, y_train, epochs=10, verbose=0)
            self.model.save(self.model_path)
            logger.info("Model initialized with dummy data")
        except Exception as e:
            logger.error(f"Error creating dummy training data: {e}")
    
    def load_settings(self):
        """Load user settings from file"""
        default_settings = {
            "sensitivity": 1.0,
            "voice_feedback": True,
            "cursor_speed": 2.0,
            "scroll_speed": 40,
            "zoom_sensitivity": 1.2,
            "gesture_hold_time": 1.0,
            "double_click_speed": 0.3,
            "gesture_confidence_threshold": 0.7,
            "voice_control": False,
            "auto_calibration": True,
            "dark_mode": False,
            "audio_feedback": True,
            "haptic_feedback": False
        }
        
        try:
            settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    loaded_settings = json.load(f)
                    return self.validate_settings({**default_settings, **loaded_settings})
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
        
        return default_settings
    
    def validate_settings(self, settings):
        """Validate settings before applying"""
        validated = {}
        for key, value in settings.items():
            if key == "sensitivity":
                validated[key] = max(0.1, min(3.0, float(value)))
            elif key == "cursor_speed":
                validated[key] = max(0.5, min(5.0, float(value)))
            elif key == "scroll_speed":
                validated[key] = max(10, min(100, int(value)))
            elif key == "gesture_confidence_threshold":
                validated[key] = max(0.3, min(0.95, float(value)))
            elif key == "gesture_hold_time":
                validated[key] = max(0.5, min(3.0, float(value)))
            elif key in ["voice_feedback", "voice_control", "auto_calibration", 
                        "dark_mode", "audio_feedback", "haptic_feedback"]:
                validated[key] = bool(value)
            else:
                validated[key] = value
        
        return validated
    
    def save_settings(self):
        """Save user settings to file"""
        try:
            settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')
            with open(settings_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
            logger.info("Settings saved successfully")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
    
    def speak(self, text):
        """Provide voice feedback (text-to-speech only)"""
        if self.voice_enabled and self.settings.get("voice_feedback", True) and self.voice_engine:
            try:
                # Run TTS in separate thread to avoid blocking
                def speak_async():
                    try:
                        self.voice_engine.say(text)
                        self.voice_engine.runAndWait()
                    except Exception as e:
                        logger.error(f"Speech error: {e}")
                
                threading.Thread(target=speak_async, daemon=True).start()
            except Exception as e:
                logger.error(f"Speech setup error: {e}")
    
    def mediapipe_detection(self, image):
        """Process image with MediaPipe Hands"""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.hands.process(image_rgb)
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            return image_bgr, results
        except Exception as e:
            logger.error(f"MediaPipe detection error: {e}")
            return image, None
    
    def extract_keypoints(self, results):
        """Extract hand landmarks from MediaPipe results"""
        if results and results.multi_hand_landmarks:
            try:
                # Use the first hand detected
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                return np.array(landmarks)
            except Exception as e:
                logger.error(f"Keypoint extraction error: {e}")
                return np.zeros(21*3)
        return np.zeros(21*3)
    
    def draw_landmarks(self, image, results):
        """Draw hand landmarks on the image"""
        if results and results.multi_hand_landmarks:
            try:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                    )
            except Exception as e:
                logger.error(f"Landmark drawing error: {e}")
        return image
    
    def recognize_gesture(self, landmarks):
        """Recognize gesture from landmarks with better validation"""
        if (self.model is None or landmarks is None or 
            len(landmarks) != 63 or np.all(landmarks == 0)):
            return "none", 0.0
        
        try:
            # Reshape for model prediction
            landmarks = np.expand_dims(landmarks, axis=0)
            
            # Predict gesture
            prediction = self.model.predict(landmarks, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            
            # Only return if confidence is above threshold
            if confidence > self.settings.get("gesture_confidence_threshold", 0.5):
                return self.gestures[predicted_class], confidence
            else:
                return "none", confidence
                
        except Exception as e:
            logger.error(f"Gesture recognition error: {e}")
            return "none", 0.0
    
    def control_cursor(self, hand_landmarks):
        """Control mouse cursor with hand position"""
        if not self.cursor_mode or hand_landmarks is None:
            return
        
        try:
            # Get index finger tip position (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            x = int(index_finger_tip.x * self.screen_width)
            y = int(index_finger_tip.y * self.screen_height)
            
            # Move cursor with smoothing
            current_x, current_y = self.mouse_controller.position
            new_x = current_x + (x - current_x) * 0.3 * self.settings["cursor_speed"]
            new_y = current_y + (y - current_y) * 0.3 * self.settings["cursor_speed"]
            
            self.mouse_controller.position = (new_x, new_y)
        except Exception as e:
            logger.error(f"Cursor control error: {e}")
    
    def safe_execute_action(self, action_func, action_name="action"):
        """Safely execute actions with error handling"""
        try:
            action_func()
        except Exception as e:
            logger.error(f"{action_name} execution error: {e}")
            self.speak("Action failed")
    
    def execute_gesture_action(self, gesture, hand_landmarks=None, confidence=0.0):
        """Execute computer control actions based on recognized gesture"""
        if confidence < self.settings.get("gesture_confidence_threshold", 0.7):
            return
        
        current_time = time.time()
        gesture_duration = current_time - self.gesture_start_time
        
        # Handle new gesture
        if gesture != self.previous_gesture:
            self.gesture_start_time = current_time
            self.previous_gesture = gesture
            if gesture != "none":
                self.speak(gesture.replace('_', ' '))
        
        # Execute actions based on gesture
        action_map = {
            "cursor_mode": lambda: self.toggle_cursor_mode(gesture_duration),
            "click": lambda: self.safe_execute_action(
                lambda: self.mouse_controller.click(Button.left) if self.cursor_mode else None, "click"),
            "right_click": lambda: self.safe_execute_action(
                lambda: self.mouse_controller.click(Button.right) if self.cursor_mode else None, "right_click"),
            "double_click": lambda: self.safe_execute_action(
                lambda: self.mouse_controller.click(Button.left, 2) if self.cursor_mode else None, "double_click"),
            "drag": lambda: self.handle_drag(hand_landmarks),
            "scroll_up": lambda: self.safe_execute_action(
                lambda: pyautogui.scroll(self.settings["scroll_speed"]), "scroll_up"),
            "scroll_down": lambda: self.safe_execute_action(
                lambda: pyautogui.scroll(-self.settings["scroll_speed"]), "scroll_down"),
            "zoom_in": lambda: self.safe_execute_action(
                lambda: self.execute_key_combo(Key.ctrl, Key.plus), "zoom_in"),
            "zoom_out": lambda: self.safe_execute_action(
                lambda: self.execute_key_combo(Key.ctrl, Key.minus), "zoom_out"),
            "back": lambda: self.safe_execute_action(
                lambda: self.execute_browser_command('back'), "back"),
            "forward": lambda: self.safe_execute_action(
                lambda: self.execute_browser_command('forward'), "forward"),
            "volume_up": lambda: self.safe_execute_action(
                lambda: self.key_controller.press(Key.media_volume_up), "volume_up"),
            "volume_down": lambda: self.safe_execute_action(
                lambda: self.key_controller.press(Key.media_volume_down), "volume_down"),
            "mute": lambda: self.safe_execute_action(
                lambda: self.key_controller.press(Key.media_volume_mute), "mute"),
            "play_pause": lambda: self.safe_execute_action(
                lambda: self.key_controller.press(Key.media_play_pause), "play_pause"),
            "next_track": lambda: self.safe_execute_action(
                lambda: self.key_controller.press(Key.media_next), "next_track"),
            "prev_track": lambda: self.safe_execute_action(
                lambda: self.key_controller.press(Key.media_previous), "prev_track"),
            "copy": lambda: self.safe_execute_action(
                lambda: self.execute_key_combo(Key.ctrl, 'c'), "copy"),
            "paste": lambda: self.safe_execute_action(
                lambda: self.execute_key_combo(Key.ctrl, 'v'), "paste"),
            "cut": lambda: self.safe_execute_action(
                lambda: self.execute_key_combo(Key.ctrl, 'x'), "cut"),
            "undo": lambda: self.safe_execute_action(
                lambda: self.execute_key_combo(Key.ctrl, 'z'), "undo"),
            "redo": lambda: self.safe_execute_action(
                lambda: self.execute_key_combo(Key.ctrl, 'y'), "redo"),
            "save": lambda: self.safe_execute_action(
                lambda: self.execute_key_combo(Key.ctrl, 's'), "save"),
            "new_tab": lambda: self.safe_execute_action(
                lambda: self.execute_key_combo(Key.ctrl, 't'), "new_tab"),
            "close_tab": lambda: self.safe_execute_action(
                lambda: self.execute_key_combo(Key.ctrl, 'w'), "close_tab"),
            "switch_app": lambda: self.safe_execute_action(
                lambda: self.execute_key_combo(Key.alt, Key.tab), "switch_app"),
            "desktop": lambda: self.safe_execute_action(
                lambda: self.execute_key_combo(None, Key.cmd if platform.system() == "Darwin" else Key.cmd_r), "desktop"),
            "task_view": lambda: self.safe_execute_action(
                lambda: self.execute_key_combo(Key.cmd, Key.tab) if platform.system() == "Darwin" else 
                       self.execute_key_combo(Key.cmd, Key.tab), "task_view"),
            "help": lambda: self.show_help()
        }
        
        if gesture in action_map:
            action_map[gesture]()
    
    def execute_browser_command(self, command):
        """Execute browser-specific commands with fallbacks"""
        try:
            # Try browser-specific keys first
            if command in self.browser_keys:
                key_combo = self.browser_keys[command]
                if isinstance(key_combo, tuple):
                    if len(key_combo) == 2:
                        self.execute_key_combo(key_combo[0], key_combo[1])
                    else:
                        self.key_controller.press(key_combo[0])
                else:
                    self.key_controller.press(key_combo)
            else:
                # Fallback to standard keyboard shortcuts
                if command == 'back':
                    self.execute_key_combo(Key.alt, Key.left)
                elif command == 'forward':
                    self.execute_key_combo(Key.alt, Key.right)
        except Exception as e:
            logger.error(f"Browser command error: {e}")
    
    def toggle_cursor_mode(self, duration):
        """Toggle cursor mode with hold threshold"""
        if duration > self.gesture_hold_threshold:
            self.cursor_mode = not self.cursor_mode
            mode = "enabled" if self.cursor_mode else "disabled"
            self.speak(f"Cursor mode {mode}")
            self.gesture_start_time = time.time()  # Reset timer
    
    def handle_drag(self, hand_landmarks):
        """Handle drag gesture"""
        if not self.cursor_mode or not hand_landmarks:
            return
            
        try:
            if not self.dragging:
                self.mouse_controller.press(Button.left)
                self.dragging = True
                self.speak("Drag started")
            else:
                # Continue dragging by moving the cursor
                self.control_cursor(hand_landmarks)
        except Exception as e:
            logger.error(f"Drag error: {e}")
    
    def execute_key_combo(self, modifier, key):
        """Execute keyboard combinations safely"""
        try:
            if modifier:
                with self.key_controller.pressed(modifier):
                    if isinstance(key, str):
                        self.key_controller.press(key)
                        self.key_controller.release(key)
                    else:
                        self.key_controller.press(key)
                        self.key_controller.release(key)
            else:
                if isinstance(key, str):
                    self.key_controller.press(key)
                    self.key_controller.release(key)
                else:
                    self.key_controller.press(key)
                    self.key_controller.release(key)
        except Exception as e:
            logger.error(f"Key combo error: {e}")
    
    def emergency_stop(self):
        """Emergency stop all actions"""
        try:
            if self.dragging:
                self.mouse_controller.release(Button.left)
                self.dragging = False
            
            self.cursor_mode = False
            # Cancel any ongoing actions
            self.current_gesture = "none"
            self.previous_gesture = "none"
            self.speak("Emergency stop activated")
        except Exception as e:
            logger.error(f"Emergency stop error: {e}")
    
    def show_help(self):
        """Show available gestures and commands"""
        help_text = "Available gestures: Cursor mode, Click, Right click, Drag, Scroll, Copy, Paste, Cut, Undo, Redo, Save, New tab, Close tab, Switch app, Desktop, Task view, Help, Media controls, Volume controls, Zoom controls"
        self.speak(help_text)
    
    def start_processing(self):
        """Start the main processing loop"""
        self.running = True
        
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            if not self.cap.isOpened():
                logger.error("Could not open camera")
                return
            
            logger.info("Starting Accessibility Controller processing...")
            self.speak("Accessibility controller started")
            
            # Performance monitoring
            start_time = time.time()
            frame_count = 0
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                frame = cv2.flip(frame, 1)
                image, results = self.mediapipe_detection(frame)
                
                # Store current frame and results for web interface
                self.current_frame = self.draw_landmarks(image.copy(), results)
                self.current_results = results
                
                # Extract keypoints and recognize gesture
                keypoints = self.extract_keypoints(results)
                
                # Control cursor if in cursor mode and hand detected
                if results and results.multi_hand_landmarks and self.cursor_mode:
                    self.control_cursor(results.multi_hand_landmarks[0])
                
                # Release drag if no hand detected
                if self.dragging and (not results or not results.multi_hand_landmarks):
                    try:
                        self.mouse_controller.release(Button.left)
                        self.dragging = False
                    except Exception as e:
                        logger.error(f"Drag release error: {e}")
                
                # Recognize gesture and execute action
                if self.model is not None and keypoints is not None:
                    gesture, confidence = self.recognize_gesture(keypoints)
                    self.current_gesture = gesture
                    
                    # Show debug info in console
                    if gesture != "none" and confidence > 0.5:
                        logger.info(f"Detected: {gesture} (confidence: {confidence:.2f})")
                    
                    self.execute_gesture_action(gesture, 
                                              results.multi_hand_landmarks[0] if results and results.multi_hand_landmarks else None, 
                                              confidence)
                
                # Update performance stats
                frame_count += 1
                if time.time() - start_time >= 1.0:
                    self.performance_stats['fps'] = frame_count
                    self.performance_stats['last_update'] = time.time()
                    self.performance_stats['frame_count'] = frame_count
                    frame_count = 0
                    start_time = time.time()
                
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
        finally:
            self.cleanup()
            self.speak("Accessibility controller stopped")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.cap:
                self.cap.release()
            if hasattr(self, 'hands') and self.hands:
                self.hands.close()
            self.running = False
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def stop_processing(self):
        """Stop the processing loop"""
        self.running = False
        self.cleanup()
    
    def get_current_frame(self):
        """Get current processed frame for web interface"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return self.performance_stats
    
    def run(self):
        """Legacy method for compatibility - use start_processing instead"""
        self.start_processing()

# Main execution
if __name__ == "__main__":
    controller = AccessibilityController()
    try:
        controller.start_processing()
    except KeyboardInterrupt:
        logger.info("Stopping controller...")
        controller.stop_processing()