# accessibility_controller.py - Combined improved version
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyautogui
import pyttsx3
import time
import threading
from typing import Optional, Dict, Any, List, Tuple, Union
from pynput import keyboard, mouse
from pynput.keyboard import Controller as KeyController, Key
from pynput.mouse import Controller as MouseController, Button
import os
import json
import platform
from datetime import datetime
import pickle
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from logging_config import logger, log_exceptions
from config import config_manager, SystemConfig

class AccessibilityController:
    def __init__(self):
        self.config = config_manager.get_config()
        
        # Initialize MediaPipe Hands with error handling
        self._initialize_mediapipe()
        
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
        
        self.model: Optional[tf.keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self._initialize_model()
        
        # Control state
        self.current_gesture = "none"
        self.previous_gesture = "none"
        self.gesture_start_time = 0.0
        
        # Processing state
        self.running = False
        self.current_frame: Optional[np.ndarray] = None
        self.current_results = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Mouse control
        self.mouse_controller = MouseController()
        self.key_controller = KeyController()
        self.cursor_mode = False
        self.dragging = False
        
        # Voice feedback
        self.voice_engine: Optional[pyttsx3.Engine] = None
        self.voice_enabled = False
        self._initialize_voice()
        self._speech_lock = threading.Lock()
        
        # Training state
        self.is_training = False
        self.training_gesture: Optional[str] = None
        self.training_data: Dict[str, List[Dict]] = {}
        self._load_training_data()
        
        # Performance monitoring
        self.performance_stats = {
            'fps': 0,
            'last_update': time.time(),
            'frame_count': 0,
            'memory_usage': 0,
            'avg_frame_time': 0
        }

    @log_exceptions
    def _initialize_mediapipe(self) -> None:
        """Initialize MediaPipe with proper cleanup and error handling"""
        try:
            # Clean up existing instance completely
            if hasattr(self, 'hands'):
                try:
                    self.hands.close()
                except:
                    pass
                # Remove the attribute to force fresh initialization
                if hasattr(self, 'hands'):
                    delattr(self, 'hands')
            
            if hasattr(self, 'mp_hands'):
                delattr(self, 'mp_hands')
            
            if hasattr(self, 'mp_draw'):
                delattr(self, 'mp_draw')
            
            # Force garbage collection
            gc.collect()
            
            # Fresh initialization
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            self.mp_draw = mp.solutions.drawing_utils
            logger.info("MediaPipe initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            self.hands = None
            raise

    @log_exceptions
    def _initialize_model(self) -> None:
        """Load or create the gesture recognition model"""
        try:
            model_dir = os.path.dirname(self.config.model_path)
            os.makedirs(model_dir, exist_ok=True)
            
            if os.path.exists(self.config.model_path):
                logger.info("Loading existing model...")
                self.model = tf.keras.models.load_model(self.config.model_path)
                logger.info("Model loaded successfully")
                
                # Try to load scaler
                scaler_path = self.config.model_path.replace('.h5', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    logger.info("Scaler loaded successfully")
                else:
                    logger.warning("Scaler not found, creating new one")
                    self.scaler = StandardScaler()
            else:
                logger.info("Creating new accessibility model...")
                self.model = self._create_improved_model()
                self.scaler = StandardScaler()
                # Don't create dummy data - let user train properly
                logger.info("Model created. Please train with real gestures.")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            self.model = None
            self.scaler = None

    @log_exceptions
    def _create_improved_model(self) -> tf.keras.Model:
        """Create an improved model for gesture recognition"""
        try:
            # Improved model architecture with regularization
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu', input_shape=(63,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.4),
                
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                
                tf.keras.layers.Dense(len(self.gestures), activation='softmax')
            ])
            
            # Use a lower learning rate and better optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(
                optimizer=optimizer, 
                loss='categorical_crossentropy', 
                metrics=['categorical_accuracy']
            )
            return model
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise

    @log_exceptions
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize hand landmarks for better training"""
        if landmarks is None or len(landmarks) != 63:
            return landmarks
        
        # Reshape to (21, 3) for processing
        landmarks_2d = landmarks.reshape(21, 3)
        
        # Get wrist position (landmark 0) as reference
        wrist = landmarks_2d[0]
        
        # Translate all points relative to wrist
        normalized = landmarks_2d - wrist
        
        # Scale by hand size (distance from wrist to middle finger tip)
        middle_finger_tip = normalized[12]  # landmark 12
        hand_size = np.linalg.norm(middle_finger_tip)
        
        if hand_size > 0.01:  # Avoid division by very small numbers
            normalized = normalized / hand_size
        
        # Flatten back to 63 features
        return normalized.flatten()

    @log_exceptions
    def _initialize_voice(self) -> None:
        """Initialize voice feedback system"""
        try:
            self.voice_engine = pyttsx3.init()
            self.voice_engine.setProperty('rate', 150)
            self.voice_engine.setProperty('volume', 0.8)
            self.voice_enabled = True
            logger.info("Voice engine initialized successfully")
        except Exception as e:
            logger.warning(f"Voice engine initialization failed: {e}")
            self.voice_engine = None
            self.voice_enabled = False

    @log_exceptions
    def _load_training_data(self) -> None:
        """Load training data from file using standardized format"""
        training_dir = os.path.dirname(self.config.training_data_path)
        os.makedirs(training_dir, exist_ok=True)
        
        if os.path.exists(self.config.training_data_path):
            try:
                with open(self.config.training_data_path, 'rb') as f:
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

    @log_exceptions
    def start_processing(self) -> bool:
        """Start the main processing loop in a separate thread"""
        if self.running:
            logger.warning("Controller is already running")
            return False
        
        try:
            # Reinitialize MediaPipe if it was closed
            if not hasattr(self, 'hands') or self.hands is None or (hasattr(self.hands, '_graph') and self.hands._graph is None):
                self._initialize_mediapipe()
            
            self.shutdown_event.clear()
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="AccessibilityProcessor"
            )
            self.processing_thread.start()
            
            # Wait for thread to start
            time.sleep(0.5)
            if self.running:
                logger.info("Accessibility controller started successfully")
                return True
            else:
                logger.error("Failed to start processing thread")
                return False
                
        except Exception as e:
            logger.error(f"Error starting processing: {e}")
            return False

    @log_exceptions
    def _processing_loop(self) -> None:
        """Main processing loop with proper resource management"""
        self.running = True
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Could not open camera")
                self.running = False
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            
            logger.info("Starting Accessibility Controller processing...")
            self.speak("Accessibility controller started")
            
            # Performance monitoring
            start_time = time.time()
            frame_count = 0
            frame_times = []
            
            while self.running and not self.shutdown_event.is_set():
                loop_start = time.time()
                
                # Read frame with timeout
                ret, frame = self._read_frame_with_timeout()
                if not ret:
                    continue
                
                # Process frame
                processed_frame = self._process_frame(frame)
                if processed_frame is not None:
                    self.current_frame = processed_frame
                
                # Performance monitoring
                frame_count += 1
                frame_time = time.time() - loop_start
                frame_times.append(frame_time)
                
                # Update stats every second
                if time.time() - start_time >= 1.0:
                    self._update_performance_stats(frame_count, frame_times)
                    frame_count = 0
                    frame_times = []
                    start_time = time.time()
                
                # Memory management
                if frame_count % 30 == 0:
                    self._cleanup_memory()
                
                # Maintain target FPS
                self._maintain_fps(loop_start)
                
        except Exception as e:
            logger.error(f"Processing loop error: {e}")
        finally:
            self._cleanup_resources()
            logger.info("Processing loop stopped")

    @log_exceptions
    def _read_frame_with_timeout(self, timeout_ms: int = 1000) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame with timeout to prevent hanging - cross-platform implementation"""
        if self.cap is None:
            return False, None
        
        # Implementation for different platforms
        try:
            # Try setting timeout if supported
            if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            elif hasattr(cv2, 'CAP_PROP_TIMEOUT_MSEC'):
                self.cap.set(cv2.CAP_PROP_TIMEOUT_MSEC, timeout_ms)
            
            ret, frame = self.cap.read()
            
            # Fallback: if no timeout support, implement manual timeout
            if not ret and timeout_ms > 0:
                start_time = time.time()
                while not ret and (time.time() - start_time) * 1000 < timeout_ms:
                    time.sleep(0.01)
                    ret, frame = self.cap.read()
            
            return ret, frame
            
        except Exception as e:
            logger.error(f"Frame reading error: {e}")
            return False, None

    @log_exceptions
    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a single frame with comprehensive error handling"""
        try:
            frame = cv2.flip(frame, 1)
            image, results = self.mediapipe_detection(frame)
            image = self.draw_landmarks(image, results)
            
            # Store current results for web interface
            self.current_results = results
            
            # Extract keypoints and recognize gesture
            keypoints = self.extract_keypoints(results)
            
            # Control cursor if in cursor mode and hand detected
            if results and results.multi_hand_landmarks and self.cursor_mode:
                self.control_cursor(results.multi_hand_landmarks[0])
            
            # Release drag if no hand detected
            if self.dragging and (not results or not results.multi_hand_landmarks):
                self._release_drag()
            
            # Recognize gesture and execute action
            if self.model is not None and keypoints is not None:
                gesture, confidence = self.recognize_gesture(keypoints)
                self.current_gesture = gesture
                
                if gesture != "none" and confidence > 0.5:
                    logger.debug(f"Detected: {gesture} (confidence: {confidence:.2f})")
                
                self.execute_gesture_action(
                    gesture, 
                    results.multi_hand_landmarks[0] if results and results.multi_hand_landmarks else None, 
                    confidence
                )
            
            return image
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return None

    @log_exceptions
    def _update_performance_stats(self, frame_count: int, frame_times: List[float]) -> None:
        """Update performance statistics"""
        try:
            if frame_times:
                avg_fps = frame_count / (time.time() - self.performance_stats['last_update'])
                avg_frame_time = np.mean(frame_times) * 1000  # ms
                
                self.performance_stats.update({
                    'fps': avg_fps,
                    'frame_count': frame_count,
                    'avg_frame_time': avg_frame_time,
                    'last_update': time.time(),
                    'memory_usage': self._get_memory_usage()
                })
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")

    @log_exceptions
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    @log_exceptions
    def _cleanup_memory(self) -> None:
        """Clean up memory to prevent leaks"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear TensorFlow session if it exists
            if hasattr(tf, 'keras'):
                tf.keras.backend.clear_session()
                
        except Exception as e:
            logger.warning(f"Memory cleanup error: {e}")

    @log_exceptions
    def _maintain_fps(self, loop_start: float) -> None:
        """Maintain target FPS by sleeping if necessary"""
        try:
            elapsed = time.time() - loop_start
            target_time = 1.0 / self.config.target_fps
            
            if elapsed < target_time:
                time.sleep(target_time - elapsed)
        except Exception as e:
            logger.warning(f"FPS maintenance error: {e}")

    @log_exceptions
    def _release_drag(self) -> None:
        """Release drag operation safely"""
        try:
            if self.dragging:
                self.mouse_controller.release(Button.left)
                self.dragging = False
                logger.debug("Drag released due to hand loss")
        except Exception as e:
            logger.error(f"Drag release error: {e}")
            self.dragging = False

    @log_exceptions
    def stop_processing(self) -> bool:
        """Stop the processing loop gracefully"""
        if not self.running:
            return True
        
        logger.info("Stopping accessibility controller...")
        self.running = False
        self.shutdown_event.set()
        
        # Wait for thread to finish with timeout
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            if self.processing_thread.is_alive():
                logger.warning("Processing thread did not stop gracefully")
        
        self._cleanup_resources()
        logger.info("Accessibility controller stopped")
        return True

    @log_exceptions
    def _cleanup_resources(self) -> None:
        """Clean up all resources but don't destroy MediaPipe permanently"""
        try:
            # Release camera
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Close MediaPipe safely - but don't set to None so we can check if it exists
            if hasattr(self, 'hands') and self.hands:
                try:
                    self.hands.close()
                except Exception as e:
                    logger.debug(f"MediaPipe cleanup warning: {e}")
            
            # Clear TensorFlow session
            if hasattr(tf, 'keras'):
                tf.keras.backend.clear_session()
            
            # Clear frame variables but keep other state
            self.current_frame = None
            self.current_results = None
            
            logger.info("Resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self.running = False

    @log_exceptions
    def mediapipe_detection(self, image: np.ndarray) -> Tuple[np.ndarray, Any]:
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

    @log_exceptions
    def extract_keypoints(self, results: Any) -> Optional[np.ndarray]:
        """Extract and normalize hand landmarks from MediaPipe results"""
        if results and results.multi_hand_landmarks:
            try:
                # Use the first hand detected
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # Normalize the landmarks
                landmarks_array = np.array(landmarks)
                normalized_landmarks = self.normalize_landmarks(landmarks_array)
                
                return normalized_landmarks
            except Exception as e:
                logger.error(f"Keypoint extraction error: {e}")
                return np.zeros(21*3)
        return np.zeros(21*3)

    @log_exceptions
    def draw_landmarks(self, image: np.ndarray, results: Any) -> np.ndarray:
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

    @log_exceptions
    def recognize_gesture(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """Improved gesture recognition with proper preprocessing"""
        if (self.model is None or self.scaler is None or landmarks is None or 
            len(landmarks) != 63 or np.all(landmarks == 0)):
            return "none", 0.0
        
        try:
            # Reshape and scale the input
            landmarks_scaled = self.scaler.transform(landmarks.reshape(1, -1))
            
            # Predict gesture
            prediction = self.model.predict(landmarks_scaled, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            
            # Use a more reasonable confidence threshold
            if confidence > 0.6:  # Adjusted threshold
                return self.gestures[predicted_class], confidence
            else:
                return "none", confidence
                
        except Exception as e:
            logger.error(f"Gesture recognition error: {e}")
            return "none", 0.0

    @log_exceptions
    def control_cursor(self, hand_landmarks: Any) -> None:
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
            new_x = current_x + (x - current_x) * 0.3 * self.config.cursor_speed
            new_y = current_y + (y - current_y) * 0.3 * self.config.cursor_speed
            
            self.mouse_controller.position = (new_x, new_y)
        except Exception as e:
            logger.error(f"Cursor control error: {e}")

    @log_exceptions
    def safe_execute_action(self, action_func, action_name: str = "action") -> None:
        """Safely execute actions with error handling"""
        try:
            action_func()
        except Exception as e:
            logger.error(f"{action_name} execution error: {e}")
            self.speak("Action failed")

    @log_exceptions
    def execute_gesture_action(self, gesture: str, hand_landmarks: Any = None, confidence: float = 0.0) -> None:
        """Execute computer control actions based on recognized gesture"""
        if confidence < self.config.gesture_confidence_threshold:
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
                lambda: pyautogui.scroll(self.config.scroll_speed), "scroll_up"),
            "scroll_down": lambda: self.safe_execute_action(
                lambda: pyautogui.scroll(-self.config.scroll_speed), "scroll_down"),
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

    @log_exceptions
    def execute_browser_command(self, command: str) -> None:
        """Execute browser-specific commands with fallbacks"""
        try:
            # Try browser-specific keys first
            browser_keys = {
                'back': ('browser_back', 'chrome_back'),
                'forward': ('browser_forward', 'chrome_forward'),
                'new_tab': (Key.ctrl, 't'),
                'close_tab': (Key.ctrl, 'w')
            }
            
            if command in browser_keys:
                key_combo = browser_keys[command]
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

    @log_exceptions
    def toggle_cursor_mode(self, duration: float) -> None:
        """Toggle cursor mode with hold threshold"""
        if duration > self.config.gesture_hold_time:
            self.cursor_mode = not self.cursor_mode
            mode = "enabled" if self.cursor_mode else "disabled"
            self.speak(f"Cursor mode {mode}")
            self.gesture_start_time = time.time()  # Reset timer

    @log_exceptions
    def handle_drag(self, hand_landmarks: Any) -> None:
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

    @log_exceptions
    def execute_key_combo(self, modifier: Any, key: Any) -> None:
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

    @log_exceptions
    def emergency_stop(self) -> None:
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

    @log_exceptions
    def show_help(self) -> None:
        """Show available gestures and commands"""
        help_text = "Available gestures: Cursor mode, Click, Right click, Drag, Scroll, Copy, Paste, Cut, Undo, Redo, Save, New tab, Close tab, Switch app, Desktop, Task view, Help, Media controls, Volume controls, Zoom controls"
        self.speak(help_text)

    @log_exceptions
    def speak(self, text: str) -> None:
        """Provide voice feedback with proper thread safety"""
        if not self.voice_enabled or not self.config.voice_feedback or not self.voice_engine:
            return
        
        try:
            # Use a lock to ensure thread safety
            with self._speech_lock:
                # Stop any ongoing speech
                try:
                    self.voice_engine.stop()
                except:
                    pass
                
                # Say new text
                self.voice_engine.say(text)
                self.voice_engine.runAndWait()
                
        except Exception as e:
            logger.error(f"Speech error: {e}")
            # Try to reinitialize voice engine on failure
            try:
                self._initialize_voice()
            except:
                self.voice_enabled = False

    @log_exceptions
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current processed frame for web interface"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None

    @log_exceptions
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_stats

    @log_exceptions
    def get_status(self) -> Dict[str, Any]:
        """Get system status with data sanitization"""
        try:
            hand_detected = (
                self.current_results and 
                hasattr(self.current_results, 'multi_hand_landmarks') and 
                self.current_results.multi_hand_landmarks
            )
            hand_count = len(self.current_results.multi_hand_landmarks) if hand_detected else 0
            
            status = {
                "running": self.running,
                "cursor_mode": self.cursor_mode,
                "voice_enabled": self.voice_enabled,
                "current_gesture": self.current_gesture,
                "hand_detected": hand_detected,
                "hand_count": hand_count,
                "performance": self.performance_stats
            }
            
            # Sanitize data for JSON serialization
            return self._sanitize_for_json(status)
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": "Failed to get status"}

    def _sanitize_for_json(self, data: Any) -> Any:
        """Recursively sanitize data for JSON serialization"""
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, dict):
            return {k: self._sanitize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_for_json(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, '__dict__'):
            return self._sanitize_for_json(data.__dict__)
        else:
            return str(data)

    # Training methods
    @log_exceptions
    def validate_gesture_name(self, gesture: str) -> bool:
        """Validate gesture name format"""
        if not isinstance(gesture, str):
            return False
        if gesture not in self.gestures:
            return False
        # Allow only alphanumeric and underscore characters
        import re
        if not re.match(r'^[a-zA-Z0-9_]+$', gesture):
            return False
        return True

    @log_exceptions
    def validate_training_sample(self, keypoints: np.ndarray) -> bool:
        """Improved validation for training samples"""
        if keypoints is None or not isinstance(keypoints, np.ndarray):
            return False
        if keypoints.shape != (63,):
            return False
        if np.any(np.isnan(keypoints)) or np.any(np.isinf(keypoints)):
            return False
        
        # More lenient validation - just check if hand is detected
        # (normalized landmarks can have negative values, which is fine)
        if np.all(keypoints == 0):
            return False
        
        # Check for reasonable variation (hand should have some structure)
        if np.std(keypoints) < 0.001:
            return False
        
        return True

    @log_exceptions
    def start_training(self, gesture: str) -> bool:
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

    @log_exceptions
    def capture_training_sample(self, gesture: str) -> int:
        """Improved training sample capture with better validation"""
        if not self.is_training or gesture != self.training_gesture:
            logger.warning("Not in training mode or gesture mismatch")
            return 0
        
        if self.current_results and self.current_results.multi_hand_landmarks:
            try:
                # Extract keypoints from the current frame
                keypoints = self.extract_keypoints(self.current_results)
                
                # Validate sample
                if self.validate_training_sample(keypoints):
                    # Add to training data with improved format
                    sample = {
                        'keypoints': keypoints,
                        'timestamp': datetime.now().isoformat(),
                        'gesture': gesture,
                        'version': '2.0',  # Updated format version
                        'normalized': True  # Flag indicating data is normalized
                    }
                    
                    # Initialize if needed
                    if gesture not in self.training_data:
                        self.training_data[gesture] = []
                    
                    self.training_data[gesture].append(sample)
                    
                    # Save training data
                    self.save_training_data()
                    
                    sample_count = len(self.training_data[gesture])
                    logger.info(f"Captured sample {sample_count} for {gesture}")
                    
                    return sample_count
                else:
                    logger.warning("Invalid sample (validation failed)")
                    return len(self.training_data.get(gesture, []))
                
            except Exception as e:
                logger.error(f"Error capturing training sample: {e}")
                return len(self.training_data.get(gesture, []))
        else:
            logger.warning("No hand landmarks detected")
            return len(self.training_data.get(gesture, []))

    @log_exceptions
    def save_training_data(self) -> bool:
        """Save training data to file using standardized format"""
        try:
            training_dir = os.path.dirname(self.config.training_data_path)
            os.makedirs(training_dir, exist_ok=True)
            
            # Ensure data is in correct format before saving
            validated_data = {}
            for gesture, samples in self.training_data.items():
                if gesture in self.gestures and isinstance(samples, list):
                    validated_data[gesture] = samples
            
            with open(self.config.training_data_path, 'wb') as f:
                pickle.dump(validated_data, f)
            logger.info("Training data saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
            return False
        
    @log_exceptions
    def train_model(self) -> Dict[str, Any]:
        """Improved model training with better data handling"""
        try:
            # Collect valid training samples
            X_train = []
            y_train = []
            
            # Count samples per gesture
            gesture_counts = {}
            
            for i, gesture in enumerate(self.gestures):
                if gesture in self.training_data and self.training_data[gesture]:
                    valid_samples = []
                    for sample in self.training_data[gesture]:
                        if (isinstance(sample, dict) and 
                            'keypoints' in sample and 
                            self.validate_training_sample(sample['keypoints']) and
                            sample.get('gesture') == gesture):
                            valid_samples.append(sample['keypoints'])
                    
                    gesture_counts[gesture] = len(valid_samples)
                    
                    # Add valid samples
                    for keypoints in valid_samples:
                        X_train.append(keypoints)
                        y_train.append(i)
            
            # Check if we have enough data
            total_samples = len(X_train)
            if total_samples < 15:  # Minimum samples needed
                return {
                    "status": "error",
                    "message": f"Not enough training data. Need at least 15 samples, have {total_samples}"
                }
            
            # Check class distribution
            unique_labels, counts = np.unique(y_train, return_counts=True)
            min_samples_per_class = np.min(counts) if len(counts) > 0 else 0
            
            if min_samples_per_class < 3:
                return {
                    "status": "error",
                    "message": f"Some gestures have too few samples. Need at least 3 samples per gesture. Current distribution: {dict(zip([self.gestures[i] for i in unique_labels], counts))}"
                }
            
            # Convert to numpy arrays
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Apply data scaling if we have a scaler
            if self.scaler is None:
                self.scaler = StandardScaler()
            
            # Fit and transform the data
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Convert labels to categorical
            num_classes = len(self.gestures)
            y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
            
            # Split data for validation
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_scaled, y_train_categorical, 
                test_size=0.2, 
                random_state=42,
                stratify=y_train
            )
            
            logger.info(f"Training on {len(X_train_final)} samples, validating on {len(X_val)} samples")
            logger.info(f"Gesture distribution: {gesture_counts}")
            
            # Create or reset model
            self.model = self._create_improved_model()
            
            # Define callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_categorical_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_categorical_accuracy',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Train the model with more epochs
            history = self.model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val, y_val),
                epochs=100,  # Increased epochs
                batch_size=min(32, len(X_train_final) // 2),  # Adaptive batch size
                verbose=1,
                callbacks=callbacks
            )
            
            # Get training results
            final_accuracy = history.history['categorical_accuracy'][-1]
            final_val_accuracy = history.history['val_categorical_accuracy'][-1]
            final_loss = history.history['loss'][-1]
            
            logger.info(f"Training completed. Accuracy: {final_accuracy:.2%}, Validation Accuracy: {final_val_accuracy:.2%}")
            
            # Save model and scaler
            self.save_model()
            
            return {
                "status": "success",
                "accuracy": float(final_accuracy),
                "val_accuracy": float(final_val_accuracy),
                "loss": float(final_loss),
                "total_samples": total_samples,
                "gesture_counts": gesture_counts
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    @log_exceptions
    def save_model(self) -> bool:
        """Save model and scaler"""
        try:
            if self.model is not None:
                model_dir = os.path.dirname(self.config.model_path)
                os.makedirs(model_dir, exist_ok=True)
                
                # Save model
                self.model.save(self.config.model_path)
                logger.info("Model saved successfully")
                
                # Save scaler
                if self.scaler is not None:
                    scaler_path = self.config.model_path.replace('.h5', '_scaler.pkl')
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scaler, f)
                    logger.info("Scaler saved successfully")
                
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    @log_exceptions
    def get_training_status(self) -> Dict[str, Any]:
        """Get training status with improved information"""
        total_samples = sum(len(samples) for samples in self.training_data.values())
        current_samples = len(self.training_data[self.training_gesture]) if self.training_gesture in self.training_data else 0
        
        # Calculate per-gesture statistics
        gesture_stats = {}
        for gesture in self.gestures:
            if gesture in self.training_data:
                gesture_stats[gesture] = len(self.training_data[gesture])
            else:
                gesture_stats[gesture] = 0
        
        return {
            "current_gesture": self.training_gesture,
            "samples_collected": current_samples,
            "total_samples": total_samples,
            "is_training": self.is_training,
            "gestures_trained": [g for g in self.gestures if g in self.training_data and self.training_data[g]],
            "format_version": "2.0",
            "gesture_stats": gesture_stats,
            "min_samples_per_gesture": 5,  # Recommended minimum
            "ready_for_training": total_samples >= 15 and len([g for g in gesture_stats.values() if g >= 3]) >= 2
        }

    @log_exceptions
    def reset_training_data(self, gesture: Optional[str] = None) -> bool:
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

    @log_exceptions
    def export_training_data(self, format: str = 'pkl') -> Optional[str]:
        """Export training data in standardized format"""
        try:
            export_data = {
                'metadata': {
                    'export_date': datetime.now().isoformat(),
                    'total_samples': sum(len(samples) for samples in self.training_data.values()),
                    'format_version': '2.0',
                    'gestures': list(self.training_data.keys())
                },
                'samples': self.training_data
            }
            
            if format == 'pkl':
                export_path = os.path.join(os.path.dirname(self.config.training_data_path), 'training_export.pkl')
                with open(export_path, 'wb') as f:
                    pickle.dump(export_data, f)
            elif format == 'json':
                export_path = os.path.join(os.path.dirname(self.config.training_data_path), 'training_export.json')
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

    @log_exceptions
    def restart_processing(self) -> bool:
        """Restart the processing loop with fresh MediaPipe instance"""
        self.stop_processing()
        time.sleep(1.0)  # Give time for cleanup
        
        # Force reinitialization of MediaPipe
        if hasattr(self, 'hands'):
            try:
                self.hands.close()
            except:
                pass
            if hasattr(self, 'hands'):
                del self.hands
        
        # Reinitialize MediaPipe
        self._initialize_mediapipe()
        
        return self.start_processing()   

    @log_exceptions
    def run(self) -> None:
        """Legacy method for compatibility - use start_processing instead"""
        self.start_processing()

    @log_exceptions
    def cleanup_resources(self) -> None:
        """Public method to cleanup resources"""
        self._cleanup_resources()

# Main execution
if __name__ == "__main__":
    controller = AccessibilityController()
    try:
        controller.start_processing()
    except KeyboardInterrupt:
        logger.info("Stopping controller...")
        controller.stop_processing()
    except Exception as e:
        logger.error(f"Controller error: {e}")
        controller.stop_processing()