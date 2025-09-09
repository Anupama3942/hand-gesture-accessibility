import cv2
import numpy as np
import json
import os
import tensorflow as tf
from accessibility_controller import AccessibilityController
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingUI:
    def __init__(self):
        self.controller = AccessibilityController()
        self.current_gesture = None
        self.recording = False
        self.sequence_count = 0
        self.total_sequences = 30
        self.sequences = []
        
        # Create training directory
        training_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data')
        os.makedirs(training_dir, exist_ok=True)
        
        # Gestures to train
        self.gestures = [
            'cursor_mode', 'click', 'right_click', 'double_click', 'drag',
            'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out', 'back',
            'forward', 'volume_up', 'volume_down', 'mute', 'play_pause',
            'next_track', 'prev_track', 'copy', 'paste', 'cut',
            'undo', 'redo', 'save', 'new_tab', 'close_tab',
            'switch_app', 'desktop', 'task_view', 'help'
        ]
    
    def draw_training_ui(self, image, gesture, count, total):
        """Draw training interface on the image"""
        # Draw background for text
        cv2.rectangle(image, (0, 0), (640, 120), (0, 0, 0), -1)
        
        # Draw current gesture
        cv2.putText(image, f"Training: {gesture}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw progress
        cv2.putText(image, f"Sequences: {count}/{total}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw instructions
        if not self.recording:
            cv2.putText(image, "Press SPACE to start recording", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        else:
            cv2.putText(image, "Recording... Perform the gesture", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(image, "Press SPACE to stop", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return image
    
    def save_sequences(self, gesture, sequences):
        """Save training sequences"""
        gesture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data', gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        
        # Save sequences as numpy array
        sequences_array = np.array(sequences)
        np.save(os.path.join(gesture_dir, 'sequences.npy'), sequences_array)
        
        # Save metadata
        metadata = {
            "gesture": gesture,
            "sequences_count": len(sequences),
            "timestamp": str(np.datetime64('now')),
            "sequence_length": len(sequences[0]) if sequences else 0
        }
        
        with open(os.path.join(gesture_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run_training(self):
        """Run the training UI"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        current_gesture_index = 0
        self.current_gesture = self.gestures[current_gesture_index]
        
        print("Gesture Training System")
        print("======================")
        print("Press SPACE to start/stop recording")
        print("Press N for next gesture")
        print("Press P for previous gesture")
        print("Press Q to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            image, results = self.controller.mediapipe_detection(frame)
            image = self.controller.draw_landmarks(image, results)
            
            # Draw training UI
            image = self.draw_training_ui(image, self.current_gesture, 
                                        self.sequence_count, self.total_sequences)
            
            # Extract keypoints if recording
            if self.recording:
                keypoints = self.controller.extract_keypoints(results)
                self.sequences.append(keypoints)
                
                # Check if we have enough frames
                if len(self.sequences) >= 30:
                    self.recording = False
                    self.sequence_count += 1
                    
                    # Save the sequence
                    if len(self.sequences) > 30:
                        self.sequences = self.sequences[:30]
                    
                    # Save every 5 sequences or when done
                    if self.sequence_count % 5 == 0 or self.sequence_count >= self.total_sequences:
                        self.save_sequences(self.current_gesture, [self.sequences])
                        self.sequences = []
                    
                    print(f"Saved sequence {self.sequence_count}/{self.total_sequences} for {self.current_gesture}")
            
            cv2.imshow('Gesture Training', image)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord(' '):  # SPACE to toggle recording
                self.recording = not self.recording
                if self.recording:
                    self.sequences = []  # Start new sequence
                    print(f"Recording sequence {self.sequence_count + 1} for {self.current_gesture}")
            
            elif key == ord('n'):  # Next gesture
                if self.sequence_count >= self.total_sequences:
                    # Save current gesture data
                    if self.sequences:
                        self.save_sequences(self.current_gesture, [self.sequences])
                    
                    # Move to next gesture
                    current_gesture_index = (current_gesture_index + 1) % len(self.gestures)
                    self.current_gesture = self.gestures[current_gesture_index]
                    self.sequence_count = 0
                    self.sequences = []
                    print(f"Switched to {self.current_gesture}")
                else:
                    print(f"Complete {self.total_sequences} sequences for {self.current_gesture} first")
            
            elif key == ord('p'):  # Previous gesture
                if self.sequence_count >= self.total_sequences or self.sequence_count == 0:
                    # Save current gesture data
                    if self.sequences:
                        self.save_sequences(self.current_gesture, [self.sequences])
                    
                    # Move to previous gesture
                    current_gesture_index = (current_gesture_index - 1) % len(self.gestures)
                    self.current_gesture = self.gestures[current_gesture_index]
                    self.sequence_count = 0
                    self.sequences = []
                    print(f"Switched to {self.current_gesture}")
                else:
                    print(f"Complete {self.total_sequences} sequences for {self.current_gesture} first")
            
            elif key == ord('q'):  # Quit
                # Save any remaining data
                if self.sequences:
                    self.save_sequences(self.current_gesture, [self.sequences])
                break
            
            # Check if current gesture is complete
            if self.sequence_count >= self.total_sequences:
                print(f"Completed training for {self.current_gesture}!")
                print("Press N to move to next gesture or Q to quit")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Train model after collection
        self.train_model()
    
    def train_model(self):
        """Train the model with collected data"""
        print("\nStarting model training...")
        
        # Load all training data
        X_train = []
        y_train = []
        
        for i, gesture in enumerate(self.gestures):
            gesture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data', gesture)
            if not os.path.exists(gesture_dir):
                continue
            
            # Load sequences
            sequences_path = os.path.join(gesture_dir, 'sequences.npy')
            if os.path.exists(sequences_path):
                sequences = np.load(sequences_path)
                
                # Ensure sequences are 30 frames long
                for seq in sequences:
                    if len(seq) == 30:
                        X_train.append(seq)
                        y_train.append(i)
        
        if not X_train:
            print("No training data found!")
            return
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(self.gestures))
        
        # Split data (80% train, 20% validation)
        split_idx = int(0.8 * len(X_train))
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]
        X_train, y_train = X_train[:split_idx], y_train[:split_idx]
        
        print(f"Training on {len(X_train)} sequences, validating on {len(X_val)} sequences")
        
        # Create a new model if none exists
        if self.controller.model is None:
            self.controller.model = self.controller.create_model()
        
        # Train the model
        self.controller.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
            ]
        )
        
        # Save the model
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(model_dir, exist_ok=True)
        self.controller.model.save(os.path.join(model_dir, 'accessibility_gesture_model.h5'))
        print("Model trained and saved successfully!")
        
        # Evaluate the model
        loss, accuracy = self.controller.model.evaluate(X_val, y_val)
        print(f"Validation accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    trainer = TrainingUI()
    trainer.run_training()
