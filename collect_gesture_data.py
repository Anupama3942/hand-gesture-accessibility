#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
from accessibility_controller import AccessibilityController
import time
import os
import shutil

def collect_gesture_data():
    """Collect training data for specific gestures"""
    controller = AccessibilityController()
    
    # Delete the old model if it exists (to force creating a new one)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'accessibility_gesture_model.h5')
    if os.path.exists(model_path):
        os.remove(model_path)
        print("Deleted old model to create a new one")
    
    # Reinitialize the controller to create a new model
    controller = AccessibilityController()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    gestures = controller.gestures[1:]  # Skip 'none' gesture
    
    print("Gesture Data Collection")
    print("======================")
    
    for i, gesture in enumerate(gestures):
        print(f"\n{i+1}. {gesture}")
    
    try:
        choice = int(input("\nSelect gesture to collect (number): ")) - 1
        if choice < 0 or choice >= len(gestures):
            print("Invalid choice")
            return
        
        selected_gesture = gestures[choice]
        samples_to_collect = int(input("How many samples to collect? "))
        
        print(f"\nCollecting {samples_to_collect} samples for {selected_gesture}")
        print("Press 'c' to capture sample, 'q' to quit")
        print("Make sure your hand is clearly visible in the camera")
        
        samples_collected = 0
        
        while samples_collected < samples_to_collect:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            image, results = controller.mediapipe_detection(frame)
            image = controller.draw_landmarks(image, results)
            
            # Display instructions
            cv2.putText(image, f"Gesture: {selected_gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Samples: {samples_collected}/{samples_to_collect}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, "Press 'c' to capture, 'q' to quit", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show hand detection status
            if results and results.multi_hand_landmarks:
                hand_status = "Hand detected ✓"
                color = (0, 255, 0)
            else:
                hand_status = "No hand detected ✗"
                color = (0, 0, 255)
            
            cv2.putText(image, hand_status, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('Collect Gesture Data', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Capture sample
                if results and results.multi_hand_landmarks:
                    keypoints = controller.extract_keypoints(results)
                    if keypoints is not None and np.any(keypoints) and not np.all(keypoints == 0):
                        # Add to training data
                        if selected_gesture not in controller.training_data:
                            controller.training_data[selected_gesture] = []
                        
                        controller.training_data[selected_gesture].append({
                            'keypoints': keypoints,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        samples_collected += 1
                        print(f"Captured sample {samples_collected}/{samples_to_collect}")
                    else:
                        print("Invalid keypoints (all zeros or no hand detected)")
                else:
                    print("No hand detected")
            
            elif key == ord('q'):
                break
        
        # Save training data
        if samples_collected > 0:
            controller.save_training_data()
            print(f"\nSaved {samples_collected} samples for {selected_gesture}")
            
            # Train model
            train = input("Train model with new data? (y/n): ")
            if train.lower() == 'y':
                result = controller.train_model()
                if result["status"] == "success":
                    print(f"Model trained successfully! Accuracy: {result['accuracy']:.2%}")
                    save = input("Save model? (y/n): ")
                    if save.lower() == 'y':
                        if controller.save_model():
                            print("Model saved successfully!")
                        else:
                            print("Failed to save model")
                else:
                    print(f"Training failed: {result['message']}")
    
    except ValueError:
        print("Please enter a valid number")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_gesture_data()