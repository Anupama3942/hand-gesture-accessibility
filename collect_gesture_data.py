#!/usr/bin/env python3
import cv2
import numpy as np
from accessibility_controller import AccessibilityController
import time
import os
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_gesture_data():
    """Collect training data for specific gestures using standardized format"""
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
    print("Using standardized data format")
    print("")
    
    for i, gesture in enumerate(gestures):
        print(f"{i+1}. {gesture}")
    
    try:
        choice = int(input("\nSelect gesture to collect (number): ")) - 1
        if choice < 0 or choice >= len(gestures):
            print("Invalid choice")
            return
        
        selected_gesture = gestures[choice]
        samples_to_collect = int(input("How many samples to collect? "))
        
        # Enable training mode for the selected gesture
        if not controller.start_training(selected_gesture):
            print(f"Failed to start training for {selected_gesture}")
            return
        
        print(f"\nCollecting {samples_to_collect} samples for {selected_gesture}")
        print("Press 'c' to capture sample, 'q' to quit")
        print("Make sure your hand is clearly visible in the camera")
        
        samples_collected = 0
        failed_attempts = 0
        max_failed_attempts = 10
        
        while samples_collected < samples_to_collect and failed_attempts < max_failed_attempts:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            image, results = controller.mediapipe_detection(frame)
            image = controller.draw_landmarks(image, results)

            # Keep controller state aligned with current detection frame
            controller.current_results = results
            keypoints = controller.extract_keypoints(results)
            if keypoints is not None and controller.validate_training_sample(keypoints):
                controller.latest_keypoints = keypoints.copy()
                controller.latest_keypoints_timestamp = time.time()
                controller.latest_hand_detected = True
            else:
                controller.latest_hand_detected = False
            
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
                # Ensure training mode is enabled
                if not controller.is_training or controller.training_gesture != selected_gesture:
                    controller.start_training(selected_gesture)
                
                # Capture sample using controller's standardized method
                sample_count = controller.capture_training_sample(selected_gesture)
                
                if sample_count > 0:
                    samples_collected = sample_count
                    print(f"Captured sample {samples_collected}/{samples_to_collect}")
                    failed_attempts = 0
                else:
                    print("Failed to capture sample. Ensure hand is visible.")
                    failed_attempts += 1
            
            elif key == ord('q'):
                break
        
        if failed_attempts >= max_failed_attempts:
            print("\nToo many failed attempts. Stopping collection.")
        
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
        
        # Disable training mode
        controller.is_training = False
        controller.training_gesture = None
    
    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
    except ValueError:
        print("Please enter a valid number")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            controller.cleanup_resources()
        except Exception:
            pass

def batch_collect_gestures():
    """Collect data for multiple gestures in batch mode"""
    controller = AccessibilityController()
    
    gestures = controller.gestures[1:]  # Skip 'none' gesture
    samples_per_gesture = 20
    
    print("Batch Gesture Data Collection")
    print("============================")
    print(f"Will collect {samples_per_gesture} samples for each gesture")
    print("")
    
    for gesture in gestures:
        print(f"Collecting data for: {gesture}")
        print("Perform the gesture and press 'c' to capture samples")
        print("Press 'q' to quit or 'n' for next gesture")
        print("")
        
        # Enable training mode for the current gesture
        if not controller.start_training(gesture):
            print(f"Failed to start training for {gesture}. Skipping.")
            continue
        
        samples_collected = 0
        
        # Initialize camera for this gesture
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while samples_collected < samples_per_gesture:
            try:
                ret, frame = cap.read()
                if not ret:
                    continue
            
                frame = cv2.flip(frame, 1)
                image, results = controller.mediapipe_detection(frame)
                image = controller.draw_landmarks(image, results)

                # Keep controller state aligned with current detection frame
                controller.current_results = results
                keypoints = controller.extract_keypoints(results)
                if keypoints is not None and controller.validate_training_sample(keypoints):
                    controller.latest_keypoints = keypoints.copy()
                    controller.latest_keypoints_timestamp = time.time()
                    controller.latest_hand_detected = True
                else:
                    controller.latest_hand_detected = False
            
                # Display instructions
                cv2.putText(image, f"Gesture: {gesture}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Samples: {samples_collected}/{samples_per_gesture}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, "Press 'c' to capture, 'n' for next, 'q' to quit", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
                cv2.imshow('Batch Collection', image)
            
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    # Ensure training mode is enabled
                    if not controller.is_training or controller.training_gesture != gesture:
                        controller.start_training(gesture)
                
                    sample_count = controller.capture_training_sample(gesture)
                    if sample_count > 0:
                        samples_collected = sample_count
                        print(f"Captured sample {samples_collected}/{samples_per_gesture}")
            
                elif key == ord('n'):
                    print(f"Moving to next gesture. Collected {samples_collected} samples for {gesture}")
                    break
            
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    controller.is_training = False
                    controller.training_gesture = None
                    try:
                        controller.cleanup_resources()
                    except Exception:
                        pass
                    return
            except KeyboardInterrupt:
                print("\nBatch collection interrupted by user.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Disable training mode after each gesture
        controller.is_training = False
        controller.training_gesture = None
    
    # Save all data and train model
    controller.save_training_data()
    print("\nAll data collected. Training model...")
    
    result = controller.train_model()
    if result["status"] == "success":
        print(f"Model trained successfully! Accuracy: {result['accuracy']:.2%}")
        if controller.save_model():
            print("Model saved successfully!")
    else:
        print(f"Training failed: {result['message']}")
    try:
        controller.cleanup_resources()
    except Exception:
        pass

def check_controller_methods(controller):
    """Helper function to check available methods in the controller"""
    print("Available methods in AccessibilityController:")
    methods = [method for method in dir(controller) if not method.startswith('_')]
    for method in methods:
        print(f"  - {method}")
    
    print("\nChecking for training-related attributes:")
    training_attrs = ['start_training', 'capture_training_sample',
                     'is_training', 'training_gesture',
                     'latest_keypoints', 'latest_hand_detected']
    
    for attr in training_attrs:
        if hasattr(controller, attr):
            print(f"  ✓ {attr}")
        else:
            print(f"  ✗ {attr}")

if __name__ == "__main__":
    # Optional: Check controller methods for debugging
    debug_controller = input("Debug controller methods? (y/n): ")
    if debug_controller.lower() == 'y':
        controller = AccessibilityController()
        check_controller_methods(controller)
        print("\n")
    
    print("Choose collection mode:")
    print("1. Single gesture collection")
    print("2. Batch collection (all gestures)")
    
    try:
        choice = int(input("Enter choice (1 or 2): "))
        if choice == 1:
            collect_gesture_data()
        elif choice == 2:
            batch_collect_gestures()
        else:
            print("Invalid choice")
    except ValueError:
        print("Please enter a valid number")