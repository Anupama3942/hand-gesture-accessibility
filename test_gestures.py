#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
from accessibility_controller import AccessibilityController
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gesture_recognition():
    """Test gesture recognition with live camera feed"""
    controller = AccessibilityController()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Testing gesture recognition. Press 'q' to quit.")
    print("Make gestures in front of the camera to see recognition results.")
    print("")
    
    # Performance tracking
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        image, results = controller.mediapipe_detection(frame)
        image = controller.draw_landmarks(image, results)
        
        # Extract keypoints and recognize gesture
        keypoints = controller.extract_keypoints(results)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        if keypoints is not None and np.any(keypoints) and not np.all(keypoints == 0):
            gesture, confidence = controller.recognize_gesture(keypoints)
            
            # Count successful detections
            if gesture != "none" and confidence > 0.5:
                detection_count += 1
            
            # Display results
            cv2.putText(image, f"Gesture: {gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"FPS: {fps:.1f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display detection rate
            detection_rate = (detection_count / frame_count) * 100
            cv2.putText(image, f"Detection: {detection_rate:.1f}%", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show performance metrics periodically
            if frame_count % 30 == 0:
                print(f"Frame: {frame_count}, FPS: {fps:.1f}, Detection Rate: {detection_rate:.1f}%")
                if gesture != "none":
                    print(f"Last detection: {gesture} (confidence: {confidence:.2f})")
                    print("-" * 40)
        
        cv2.imshow('Gesture Test', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Print final statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print("\n" + "="*50)
    print("FINAL STATISTICS:")
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Successful detections: {detection_count}")
    print(f"Detection rate: {(detection_count / frame_count) * 100:.1f}%")
    print("="*50)
    
    cap.release()
    cv2.destroyAllWindows()

def test_model_performance():
    """Test model performance with sample data"""
    controller = AccessibilityController()
    
    if controller.model is None:
        print("No model available. Please train a model first.")
        return
    
    print("Testing Model Performance")
    print("========================")
    
    # Generate test data
    test_samples = 100
    print(f"Generating {test_samples} test samples...")
    
    X_test = []
    y_test = []
    
    for i in range(test_samples):
        # Create realistic test data
        if i % 4 == 0:
            # "none" gesture - more random
            keypoints = np.random.normal(0.5, 0.3, 63)
            y_test.append(0)  # Add label for "none" gesture
        else:
            # Actual gesture - more structured
            gesture_idx = (i % (len(controller.gestures) - 1)) + 1  # Skip "none"
            keypoints = np.random.normal(0.5, 0.15, 63)
            y_test.append(gesture_idx)
        
        X_test.append(keypoints)
    
    X_test = np.array(X_test)
    
    # Predict
    print("Running predictions...")
    predictions = controller.model.predict(X_test, verbose=0)
    
    # Calculate accuracy for non-"none" gestures
    correct = 0
    total = len(y_test)
    
    for i, true_label in enumerate(y_test):
        predicted_label = np.argmax(predictions[i])
        if predicted_label == true_label:
            correct += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"Test accuracy: {accuracy:.1f}%")
    print(f"Correct: {correct}/{total}")
    
    # Show confidence distribution
    confidences = np.max(predictions, axis=1)
    print(f"Average confidence: {np.mean(confidences):.3f}")
    print(f"Confidence std: {np.std(confidences):.3f}")
    print(f"Min confidence: {np.min(confidences):.3f}")
    print(f"Max confidence: {np.max(confidences):.3f}")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Live camera testing")
    print("2. Model performance testing")
    
    try:
        choice = int(input("Enter choice (1 or 2): "))
        if choice == 1:
            test_gesture_recognition()
        elif choice == 2:
            test_model_performance()
        else:
            print("Invalid choice")
    except ValueError:
        print("Please enter a valid number")