#!/usr/bin/env python3
import cv2
import numpy as np
from accessibility_controller import AccessibilityController
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_gesture_recognition():
    """Debug gesture recognition with detailed output"""
    controller = AccessibilityController()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Debugging gesture recognition. Press 'q' to quit.")
    print("Make gestures in front of the camera to see recognition results.")
    print("")
    
    frame_count = 0
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
        
        if keypoints is not None:
            print(f"Frame: {frame_count}, FPS: {fps:.1f}")
            print(f"Keypoints shape: {keypoints.shape}")
            print(f"Keypoints range: [{np.min(keypoints):.3f}, {np.max(keypoints):.3f}]")
            print(f"Non-zero values: {np.count_nonzero(keypoints)}/{len(keypoints)}")
            
            gesture, confidence = controller.recognize_gesture(keypoints)
            
            # Display results
            cv2.putText(image, f"Gesture: {gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"FPS: {fps:.1f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display hand detection status
            if results and results.multi_hand_landmarks:
                hands_text = f"Hands: {len(results.multi_hand_landmarks)}"
                cv2.putText(image, hands_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            print(f"Detected: {gesture} (confidence: {confidence:.2f}), Hands: {len(results.multi_hand_landmarks) if results and results.multi_hand_landmarks else 0}")
            
            # Display model predictions for all classes
            if controller.model is not None and np.any(keypoints):
                landmarks_input = np.expand_dims(keypoints, axis=0)
                predictions = controller.model.predict(landmarks_input, verbose=0)[0]
                
                print("Top predictions:")
                y_offset = 150
                # Get top 5 predictions
                top_indices = np.argsort(predictions)[-5:][::-1]
                
                for i, idx in enumerate(top_indices):
                    gest = controller.gestures[idx]
                    pred = predictions[idx]
                    if pred > 0.01:  # Only show predictions above 1%
                        color = (0, 255, 0) if idx == np.argmax(predictions) else (255, 255, 0)
                        cv2.putText(image, f"{gest}: {pred:.3f}", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        y_offset += 20
                        print(f"  {gest}: {pred:.3f}")
            
            print("-" * 50)
        
        cv2.imshow('Gesture Debug', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def test_individual_gestures():
    """Test recognition of individual gestures"""
    controller = AccessibilityController()
    
    print("Testing Individual Gestures")
    print("===========================")
    print("This will test each gesture's recognition")
    print("")
    
    # Create test keypoints for each gesture (simplified)
    test_gestures = ['click', 'scroll_up', 'cursor_mode', 'zoom_in']
    
    for gesture in test_gestures:
        print(f"Testing: {gesture}")
        
        # Create dummy keypoints (in real usage, these would come from actual gestures)
        if gesture == 'click':
            # Simulated click gesture keypoints
            keypoints = np.random.normal(0.6, 0.1, 63)
        elif gesture == 'scroll_up':
            keypoints = np.random.normal(0.5, 0.2, 63)
        elif gesture == 'cursor_mode':
            keypoints = np.random.normal(0.7, 0.15, 63)
        else:
            keypoints = np.random.normal(0.4, 0.25, 63)
        
        # Ensure keypoints are valid
        keypoints = np.clip(keypoints, 0, 1)
        
        # Recognize gesture
        detected_gesture, confidence = controller.recognize_gesture(keypoints)
        
        print(f"  Expected: {gesture}")
        print(f"  Detected: {detected_gesture}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Match: {'✓' if detected_gesture == gesture else '✗'}")
        print("")

if __name__ == "__main__":
    print("Choose debug mode:")
    print("1. Live camera debugging")
    print("2. Individual gesture testing")
    
    try:
        choice = int(input("Enter choice (1 or 2): "))
        if choice == 1:
            debug_gesture_recognition()
        elif choice == 2:
            test_individual_gestures()
        else:
            print("Invalid choice")
    except ValueError:
        print("Please enter a valid number")