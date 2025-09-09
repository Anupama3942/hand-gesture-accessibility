#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
from accessibility_controller import AccessibilityController
import time

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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        image, results = controller.mediapipe_detection(frame)
        image = controller.draw_landmarks(image, results)
        
        # Extract keypoints and recognize gesture
        keypoints = controller.extract_keypoints(results)
        
        if keypoints is not None and np.any(keypoints) and not np.all(keypoints == 0):
            gesture, confidence = controller.recognize_gesture(keypoints)
            
            # Display results
            cv2.putText(image, f"Gesture: {gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            print(f"Detected: {gesture} (confidence: {confidence:.2f})")
        
        cv2.imshow('Gesture Test', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_gesture_recognition()