#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
from accessibility_controller import AccessibilityController
import time

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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        image, results = controller.mediapipe_detection(frame)
        image = controller.draw_landmarks(image, results)
        
        # Extract keypoints and recognize gesture
        keypoints = controller.extract_keypoints(results)
        
        if keypoints is not None:
            print(f"Keypoints shape: {keypoints.shape}")
            print(f"Keypoints range: [{np.min(keypoints):.3f}, {np.max(keypoints):.3f}]")
            print(f"Non-zero values: {np.count_nonzero(keypoints)}/{len(keypoints)}")
            
            gesture, confidence = controller.recognize_gesture(keypoints)
            
            # Display results
            cv2.putText(image, f"Gesture: {gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display hand detection status
            if results and results.multi_hand_landmarks:
                hands_text = f"Hands: {len(results.multi_hand_landmarks)}"
                cv2.putText(image, hands_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            print(f"Detected: {gesture} (confidence: {confidence:.2f}), Hands: {len(results.multi_hand_landmarks) if results and results.multi_hand_landmarks else 0}")
            
            # Display model predictions for all classes
            if controller.model is not None and np.any(keypoints):
                landmarks_input = np.expand_dims(keypoints, axis=0)
                predictions = controller.model.predict(landmarks_input, verbose=0)[0]
                
                print("All predictions:")
                y_offset = 120
                for i, (gest, pred) in enumerate(zip(controller.gestures, predictions)):
                    if pred > 0.1:  # Only show predictions above 10%
                        cv2.putText(image, f"{gest}: {pred:.3f}", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_offset += 20
                        print(f"  {gest}: {pred:.3f}")
        
        cv2.imshow('Gesture Debug', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    debug_gesture_recognition()