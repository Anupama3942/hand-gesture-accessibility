#!/usr/bin/env python3
import numpy as np
import os
import pickle
from accessibility_controller import AccessibilityController

def check_training_data():
    """Check the quality of training data"""
    controller = AccessibilityController()
    
    print("Training Data Quality Check")
    print("==========================")
    
    if not controller.training_data:
        print("No training data found!")
        return
    
    total_samples = sum(len(samples) for samples in controller.training_data.values())
    print(f"Total samples: {total_samples}")
    
    for gesture, samples in controller.training_data.items():
        print(f"\n{gesture}: {len(samples)} samples")
        
        if samples:
            # Check sample quality
            valid_samples = 0
            invalid_samples = 0
            
            for sample in samples:
                keypoints = sample['keypoints']
                if np.any(keypoints) and not np.all(keypoints == 0):
                    valid_samples += 1
                else:
                    invalid_samples += 1
            
            print(f"  Valid samples: {valid_samples}")
            print(f"  Invalid samples (all zeros): {invalid_samples}")
            
            if valid_samples > 0:
                # Analyze keypoint ranges
                all_keypoints = np.array([s['keypoints'] for s in samples if np.any(s['keypoints'])])
                print(f"  Keypoints range: [{np.min(all_keypoints):.3f}, {np.max(all_keypoints):.3f}]")
                print(f"  Keypoints mean: {np.mean(all_keypoints):.3f}")
                print(f"  Keypoints std: {np.std(all_keypoints):.3f}")

if __name__ == "__main__":
    check_training_data()