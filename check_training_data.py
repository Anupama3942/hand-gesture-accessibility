#!/usr/bin/env python3
import numpy as np
import os
import pickle
import json
from accessibility_controller import AccessibilityController
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_training_data():
    """Check the quality of training data using standardized format"""
    controller = AccessibilityController()
    
    print("Training Data Quality Check")
    print("==========================")
    print(f"Data Format Version: {controller.get_training_status().get('format_version', 'unknown')}")
    print("")
    
    if not controller.training_data:
        print("No training data found!")
        return
    
    total_samples = sum(len(samples) for samples in controller.training_data.values())
    print(f"Total samples: {total_samples}")
    print(f"Gestures trained: {len(controller.training_data)}")
    print("")
    
    for gesture, samples in controller.training_data.items():
        print(f"{gesture}: {len(samples)} samples")
        
        if samples:
            # Check sample quality
            valid_samples = 0
            invalid_samples = 0
            sample_qualities = []
            
            for sample in samples:
                keypoints = sample['keypoints']
                if controller.validate_training_sample(keypoints):
                    valid_samples += 1
                    # Calculate quality metrics
                    quality = np.std(keypoints)  # Higher std = more variation = better
                    sample_qualities.append(quality)
                else:
                    invalid_samples += 1
            
            print(f"  Valid samples: {valid_samples}")
            print(f"  Invalid samples: {invalid_samples}")
            
            if valid_samples > 0:
                # Analyze keypoint ranges and quality
                all_keypoints = np.array([s['keypoints'] for s in samples if controller.validate_training_sample(s['keypoints'])])
                print(f"  Keypoints range: [{np.min(all_keypoints):.3f}, {np.max(all_keypoints):.3f}]")
                print(f"  Keypoints mean: {np.mean(all_keypoints):.3f}")
                print(f"  Keypoints std: {np.std(all_keypoints):.3f}")
                print(f"  Sample quality (avg std): {np.mean(sample_qualities):.3f}")
                
                # Check for timestamp consistency
                timestamps = [s.get('timestamp') for s in samples if 'timestamp' in s]
                if timestamps:
                    print(f"  Latest sample: {max(timestamps)}")
            
            print("")

def export_training_report():
    """Export a detailed training data report"""
    controller = AccessibilityController()
    
    report = {
        "generated_at": np.datetime64('now').astype(str),
        "format_version": controller.get_training_status().get('format_version', 'unknown'),
        "total_samples": sum(len(samples) for samples in controller.training_data.values()),
        "gestures_trained": [],
        "quality_metrics": {}
    }
    
    for gesture, samples in controller.training_data.items():
        gesture_report = {
            "sample_count": len(samples),
            "valid_samples": 0,
            "invalid_samples": 0,
            "keypoint_stats": {},
            "quality_metrics": {}
        }
        
        valid_keypoints = []
        sample_qualities = []
        
        for sample in samples:
            if controller.validate_training_sample(sample['keypoints']):
                gesture_report["valid_samples"] += 1
                valid_keypoints.append(sample['keypoints'])
                sample_qualities.append(np.std(sample['keypoints']))
            else:
                gesture_report["invalid_samples"] += 1
        
        if valid_keypoints:
            valid_keypoints = np.array(valid_keypoints)
            gesture_report["keypoint_stats"] = {
                "min": float(np.min(valid_keypoints)),
                "max": float(np.max(valid_keypoints)),
                "mean": float(np.mean(valid_keypoints)),
                "std": float(np.std(valid_keypoints))
            }
            gesture_report["quality_metrics"] = {
                "avg_quality": float(np.mean(sample_qualities)),
                "quality_std": float(np.std(sample_qualities))
            }
        
        report["gestures_trained"].append({
            "gesture": gesture,
            **gesture_report
        })
    
    # Save report
    report_path = os.path.join('training_data', 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Training report saved to {report_path}")
    return report

if __name__ == "__main__":
    check_training_data()
    print("\n" + "="*50 + "\n")
    export_training_report()