from accessibility_controller import AccessibilityController
import os
import logging
import argparse
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train Accessibility Gesture Recognition Model')
    parser.add_argument('--gesture', '-g', help='Specific gesture to train')
    parser.add_argument('--samples', '-s', type=int, default=20, help='Number of samples to collect')
    parser.add_argument('--batch', '-b', action='store_true', help='Batch train all gestures')
    parser.add_argument('--export', '-e', action='store_true', help='Export training data after collection')
    parser.add_argument('--format', '-f', choices=['pkl', 'json'], default='pkl', help='Export format')
    
    args = parser.parse_args()
    
    controller = AccessibilityController()
    
    gestures = [
        'cursor_mode', 'click', 'right_click', 'double_click', 'drag',
        'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out', 'back',
        'forward', 'volume_up', 'volume_down', 'mute', 'play_pause',
        'next_track', 'prev_track', 'copy', 'paste', 'cut',
        'undo', 'redo', 'save', 'new_tab', 'close_tab',
        'switch_app', 'desktop', 'task_view', 'help'
    ]
    
    print("Accessibility Gesture Training System")
    print("=====================================")
    print(f"Using standardized data format: {controller.get_training_status().get('format_version', '1.0')}")
    print("")
    
    if args.batch:
        print("Batch training all gestures...")
        print(f"Target samples per gesture: {args.samples}")
        print("")
        
        for gesture in gestures:
            print(f"Training: {gesture}")
            
            # Start training
            if controller.start_training(gesture):
                print(f"Perform the gesture {args.samples} times")
                print("The system will automatically capture samples")
                print("Press Ctrl+C to stop training this gesture")
                print("")
                
                try:
                    samples_collected = 0
                    while samples_collected < args.samples:
                        # Simulate sample collection (in real usage, this would capture frames)
                        sample_count = controller.capture_training_sample(gesture)
                        if sample_count > samples_collected:
                            samples_collected = sample_count
                            print(f"Collected {samples_collected}/{args.samples} samples")
                        time.sleep(0.5)  # Simulate time between samples
                        
                except KeyboardInterrupt:
                    print(f"\nStopped training for {gesture}. Collected {samples_collected} samples")
                    continue
        
        print("\nFinished batch training")
        
    elif args.gesture:
        gesture = args.gesture
        if gesture not in gestures:
            print(f"Unknown gesture: {gesture}")
            print(f"Available gestures: {', '.join(gestures)}")
            return
        
        print(f"Training gesture: {gesture}")
        print(f"Target samples: {args.samples}")
        print("")
        
        # Start training
        if controller.start_training(gesture):
            print("Perform the gesture in front of the camera.")
            print("The system will automatically capture samples.")
            print("Press Ctrl+C to stop training.")
            print("")
            
            try:
                samples_collected = 0
                while samples_collected < args.samples:
                    sample_count = controller.capture_training_sample(gesture)
                    if sample_count > samples_collected:
                        samples_collected = sample_count
                        print(f"Collected {samples_collected}/{args.samples} samples")
                    time.sleep(0.5)
                    
            except KeyboardInterrupt:
                print(f"\nTraining stopped. Collected {samples_collected} samples")
    
    else:
        print("Please use the web interface for interactive training:")
        print("1. Run: python run_app.py")
        print("2. Open http://localhost:5000/training in your browser")
        print("3. Follow the instructions to train gestures")
        print("")
        print("Or use command line options:")
        print("  --gesture GESTURE_NAME   Train a specific gesture")
        print("  --batch                  Batch train all gestures")
        print("  --samples NUMBER         Number of samples to collect")
        print("  --export                 Export training data")
        print("  --format FORMAT          Export format (pkl or json)")
        return
    
    # Train the model with collected data
    print("\nTraining model with collected data...")
    result = controller.train_model()
    
    if result["status"] == "success":
        print(f"Model trained successfully!")
        print(f"Accuracy: {result['accuracy']:.2%}")
        print(f"Validation Accuracy: {result['val_accuracy']:.2%}")
        print(f"Loss: {result['loss']:.4f}")
        
        # Save model
        if controller.save_model():
            print("Model saved successfully!")
        else:
            print("Failed to save model")
    else:
        print(f"Training failed: {result['message']}")
    
    # Export data if requested
    if args.export:
        export_path = controller.export_training_data(args.format)
        if export_path:
            print(f"Training data exported to {export_path}")
        else:
            print("Failed to export training data")

if __name__ == "__main__":
    main()