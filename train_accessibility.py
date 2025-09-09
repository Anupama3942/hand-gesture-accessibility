from accessibility_controller import AccessibilityController
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    controller = AccessibilityController()
    
    # Gestures for accessibility control
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
    print("This script helps you train gestures for the accessibility system.")
    print("For a more interactive training experience, use the web interface.")
    print("Run: python run_app.py and navigate to /training")
    
    # Create training directory
    training_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data')
    os.makedirs(training_dir, exist_ok=True)
    
    response = input("Do you want to train a specific gesture? (y/n): ")
    
    if response.lower() == 'y':
        print("\nAvailable gestures:")
        for i, gesture in enumerate(gestures):
            print(f"{i+1}. {gesture}")
        
        try:
            choice = int(input("\nEnter the number of the gesture you want to train: ")) - 1
            if 0 <= choice < len(gestures):
                gesture = gestures[choice]
                print(f"\nTraining gesture: {gesture}")
                
                # Start training for this gesture
                if controller.start_training(gesture):
                    print("Perform the gesture in front of the camera.")
                    print("The system will automatically capture samples.")
                    print("Press Ctrl+C to stop training.")
                    
                    try:
                        while True:
                            # In a real implementation, you would capture frames here
                            # For now, we'll just simulate training
                            import time
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("\nTraining stopped.")
                        
                        # Train the model with collected data
                        print("Training model with collected data...")
                        result = controller.train_model()
                        
                        if result["status"] == "success":
                            print(f"Model trained successfully with accuracy: {result['accuracy']:.2%}")
                            if input("Save model? (y/n): ").lower() == 'y':
                                if controller.save_model():
                                    print("Model saved successfully!")
                                else:
                                    print("Failed to save model.")
                        else:
                            print(f"Training failed: {result['message']}")
                else:
                    print("Failed to start training.")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")
    else:
        print("\nPlease use the web interface for training:")
        print("1. Run: python run_app.py")
        print("2. Open http://localhost:5000/training in your browser")
        print("3. Follow the instructions to train gestures")

if __name__ == "__main__":
    main()
