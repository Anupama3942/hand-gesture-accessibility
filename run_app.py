# run_app.py - Update to handle Flask 2.3+
#!/usr/bin/env python3
import os
import warnings
import logging
import signal
import sys
from datetime import datetime

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('accessibility_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Handle numpy compatibility
try:
    import numpy as np
except ImportError:
    logger.error("NumPy is not installed. Please install it with: pip install numpy")
    sys.exit(1)

# Suppress TensorFlow warnings and logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduced from 3 to 2 to see important warnings

# Filter out specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Global variable to track running state
is_running = True

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global is_running
    logger.info("Shutdown signal received. Stopping server...")
    is_running = False
    sys.exit(0)

def cleanup_resources():
    """Clean up any resources before exit"""
    logger.info("Cleaning up resources...")
    # Add any additional cleanup logic here

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import flask
        import mediapipe
        try:
            import tensorflow
        except ImportError:
            logger.warning("TensorFlow not available - gesture recognition will be disabled")
        import cv2
        import pyautogui
        import pynput
        # pyttsx3 is optional, don't fail if not available
        try:
            import pyttsx3
        except ImportError:
            logger.warning("pyttsx3 not available - voice feedback will be disabled")
        logger.info("All dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def setup_environment():
    """Setup the application environment"""
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('training_data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Set up environment variables with defaults
    os.environ.setdefault('REQUIRE_AUTH', 'false')
    os.environ.setdefault('AUTH_USERNAME', 'admin')
    os.environ.setdefault('AUTH_PASSWORD', 'password')
    
    logger.info("Environment setup completed")

def main():
    """Main application entry point"""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Setup environment
        setup_environment()
        
        # Check dependencies
        if not check_dependencies():
            logger.error("Missing required dependencies. Please install them with: pip install -r requirements.txt")
            return 1
        
        # Now import and run your app
        from accessibility_web import app, controller
        
        logger.info("Starting Accessibility Control System...")
        logger.info("Server will be available at http://localhost:5000")
        logger.info("Press Ctrl+C to stop the server")
        
        # Check if authentication is enabled
        if os.environ.get('REQUIRE_AUTH', 'false').lower() == 'true':
            logger.info("Authentication is ENABLED")
            logger.info(f"Username: {os.environ.get('AUTH_USERNAME', 'admin')}")
        else:
            logger.info("Authentication is DISABLED")
        
        # Run the application with updated configuration
        app.run(
            host='0.0.0.0', 
            port=5000, 
            debug=False, 
            threaded=True,
            use_reloader=False
        )
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return 1
    finally:
        cleanup_resources()

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)