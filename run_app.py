#!/usr/bin/env python3
import os
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings and logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Even more aggressive suppression
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# Filter out specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Now import and run your app
from accessibility_web import app

if __name__ == '__main__':
    print("Starting Accessibility Control System...")
    print("Server will be available at http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('training_data', exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
