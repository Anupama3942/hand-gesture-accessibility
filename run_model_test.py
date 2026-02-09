
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from test_gestures import test_model_performance

if __name__ == "__main__":
    try:
        test_model_performance()
        print("Model performance test execution completed.")
    except Exception as e:
        print(f"Error running model test: {e}")
