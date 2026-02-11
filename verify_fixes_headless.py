import os
import numpy as np
import sys
from unittest.mock import MagicMock

# Mock GUI libraries before they are imported by accessibility_controller
mock_pyautogui = MagicMock()
mock_pyautogui.size.return_value = (1920, 1080)
sys.modules['pyautogui'] = mock_pyautogui

mock_pyttsx3 = MagicMock()
sys.modules['pyttsx3'] = mock_pyttsx3

mock_pynput = MagicMock()
sys.modules['pynput'] = mock_pynput
sys.modules['pynput.keyboard'] = MagicMock()
sys.modules['pynput.mouse'] = MagicMock()

# Now try importing
try:
    from accessibility_controller import AccessibilityController
    print("Successfully imported AccessibilityController")
    
    # Initialize controller
    controller = AccessibilityController()
    print("Successfully initialized AccessibilityController")
    
    # Test recognize_gesture with a dummy landmark array
    # This should have previously failed with NotFittedError
    dummy_landmarks = np.zeros(63)
    gesture, confidence = controller.recognize_gesture(dummy_landmarks)
    print(f"recognize_gesture result: {gesture} (confidence: {confidence})")
    
    if gesture == "none":
        print("Verification SUCCESS: recognize_gesture handled unfitted scaler gracefully.")
    else:
        print(f"Verification FAILURE: Unexpected gesture result: {gesture}")

except Exception as e:
    print(f"Verification FAILED with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
