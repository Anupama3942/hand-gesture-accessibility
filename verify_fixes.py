
import unittest
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

# Import test modules
from tests.test_config import TestSystemConfig, TestConfigManager
from tests.test_utils import TestAccessiblityLogger, TestPerformanceMonitor, TestAccessibilityPresets

def run_verification():
    print("Running verification suite...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestSystemConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigManager))
    suite.addTests(loader.loadTestsFromTestCase(TestAccessiblityLogger))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestAccessibilityPresets))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ Verification SUCCESS: All tests passed.")
        return 0
    else:
        print("\n❌ Verification FAILED: Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(run_verification())
