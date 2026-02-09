
import unittest
import os
import json
import sys
from datetime import datetime
from unittest.mock import patch, mock_open, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accessibility_utils import AccessibilityLogger, CalibrationSystem, PerformanceMonitor, AccessibilityPresets

class TestAccessiblityLogger(unittest.TestCase):
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=False)
    @patch('json.dump')
    def test_log_event(self, mock_json_dump, mock_exists, mock_file):
        logger = AccessibilityLogger()
        logger.log_event("TEST_EVENT", {"info": "test"})
        
        # Verify it tries to write to file
        mock_file.assert_called()
        args, _ = mock_json_dump.call_args
        self.assertEqual(args[0][0]['type'], "TEST_EVENT")

class TestPerformanceMonitor(unittest.TestCase):
    def test_fps_calculation(self):
        monitor = PerformanceMonitor()
        monitor.update_fps(30)
        monitor.update_fps(60)
        
        stats = monitor.get_stats()
        self.assertEqual(stats['fps'], 45.0)
        self.assertEqual(stats['fps_min'], 30)
        self.assertEqual(stats['fps_max'], 60)
        
    def test_history_limit(self):
        monitor = PerformanceMonitor()
        monitor.max_history_size = 5
        for i in range(10):
            monitor.update_fps(i)
            
        self.assertEqual(len(monitor.fps_history), 5)
        self.assertEqual(monitor.fps_history[-1], 9)

class TestAccessibilityPresets(unittest.TestCase):
    @patch('os.path.exists', return_value=False)
    def test_default_presets(self, mock_exists):
        presets = AccessibilityPresets()
        self.assertIn('default', presets.presets)
        self.assertIn('gaming', presets.presets)
        
    def test_apply_preset(self):
        presets = AccessibilityPresets()
        settings = {}
        success = presets.apply_preset('default', settings)
        self.assertTrue(success)
        self.assertEqual(settings['sensitivity'], 1.0)
        
    def test_apply_invalid_preset(self):
        presets = AccessibilityPresets()
        settings = {}
        success = presets.apply_preset('non_existent', settings)
        self.assertFalse(success)
        self.assertEqual(settings, {})

if __name__ == '__main__':
    unittest.main()
