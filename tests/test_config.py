
import unittest
import os
import json
import sys
from unittest.mock import patch, mock_open

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig, ConfigManager

class TestSystemConfig(unittest.TestCase):
    def test_default_values(self):
        config = SystemConfig()
        self.assertEqual(config.sensitivity, 1.0)
        self.assertEqual(config.cursor_speed, 2.0)
        self.assertTrue(config.voice_feedback)
        self.assertEqual(config.camera_width, 1280)
        
    def test_validation_valid(self):
        config = SystemConfig(sensitivity=2.0, cursor_speed=5.0)
        self.assertTrue(config.validate())
        
    def test_validation_warnings(self):
        # Should log warnings but return True
        config = SystemConfig(sensitivity=10.0)  # Too high
        self.assertTrue(config.validate())
        
    def test_from_dict(self):
        data = {"sensitivity": 3.0, "unknown_field": "ignore"}
        config = SystemConfig.from_dict(data)
        self.assertEqual(config.sensitivity, 3.0)
        self.assertFalse(hasattr(config, "unknown_field"))

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        # Reset singleton for testing
        ConfigManager._instance = None
        
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"sensitivity": 0.5}')
    def test_load_config(self, mock_file, mock_exists):
        mock_exists.return_value = True
        manager = ConfigManager()
        self.assertEqual(manager.config.sensitivity, 0.5)
        
    def test_update_config(self):
        manager = ConfigManager()
        with patch.object(manager, 'save_config', return_value=True):
            success = manager.update_config({"sensitivity": 4.0})
            self.assertTrue(success)
            self.assertEqual(manager.config.sensitivity, 4.0)

if __name__ == '__main__':
    unittest.main()
