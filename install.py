# install.py
#!/usr/bin/env python3
import subprocess
import sys
import os
from logging_config import logger

def install_requirements():
    """Install project requirements"""
    try:
        logger.info("Installing requirements...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Requirements installed successfully")
            return True
        else:
            logger.error(f"Failed to install requirements: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Installation error: {e}")
        return False

def setup_environment():
    """Setup environment and directories"""
    try:
        logger.info("Setting up environment...")
        
        # Create necessary directories
        directories = ['models', 'training_data', 'logs']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Create .env file if it doesn't exist
        if not os.path.exists('.env'):
            with open('.env.example', 'r') as src, open('.env', 'w') as dst:
                dst.write(src.read())
            logger.info("Created .env file from example")
        
        logger.info("Environment setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Environment setup error: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting Accessibility System Installation...")
    
    if install_requirements() and setup_environment():
        logger.info("Installation completed successfully!")
        print("\nNext steps:")
        print("1. Review and modify .env file if needed")
        print("2. Run: python run_app.py")
        print("3. Open http://localhost:5000 in your browser")
    else:
        logger.error("Installation failed!")
        sys.exit(1)