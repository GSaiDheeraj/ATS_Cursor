#!/usr/bin/env python3
"""
Setup script for ATS Resume Analyzer
This script sets up the environment and installs required dependencies
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🔧 {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'templates',
        'static/css',
        'static/js',
        'uploads',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def install_requirements():
    """Install Python requirements"""
    print("\n📦 Installing Python requirements...")
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ pip is not available. Please install pip first.")
        return False
    
    # Install requirements
    requirements_files = ['requirements_ats.txt']
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            command = f"{sys.executable} -m pip install -r {req_file}"
            if not run_command(command, f"Installing requirements from {req_file}"):
                return False
        else:
            print(f"⚠️  Requirements file {req_file} not found")
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded successfully")
        return True
    except ImportError:
        print("⚠️  NLTK not installed yet. NLTK data will be downloaded on first run.")
        return True
    except Exception as e:
        print(f"⚠️  Error downloading NLTK data: {e}")
        return True

def create_env_file():
    """Create a sample .env file"""
    env_content = """# ATS Resume Analyzer Configuration
# Copy this file to .env and update with your settings

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# File Upload Settings
MAX_CONTENT_LENGTH=16777216
UPLOAD_FOLDER=uploads

# Google Cloud Settings (if using Vertex AI)
PROJECT_ID=your-project-id
LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
MODEL_ID=gemini-pro

# API Keys (if needed)
API_KEY=your-api-key-here
API_URL=https://api.example.com

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/ats_analyzer.log
"""
    
    if not os.path.exists('.env'):
        with open('.env.example', 'w') as f:
            f.write(env_content)
        print("✅ Created .env.example file")
        print("📝 Copy .env.example to .env and update with your settings")
    else:
        print("✅ .env file already exists")

def test_installation():
    """Test if the installation was successful"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import flask
        import sklearn
        import nltk
        import numpy
        import pandas
        
        print("✅ All core dependencies imported successfully")
        
        # Test Flask app creation
        from flask import Flask
        test_app = Flask(__name__)
        print("✅ Flask app can be created")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def print_next_steps():
    """Print instructions for running the application"""
    print(f"\n{'='*60}")
    print("🎉 ATS Resume Analyzer Setup Complete!")
    print(f"{'='*60}")
    print("\n📋 Next Steps:")
    print("1. Copy .env.example to .env and update with your settings")
    print("2. Run the application:")
    print("   python app_ats.py")
    print("\n3. Open your browser and go to:")
    print("   http://localhost:5000")
    print("\n📚 Additional Information:")
    print("• Upload resumes in PDF, TXT, DOC, or DOCX format")
    print("• Maximum file size: 16MB")
    print("• The system uses machine learning for ATS analysis")
    print("• All processing is done locally on your machine")
    print("\n🔧 Troubleshooting:")
    print("• If you encounter import errors, try reinstalling requirements")
    print("• Check the logs/ directory for error logs")
    print("• Make sure Python 3.8+ is installed")
    
    if platform.system() == "Windows":
        print("\n🪟 Windows Users:")
        print("• You might need to install Visual C++ Build Tools")
        print("• Consider using conda instead of pip for some packages")
    
    print(f"\n{'='*60}")

def main():
    """Main setup function"""
    print("🚀 Setting up ATS Resume Analyzer")
    print("This will install dependencies and configure the application")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check the errors above.")
        sys.exit(1)
    
    # Download NLTK data
    download_nltk_data()
    
    # Create environment file
    create_env_file()
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed. Please check the errors above.")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()