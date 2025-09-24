#!/usr/bin/env python3
"""
Project Initialization Script
Sets up the Handwritten Equation Solver project with sample data
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_tesseract():
    """Check if Tesseract is installed."""
    try:
        result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract OCR is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Tesseract OCR not found. Please install it:")
    print("  macOS: brew install tesseract")
    print("  Ubuntu: sudo apt-get install tesseract-ocr")
    print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    return False

def install_dependencies():
    """Install Python dependencies."""
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def generate_sample_images():
    """Generate sample equation images."""
    return run_command("python generate_samples.py", "Generating sample equation images")

def run_tests():
    """Run the test suite."""
    return run_command("python -m pytest test_solver.py -v", "Running tests")

def create_directories():
    """Create necessary directories."""
    directories = ["sample_images", "uploads", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main initialization function."""
    print("🚀 Initializing Handwritten Equation Solver Project")
    print("=" * 50)
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_tesseract():
        print("\n⚠️  Tesseract is not installed, but the project will still work with TrOCR")
        print("   You can install Tesseract later for fallback OCR support")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please check your Python environment.")
        sys.exit(1)
    
    # Generate sample images
    if not generate_sample_images():
        print("\n⚠️  Failed to generate sample images, but you can create them manually")
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Some tests failed, but the project should still work")
    
    print("\n" + "=" * 50)
    print("🎉 Project initialization complete!")
    print("\nNext steps:")
    print("1. Run the web application: streamlit run modern_solver.py")
    print("2. Or use the CLI: python cli_solver.py sample_images/equation_01.png")
    print("3. Check the README.md for more information")
    print("\nHappy equation solving! 🧮")

if __name__ == "__main__":
    main()
