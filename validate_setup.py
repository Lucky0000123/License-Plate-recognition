#!/usr/bin/env python3
"""
Setup Validation Script for License Plate Recognition System
Checks all dependencies and configurations
"""

import sys
import os
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_status(check_name, status, message=""):
    """Print check status"""
    status_symbol = "✓" if status else "✗"
    status_text = "PASS" if status else "FAIL"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}{status_symbol} {check_name:40s} [{status_text}]{reset}")
    if message:
        print(f"  → {message}")


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    required = (3, 8)
    
    is_valid = version >= required
    message = f"Python {version.major}.{version.minor}.{version.micro}"
    
    if not is_valid:
        message += f" (Required: >= {required[0]}.{required[1]})"
    
    return is_valid, message


def check_node_version():
    """Check Node.js version"""
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True, timeout=5)
        version = result.stdout.strip()
        
        # Extract major version
        major = int(version.replace('v', '').split('.')[0])
        is_valid = major >= 14
        
        message = f"Node.js {version}"
        if not is_valid:
            message += " (Required: >= 14.x)"
        
        return is_valid, message
    except Exception as e:
        return False, f"Node.js not found: {str(e)}"


def check_python_packages():
    """Check if Python packages are installed"""
    required_packages = [
        'tensorflow', 'keras', 'numpy', 'opencv-python', 'flask',
        'flask-cors', 'pillow', 'pandas', 'matplotlib', 'scikit-learn'
    ]
    
    missing = []
    installed = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').split('[')[0])
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    is_valid = len(missing) == 0
    message = f"{len(installed)}/{len(required_packages)} packages installed"
    
    if missing:
        message += f"\n  Missing: {', '.join(missing)}"
    
    return is_valid, message


def check_directory_structure():
    """Check if required directories exist"""
    required_dirs = [
        'backend',
        'backend/models',
        'backend/utils',
        'backend/training',
        'backend/saved_models',
        'frontend',
        'frontend/src',
        'data',
        'data/raw',
        'data/processed',
        'data/annotations',
        'tests'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
    
    is_valid = len(missing) == 0
    message = f"{len(required_dirs) - len(missing)}/{len(required_dirs)} directories found"
    
    if missing:
        message += f"\n  Missing: {', '.join(missing)}"
    
    return is_valid, message


def check_required_files():
    """Check if required files exist"""
    required_files = [
        'backend/app.py',
        'backend/models/plate_detector.py',
        'backend/models/char_recognizer.py',
        'backend/training/train_detector.py',
        'backend/training/train_ocr.py',
        'backend/utils/image_processing.py',
        'backend/utils/data_loader.py',
        'frontend/package.json',
        'frontend/src/App.js',
        'requirements.txt',
        'demo.py'
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    is_valid = len(missing) == 0
    message = f"{len(required_files) - len(missing)}/{len(required_files)} files found"
    
    if missing:
        message += f"\n  Missing: {', '.join(missing)}"
    
    return is_valid, message


def check_frontend_dependencies():
    """Check if frontend dependencies are installed"""
    node_modules = Path('frontend/node_modules')

    if not node_modules.exists():
        return False, "node_modules not found (run: cd frontend && npm install)"

    required_packages = ['react', 'react-dom', 'axios', 'tailwindcss']
    missing = []

    for package in required_packages:
        if not (node_modules / package).exists():
            missing.append(package)

    is_valid = len(missing) == 0
    message = "Frontend dependencies installed"

    if missing:
        message = f"Missing packages: {', '.join(missing)}"

    return is_valid, message


def check_virtual_environment():
    """Check if running in virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    message = "Running in virtual environment" if in_venv else "Not in virtual environment (recommended)"
    return True, message  # Warning, not error


def main():
    """Main validation function"""
    print_header("License Plate Recognition - Setup Validation")

    checks = [
        ("Python Version", check_python_version),
        ("Node.js Version", check_node_version),
        ("Virtual Environment", check_virtual_environment),
        ("Python Packages", check_python_packages),
        ("Directory Structure", check_directory_structure),
        ("Required Files", check_required_files),
        ("Frontend Dependencies", check_frontend_dependencies),
    ]

    results = []

    print("\nRunning validation checks...\n")

    for check_name, check_func in checks:
        try:
            is_valid, message = check_func()
            print_status(check_name, is_valid, message)
            results.append((check_name, is_valid))
        except Exception as e:
            print_status(check_name, False, f"Error: {str(e)}")
            results.append((check_name, False))

    # Summary
    print_header("Validation Summary")

    passed = sum(1 for _, status in results if status)
    total = len(results)

    print(f"\nPassed: {passed}/{total} checks")

    if passed == total:
        print("\n✓ All checks passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Train models: python backend/training/train_detector.py --use-sample-data")
        print("  2. Or run demo: python demo.py")
        print("  3. Start app: ./run_app.sh")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nQuick fixes:")
        print("  • Install Python packages: pip install -r requirements.txt")
        print("  • Install frontend deps: cd frontend && npm install")
        print("  • Create virtual env: python3 -m venv venv && source venv/bin/activate")
        return 1


if __name__ == '__main__':
    sys.exit(main())

