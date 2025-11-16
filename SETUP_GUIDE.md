# Complete Setup Guide

This guide will help you set up and run the License Plate Recognition system.

## 🚀 Quick Start (Recommended)

The fastest way to get started:

```bash
# Run the automated setup script
./quick_setup.sh
```

This script will:
- ✅ Check all prerequisites
- ✅ Create virtual environment
- ✅ Install all dependencies (Python + Node.js)
- ✅ Generate sample test images
- ✅ Validate the setup

## 📋 Manual Setup

If you prefer manual setup or the quick setup fails:

### Step 1: Prerequisites

Ensure you have:
- Python 3.8+ installed
- Node.js 14+ installed
- pip and npm installed

### Step 2: Backend Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate  # On Windows

# Install Python dependencies
pip install -r requirements.txt
```

### Step 3: Frontend Setup

```bash
cd frontend
npm install
cd ..
```

### Step 4: Validate Setup

```bash
python3 validate_setup.py
```

## 🧪 Testing the System

### Option 1: Generate Sample Images

```bash
python3 create_sample_images.py
```

This creates synthetic license plate images in `sample_images/` directory.

### Option 2: Run Demo

```bash
python demo.py
```

This runs a complete demo with synthetic data.

### Option 3: Train Models (Optional)

Train with synthetic data:

```bash
# Train plate detector
python backend/training/train_detector.py --use-sample-data --epochs 10

# Train character recognizer
python backend/training/train_ocr.py --use-sample-data --epochs 20
```

## 🏃 Running the Application

### Method 1: Using the startup script

```bash
./run_app.sh
```

This starts both backend and frontend servers.

### Method 2: Manual start

Terminal 1 (Backend):
```bash
source venv/bin/activate
cd backend
python app.py
```

Terminal 2 (Frontend):
```bash
cd frontend
npm start
```

### Method 3: Using Docker

```bash
# Build and start containers
docker-compose up --build

# Or run in background
docker-compose up -d
```

Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- Health Check: http://localhost:5000/api/health

## 📁 Project Structure

```
License-Plate-recognition/
├── backend/
│   ├── app.py                      # Flask API server
│   ├── models/
│   │   ├── plate_detector.py       # CNN for plate detection
│   │   └── char_recognizer.py      # CNN for OCR
│   ├── training/
│   │   ├── train_detector.py       # Train detector
│   │   └── train_ocr.py            # Train OCR (NEW!)
│   ├── utils/
│   │   ├── image_processing.py     # Image utilities
│   │   └── data_loader.py          # Data loading
│   └── saved_models/               # Trained models
├── frontend/
│   ├── src/
│   │   ├── App.js                  # React main component
│   │   └── index.js
│   └── package.json
├── data/
│   ├── raw/                        # Raw images
│   ├── processed/                  # Processed data
│   └── annotations/                # Annotations
├── tests/                          # Unit tests
├── sample_images/                  # Generated samples (NEW!)
├── demo.py                         # Demo script
├── validate_setup.py               # Setup validator (NEW!)
├── create_sample_images.py         # Sample generator (NEW!)
├── quick_setup.sh                  # Automated setup (NEW!)
├── run_app.sh                      # App launcher
├── docker-compose.yml              # Docker config (NEW!)
├── Dockerfile.backend              # Backend Docker (NEW!)
├── Dockerfile.frontend             # Frontend Docker (NEW!)
└── requirements.txt                # Python deps
```

## 🔧 Troubleshooting

### Backend won't start
```bash
# Check if dependencies are installed
pip list | grep tensorflow

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Frontend won't start
```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Port already in use
```bash
# Find and kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Find and kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Models not loading
The system works without pre-trained models using fallback methods (Haar Cascade + Tesseract). To train models:
```bash
python backend/training/train_detector.py --use-sample-data
python backend/training/train_ocr.py --use-sample-data
```

## 📚 Additional Resources

- [README.md](README.md) - Project overview
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment instructions
- [GET_STARTED.txt](GET_STARTED.txt) - Getting started text

## 🆘 Getting Help

If you encounter issues:
1. Run `python3 validate_setup.py` to check your setup
2. Check the troubleshooting section above
3. Review error messages carefully
4. Ensure all prerequisites are installed

## ✨ New Features Added

- ✅ Complete OCR training script (`train_ocr.py`)
- ✅ Setup validation tool (`validate_setup.py`)
- ✅ Sample image generator (`create_sample_images.py`)
- ✅ Automated setup script (`quick_setup.sh`)
- ✅ Docker support (docker-compose, Dockerfiles)
- ✅ Enhanced documentation
- ✅ pytesseract support for OCR fallback

