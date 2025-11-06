# Quick Start Guide

Get your License Plate Recognition system up and running in minutes!

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- pip and npm installed

## Step 1: Clone and Navigate

```bash
cd /path/to/FINAL_PROJECT
```

## Step 2: Set Up Backend

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

## Step 3: Set Up Frontend

```bash
cd frontend
npm install
cd ..
```

## Step 4: Start Backend Server

```bash
cd backend
python app.py
```

The backend will start on `http://localhost:5000`

## Step 5: Start Frontend (In a New Terminal)

```bash
cd frontend
npm start
```

The frontend will open automatically on `http://localhost:3000`

## Step 6: Test the Application

1. Open your browser to `http://localhost:3000`
2. Upload an image with a license plate
3. Click "Recognize Plate"
4. View the results!

## Optional: Train Your Own Models

If you want to train the models with your own data:

```bash
# Train plate detector
python backend/training/train_detector.py --use-sample-data --epochs 10

# Train character recognizer
python backend/training/train_ocr.py --use-sample-data --epochs 20
```

## Troubleshooting

### Backend won't start
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 5000 is already in use

### Frontend won't start
- Make sure all dependencies are installed: `npm install`
- Check if port 3000 is already in use
- Try deleting `node_modules` and running `npm install` again

### "No license plate detected"
- Make sure the image has a clear license plate
- Try an image with better lighting
- The model works best with Indian-style license plates

### Import errors
- Make sure you're in the virtual environment
- Reinstall requirements: `pip install -r requirements.txt --upgrade`

## Features to Try

- **Drag & Drop**: Drag images directly onto the upload area
- **Real-time Processing**: See results in seconds
- **Confidence Score**: View how confident the model is
- **Bounding Box**: See exact plate location

## Next Steps

1. **Add Your Data**: Place your images in `data/raw/images/`
2. **Train Models**: Use your data to improve accuracy
3. **Customize UI**: Edit `frontend/src/App.js` for UI changes
4. **Deploy**: Follow deployment instructions in README.md

## Need Help?

- Check the main README.md for detailed documentation
- Review the code comments for implementation details
- Open an issue on GitHub for questions

Happy coding! 🚀
