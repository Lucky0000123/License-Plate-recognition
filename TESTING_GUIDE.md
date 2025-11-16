# Testing Guide

This guide covers all testing aspects of the License Plate Recognition system.

## 🧪 Running Tests

### Prerequisites

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

### Run Individual Test Classes

```bash
# Test PlateDetector
pytest tests/test_models.py::TestPlateDetector -v

# Test CharacterRecognizer
pytest tests/test_models.py::TestCharacterRecognizer -v

# Test Image Processing
pytest tests/test_models.py::TestImageProcessing -v
```

## 🎯 Test Coverage

Current test coverage includes:

### Model Tests (`tests/test_models.py`)

1. **PlateDetector Tests**
   - Initialization
   - Model building
   - Detection with sample images

2. **CharacterRecognizer Tests**
   - Initialization
   - Model building
   - Plate preprocessing
   - Recognition output format

3. **Image Processing Tests**
   - Image preprocessing
   - Plate region extraction

## 🔍 Manual Testing

### 1. Demo Script Test

```bash
python demo.py
```

Expected output:
- Creates sample plate image
- Detects plate region
- Recognizes characters
- Saves results to `demo_output/`

### 2. API Testing

Start the backend:
```bash
cd backend
python app.py
```

Test endpoints:

**Health Check:**
```bash
curl http://localhost:5000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "message": "License Plate Recognition API is running",
  "models_loaded": {
    "plate_detector": false,
    "char_recognizer": false
  }
}
```

**Predict with Base64 Image:**
```bash
# Create test image and convert to base64
python3 create_sample_images.py

# Use curl or Postman to send POST request to /api/predict
# with JSON body: {"image": "data:image/jpeg;base64,..."}
```

### 3. Frontend Testing

Start the frontend:
```bash
cd frontend
npm start
```

Manual tests:
1. ✅ Upload image button works
2. ✅ Image preview displays
3. ✅ Recognize button triggers API call
4. ✅ Results display correctly
5. ✅ Error messages show for invalid inputs
6. ✅ Clear button resets state

### 4. Integration Testing

Full system test:
```bash
# Terminal 1: Start backend
cd backend && python app.py

# Terminal 2: Start frontend
cd frontend && npm start

# Terminal 3: Generate test images
python3 create_sample_images.py
```

Then:
1. Open http://localhost:3000
2. Upload image from `sample_images/`
3. Click "Recognize Plate"
4. Verify results

## 🐛 Debugging Tests

### Enable Verbose Output

```bash
pytest tests/ -v -s
```

### Run Specific Test

```bash
pytest tests/test_models.py::TestPlateDetector::test_init -v
```

### Debug with pdb

```bash
pytest tests/ --pdb
```

### Check Test Discovery

```bash
pytest --collect-only
```

## 📊 Performance Testing

### Model Inference Speed

```python
import time
import cv2
from backend.models.plate_detector import PlateDetector

detector = PlateDetector()
image = cv2.imread('sample_images/sample_01.jpg')

start = time.time()
bbox = detector.detect(image)
end = time.time()

print(f"Detection time: {(end - start) * 1000:.2f}ms")
```

### API Load Testing

Using Apache Bench:
```bash
ab -n 100 -c 10 http://localhost:5000/api/health
```

Using Python:
```python
import requests
import time

url = "http://localhost:5000/api/health"
times = []

for i in range(100):
    start = time.time()
    response = requests.get(url)
    end = time.time()
    times.append(end - start)

print(f"Average response time: {sum(times)/len(times)*1000:.2f}ms")
```

## ✅ Test Checklist

Before deployment, ensure:

- [ ] All unit tests pass
- [ ] Demo script runs successfully
- [ ] API health check returns 200
- [ ] Frontend loads without errors
- [ ] Image upload works
- [ ] Plate detection works (with/without trained models)
- [ ] Character recognition works (with/without trained models)
- [ ] Error handling works correctly
- [ ] Docker containers build successfully
- [ ] Docker compose starts all services

## 🔧 Continuous Integration

The project includes test configurations for CI/CD:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

## 📝 Writing New Tests

Template for new tests:

```python
import pytest
from backend.models.your_module import YourClass

class TestYourClass:
    def test_initialization(self):
        """Test class initialization"""
        obj = YourClass()
        assert obj is not None
    
    def test_method(self):
        """Test specific method"""
        obj = YourClass()
        result = obj.method()
        assert result == expected_value
```

## 🆘 Troubleshooting Tests

**Import errors:**
```bash
# Add backend to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}/backend"
```

**TensorFlow warnings:**
```bash
# Suppress TF warnings
export TF_CPP_MIN_LOG_LEVEL=2
```

**GPU/CPU issues:**
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
```

