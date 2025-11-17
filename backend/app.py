"""
Flask API server for License Plate Recognition
"""

import os
import base64
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2

from models.plate_detector import PlateDetector
from models.char_recognizer import CharacterRecognizer
from utils.image_processing import preprocess_image, extract_plate_region

app = Flask(__name__)

# Configure CORS for production
# Note: Flask-CORS doesn't support wildcard subdomains, so we list specific domains
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5001",
            "https://license-plate-frontend.onrender.com",
            "https://license-plate-recognition-qzi2.onrender.com"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

# Initialize models
plate_detector = PlateDetector()
char_recognizer = CharacterRecognizer()

# Load trained models
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_models')
try:
    if os.path.exists(os.path.join(MODEL_PATH, 'plate_detector.h5')):
        plate_detector.load_model(os.path.join(MODEL_PATH, 'plate_detector.h5'))
        print("✓ Plate detector model loaded")
    else:
        print("⚠ Plate detector model not found, using untrained model")
    
    if os.path.exists(os.path.join(MODEL_PATH, 'char_recognizer.h5')):
        char_recognizer.load_model(os.path.join(MODEL_PATH, 'char_recognizer.h5'))
        print("✓ Character recognizer model loaded")
    else:
        print("⚠ Character recognizer model not found, using untrained model")
except Exception as e:
    print(f"Error loading models: {str(e)}")


@app.route('/', methods=['GET'])
def home():
    """API Home Page"""
    return jsonify({
        'success': True,
        'message': 'License Plate Recognition API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'predict': '/api/predict (POST)'
        },
        'documentation': 'https://github.com/Lucky0000123/License-Plate-recognition',
        'frontend': 'Deploy the React frontend to interact with this API'
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        'status': 'healthy',
        'message': 'License Plate Recognition API is running',
        'models_loaded': {
            'plate_detector': plate_detector.model is not None,
            'char_recognizer': char_recognizer.model is not None
        }
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict license plate from uploaded image
    
    Expected request format:
    {
        "image": "base64_encoded_image_string"
    }
    
    Returns:
    {
        "success": true,
        "plate_number": "MH12AB1234",
        "confidence": 0.95,
        "bounding_box": [x, y, width, height],
        "processing_time": 0.234
    }
    """
    try:
        # Get image from request
        if 'image' not in request.json:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Decode base64 image
        image_data = request.json['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Detect plate region
        plate_bbox = plate_detector.detect(image_array)
        
        if plate_bbox is None:
            return jsonify({
                'success': False,
                'error': 'No license plate detected in image',
                'plate_number': None,
                'confidence': 0.0
            })
        
        # Extract plate region
        x, y, w, h = plate_bbox
        plate_region = image_array[y:y+h, x:x+w]
        
        # Recognize characters
        plate_text, confidence = char_recognizer.recognize(plate_region)
        
        # Format response
        return jsonify({
            'success': True,
            'plate_number': plate_text,
            'confidence': float(confidence),
            'bounding_box': [int(x), int(y), int(w), int(h)]
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'plate_number': None
        }), 500


@app.route('/api/predict-file', methods=['POST'])
def predict_file():
    """
    Predict license plate from uploaded file
    
    Multipart form data with 'file' field
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        # Read image
        image = Image.open(file.stream)
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Detect plate
        plate_bbox = plate_detector.detect(image_array)
        
        if plate_bbox is None:
            return jsonify({
                'success': False,
                'error': 'No license plate detected',
                'plate_number': None
            })
        
        # Extract and recognize
        x, y, w, h = plate_bbox
        plate_region = image_array[y:y+h, x:x+w]
        plate_text, confidence = char_recognizer.recognize(plate_region)
        
        return jsonify({
            'success': True,
            'plate_number': plate_text,
            'confidence': float(confidence),
            'bounding_box': [int(x), int(y), int(w), int(h)]
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Create saved_models directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)

    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 5001))

    # Run Flask app
    print("=" * 50)
    print("License Plate Recognition API")
    print("=" * 50)
    print(f"Server starting on port {port}")
    print(f"Health check: http://localhost:{port}/api/health")
    print("=" * 50)

    # Use debug=False for production
    app.run(host='0.0.0.0', port=port, debug=False)
