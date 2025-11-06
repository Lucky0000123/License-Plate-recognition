"""
Unit tests for License Plate Recognition models
"""

import sys
import os
import numpy as np
import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from models.plate_detector import PlateDetector
from models.char_recognizer import CharacterRecognizer


class TestPlateDetector:
    """Test cases for PlateDetector"""
    
    def test_init(self):
        """Test detector initialization"""
        detector = PlateDetector()
        assert detector.input_shape == (224, 224, 3)
        assert detector.model is None
    
    def test_build_model(self):
        """Test model building"""
        detector = PlateDetector()
        model = detector.build_model()
        assert model is not None
        assert len(model.layers) > 0
    
    def test_detect_with_sample_image(self):
        """Test detection with sample image"""
        detector = PlateDetector()
        
        # Create sample image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Should return bbox or None
        result = detector.detect(image)
        assert result is None or isinstance(result, tuple)


class TestCharacterRecognizer:
    """Test cases for CharacterRecognizer"""
    
    def test_init(self):
        """Test recognizer initialization"""
        recognizer = CharacterRecognizer()
        assert recognizer.img_height == 32
        assert recognizer.img_width == 128
        assert recognizer.model is None
        assert len(recognizer.CHARACTERS) == 36  # A-Z + 0-9
    
    def test_build_model(self):
        """Test model building"""
        recognizer = CharacterRecognizer()
        model = recognizer.build_model()
        assert model is not None
        assert len(model.layers) > 0
    
    def test_preprocess_plate(self):
        """Test plate preprocessing"""
        recognizer = CharacterRecognizer()
        
        # Create sample plate image
        plate_image = np.random.randint(0, 255, (50, 150, 3), dtype=np.uint8)
        
        # Preprocess
        processed = recognizer.preprocess_plate(plate_image)
        
        # Check shape
        assert processed.shape == (32, 128, 1)
        
        # Check normalization
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0
    
    def test_recognize_returns_tuple(self):
        """Test recognition returns proper format"""
        recognizer = CharacterRecognizer()
        
        # Create sample plate
        plate_image = np.random.randint(0, 255, (50, 150, 3), dtype=np.uint8)
        
        # Recognize
        result = recognizer.recognize(plate_image)
        
        # Should return tuple of (text, confidence)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], (float, np.floating))


class TestImageProcessing:
    """Test cases for image processing utilities"""
    
    def test_preprocess_image(self):
        """Test image preprocessing"""
        from utils.image_processing import preprocess_image
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = preprocess_image(image, target_size=(224, 224))
        
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.float32
        assert 0 <= processed.min() <= 1
        assert 0 <= processed.max() <= 1
    
    def test_extract_plate_region(self):
        """Test plate region extraction"""
        from utils.image_processing import extract_plate_region
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 50)
        
        plate = extract_plate_region(image, bbox)
        
        assert plate.shape[0] == 50  # height
        assert plate.shape[1] == 200  # width


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
