#!/usr/bin/env python3
"""
Demo script for License Plate Recognition System
Tests the system with sample images
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from models.plate_detector import PlateDetector
from models.char_recognizer import CharacterRecognizer
from utils.image_processing import draw_bounding_box


def create_sample_plate_image():
    """Create a sample image with a license plate for testing"""
    # Create a car-like background
    img = np.ones((480, 640, 3), dtype=np.uint8) * 100
    
    # Draw a license plate region
    plate_x, plate_y = 220, 300
    plate_w, plate_h = 200, 60
    
    # White plate background
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), 
                  (255, 255, 255), -1)
    
    # Black border
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), 
                  (0, 0, 0), 2)
    
    # Add some text to simulate a license plate
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'MH12AB1234', (plate_x + 15, plate_y + 40), 
                font, 0.8, (0, 0, 0), 2)
    
    return img


def main():
    """Main demo function"""
    print("=" * 60)
    print("License Plate Recognition System - Demo")
    print("=" * 60)
    print()
    
    # Initialize models
    print("Initializing models...")
    detector = PlateDetector()
    recognizer = CharacterRecognizer()
    
    print("✓ Models initialized")
    print()
    
    # Create or load sample image
    print("Creating sample image...")
    image = create_sample_plate_image()
    
    # Save sample image
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    sample_path = output_dir / "sample_image.jpg"
    cv2.imwrite(str(sample_path), image)
    print(f"✓ Sample image saved to: {sample_path}")
    print()
    
    # Detect plate
    print("Detecting license plate...")
    bbox = detector.detect(image)
    
    if bbox is None:
        print("✗ No license plate detected")
        print()
        print("Note: This is expected if models are not trained yet.")
        print("The system uses multiple detection methods as fallback.")
        return
    
    print(f"✓ Plate detected at: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
    print()
    
    # Extract plate region
    x, y, w, h = bbox
    plate_region = image[y:y+h, x:x+w]
    
    # Save plate region
    plate_path = output_dir / "detected_plate.jpg"
    cv2.imwrite(str(plate_path), plate_region)
    print(f"✓ Plate region saved to: {plate_path}")
    print()
    
    # Recognize characters
    print("Recognizing characters...")
    plate_text, confidence = recognizer.recognize(plate_region)
    
    print(f"✓ Recognized text: {plate_text}")
    print(f"  Confidence: {confidence * 100:.1f}%")
    print()
    
    # Draw result on image
    result_image = draw_bounding_box(
        image, bbox, 
        label=f"{plate_text} ({confidence*100:.1f}%)",
        color=(0, 255, 0)
    )
    
    # Save result
    result_path = output_dir / "result.jpg"
    cv2.imwrite(str(result_path), result_image)
    print(f"✓ Result image saved to: {result_path}")
    print()
    
    # Summary
    print("=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print()
    print(f"Detected Plate: {plate_text}")
    print(f"Confidence: {confidence * 100:.1f}%")
    print(f"Bounding Box: {bbox}")
    print()
    print("Check the 'demo_output' folder for images:")
    print(f"  - {sample_path}")
    print(f"  - {plate_path}")
    print(f"  - {result_path}")
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
