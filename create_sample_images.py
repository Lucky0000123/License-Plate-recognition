#!/usr/bin/env python3
"""
Create sample license plate images for testing
"""

import os
import cv2
import numpy as np
from pathlib import Path
import string


def create_license_plate_image(plate_text, size=(240, 60), bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    """
    Create a synthetic license plate image
    
    Args:
        plate_text: Text to display on plate
        size: Image size (width, height)
        bg_color: Background color (BGR)
        text_color: Text color (BGR)
    
    Returns:
        License plate image
    """
    # Create blank image
    img = np.ones((size[1], size[0], 3), dtype=np.uint8)
    img[:] = bg_color
    
    # Add border
    cv2.rectangle(img, (2, 2), (size[0]-3, size[1]-3), (0, 0, 0), 2)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Calculate text size and position
    text_size = cv2.getTextSize(plate_text, font, font_scale, thickness)[0]
    text_x = (size[0] - text_size[0]) // 2
    text_y = (size[1] + text_size[1]) // 2
    
    cv2.putText(img, plate_text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    return img


def create_car_with_plate(plate_text, img_size=(640, 480)):
    """
    Create an image of a car with license plate
    
    Args:
        plate_text: License plate text
        img_size: Image size (width, height)
    
    Returns:
        Image with car and license plate
    """
    # Create background (road/parking lot)
    img = np.random.randint(80, 120, (img_size[1], img_size[0], 3), dtype=np.uint8)
    
    # Add some texture
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Draw simple car shape
    car_color = (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200))
    
    # Car body
    car_top = img_size[1] // 3
    car_bottom = img_size[1] - 50
    car_left = img_size[0] // 4
    car_right = 3 * img_size[0] // 4
    
    cv2.rectangle(img, (car_left, car_top), (car_right, car_bottom), car_color, -1)
    cv2.rectangle(img, (car_left, car_top), (car_right, car_bottom), (0, 0, 0), 2)
    
    # Car windows
    window_color = (100, 150, 200)
    cv2.rectangle(img, (car_left + 50, car_top + 20), (car_right - 50, car_top + 80), window_color, -1)
    
    # License plate position
    plate_width = 200
    plate_height = 50
    plate_x = (img_size[0] - plate_width) // 2
    plate_y = car_bottom - 80
    
    # Create and place license plate
    plate_img = create_license_plate_image(plate_text, (plate_width, plate_height))
    img[plate_y:plate_y+plate_height, plate_x:plate_x+plate_width] = plate_img
    
    return img


def generate_indian_plate_number():
    """Generate random Indian license plate number"""
    states = ['MH', 'DL', 'KA', 'TN', 'UP', 'GJ', 'RJ', 'MP', 'HR', 'PB', 'WB', 'AP', 'TS', 'KL']
    state = np.random.choice(states)
    district = f"{np.random.randint(1, 99):02d}"
    series = ''.join(np.random.choice(list(string.ascii_uppercase), 2))
    number = f"{np.random.randint(1, 9999):04d}"
    
    return f"{state}{district}{series}{number}"


def main():
    """Generate sample images"""
    print("=" * 60)
    print("Creating Sample License Plate Images")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("sample_images")
    output_dir.mkdir(exist_ok=True)
    
    # Generate different types of samples
    num_samples = 10
    
    print(f"\nGenerating {num_samples} sample images...")
    
    for i in range(num_samples):
        plate_number = generate_indian_plate_number()
        
        # Create car with plate
        img = create_car_with_plate(plate_number)
        
        # Add some variations
        if i % 3 == 0:
            # Add slight rotation
            angle = np.random.uniform(-3, 3)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
        
        if i % 4 == 0:
            # Add brightness variation
            brightness = np.random.randint(-30, 30)
            img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)
        
        # Save image
        filename = output_dir / f"sample_{i+1:02d}_{plate_number}.jpg"
        cv2.imwrite(str(filename), img)
        print(f"  ✓ Created: {filename.name}")
    
    # Also create just plate images
    print(f"\nGenerating {num_samples} plate-only images...")
    plates_dir = output_dir / "plates_only"
    plates_dir.mkdir(exist_ok=True)
    
    for i in range(num_samples):
        plate_number = generate_indian_plate_number()
        plate_img = create_license_plate_image(plate_number, (240, 60))
        
        filename = plates_dir / f"plate_{i+1:02d}_{plate_number}.jpg"
        cv2.imwrite(str(filename), plate_img)
        print(f"  ✓ Created: {filename.name}")
    
    print("\n" + "=" * 60)
    print("Sample images created successfully!")
    print("=" * 60)
    print(f"\nLocation: {output_dir.absolute()}")
    print(f"  - Full images: {num_samples} files")
    print(f"  - Plate only: {num_samples} files in 'plates_only' folder")
    print("\nYou can now test the system with these images!")


if __name__ == '__main__':
    main()

