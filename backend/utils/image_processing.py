"""
Image Processing Utilities for License Plate Recognition
"""

import cv2
import numpy as np
from PIL import Image


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model input
    
    Args:
        image: Input image (BGR or RGB)
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed image
    """
    # Resize image
    resized = cv2.resize(image, target_size)
    
    # Normalize pixel values to [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    return normalized


def extract_plate_region(image, bbox):
    """
    Extract plate region from image using bounding box
    
    Args:
        image: Input image
        bbox: Bounding box [x, y, width, height]
    
    Returns:
        Extracted plate region
    """
    x, y, w, h = bbox
    
    # Ensure coordinates are within image bounds
    h_img, w_img = image.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    
    # Extract region
    plate_region = image[y:y+h, x:x+w]
    
    return plate_region


def enhance_plate_image(plate_image):
    """
    Enhance plate image for better character recognition
    
    Args:
        plate_image: Plate region image
    
    Returns:
        Enhanced plate image
    """
    # Convert to grayscale if needed
    if len(plate_image.shape) == 3:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_image
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Sharpen
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened


def apply_perspective_transform(image, points):
    """
    Apply perspective transformation to correct plate angle
    
    Args:
        image: Input image
        points: Four corner points of the plate
    
    Returns:
        Transformed image
    """
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = order_points(points)
    (tl, tr, br, bl) = rect
    
    # Compute width of the new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    # Compute height of the new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype='float32')
    
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Apply transformation
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    return warped


def order_points(pts):
    """
    Order points in clockwise order starting from top-left
    
    Args:
        pts: Array of 4 points
    
    Returns:
        Ordered points
    """
    rect = np.zeros((4, 2), dtype='float32')
    
    # Sum and difference for ordering
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    # Top-left will have smallest sum, bottom-right will have largest
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right will have smallest difference, bottom-left will have largest
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def augment_image(image, augmentation_params=None):
    """
    Apply data augmentation to image
    
    Args:
        image: Input image
        augmentation_params: Dictionary of augmentation parameters
    
    Returns:
        Augmented image
    """
    if augmentation_params is None:
        augmentation_params = {
            'rotation': 5,
            'brightness': 30,
            'contrast': 0.2,
            'noise': 5
        }
    
    augmented = image.copy()
    
    # Random rotation
    if 'rotation' in augmentation_params:
        angle = np.random.uniform(-augmentation_params['rotation'], 
                                   augmentation_params['rotation'])
        h, w = augmented.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, M, (w, h))
    
    # Random brightness
    if 'brightness' in augmentation_params:
        value = np.random.randint(-augmentation_params['brightness'],
                                   augmentation_params['brightness'])
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        augmented = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    # Random contrast
    if 'contrast' in augmentation_params:
        alpha = 1.0 + np.random.uniform(-augmentation_params['contrast'],
                                         augmentation_params['contrast'])
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=0)
    
    # Random noise
    if 'noise' in augmentation_params:
        noise = np.random.randint(0, augmentation_params['noise'],
                                   augmented.shape, dtype='uint8')
        augmented = cv2.add(augmented, noise)
    
    return augmented


def resize_with_aspect_ratio(image, target_width=None, target_height=None):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        target_width: Target width (optional)
        target_height: Target height (optional)
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if target_width is None and target_height is None:
        return image
    
    if target_width is None:
        # Calculate width based on height
        ratio = target_height / h
        target_width = int(w * ratio)
    elif target_height is None:
        # Calculate height based on width
        ratio = target_width / w
        target_height = int(h * ratio)
    
    resized = cv2.resize(image, (target_width, target_height), 
                         interpolation=cv2.INTER_AREA)
    
    return resized


def draw_bounding_box(image, bbox, label=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box on image
    
    Args:
        image: Input image
        bbox: Bounding box [x, y, width, height]
        label: Optional label text
        color: Box color (BGR)
        thickness: Line thickness
    
    Returns:
        Image with bounding box
    """
    result = image.copy()
    x, y, w, h = bbox
    
    # Draw rectangle
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label if provided
    if label is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(result, 
                     (x, y - text_height - 10),
                     (x + text_width, y),
                     color, -1)
        
        # Draw text
        cv2.putText(result, label, (x, y - 5), font, font_scale,
                   (255, 255, 255), font_thickness)
    
    return result


def convert_to_grayscale(image):
    """
    Convert image to grayscale
    
    Args:
        image: Input image (BGR or RGB)
    
    Returns:
        Grayscale image
    """
    if len(image.shape) == 2:
        # Already grayscale
        return image
    elif len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invalid image shape")


def normalize_image(image):
    """
    Normalize image pixel values to [0, 1]
    
    Args:
        image: Input image
    
    Returns:
        Normalized image
    """
    return image.astype('float32') / 255.0
