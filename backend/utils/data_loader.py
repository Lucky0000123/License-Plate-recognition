"""
Data Loading and Preprocessing Utilities
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import json


class LicensePlateDataLoader:
    """
    Data loader for license plate detection and recognition datasets
    Supports multiple annotation formats (PASCAL VOC, YOLO, JSON)
    """
    
    def __init__(self, data_dir, annotation_format='pascal_voc'):
        """
        Initialize data loader
        
        Args:
            data_dir: Root directory containing images and annotations
            annotation_format: Format of annotations ('pascal_voc', 'yolo', 'json')
        """
        self.data_dir = data_dir
        self.annotation_format = annotation_format
        self.images_dir = os.path.join(data_dir, 'images')
        self.annotations_dir = os.path.join(data_dir, 'annotations')
    
    def load_pascal_voc_annotation(self, xml_file):
        """
        Load PASCAL VOC format annotation
        
        Args:
            xml_file: Path to XML annotation file
        
        Returns:
            Dictionary with image info and bounding boxes
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Image information
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Bounding boxes
        boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            label = obj.find('name').text if obj.find('name') is not None else 'license_plate'
            
            boxes.append({
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'label': label
            })
        
        return {
            'filename': filename,
            'width': width,
            'height': height,
            'boxes': boxes
        }
    
    def load_yolo_annotation(self, txt_file, image_width, image_height):
        """
        Load YOLO format annotation
        
        Args:
            txt_file: Path to YOLO format text file
            image_width: Image width
            image_height: Image height
        
        Returns:
            List of bounding boxes
        """
        boxes = []
        
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * image_width
                    y_center = float(parts[2]) * image_height
                    width = float(parts[3]) * image_width
                    height = float(parts[4]) * image_height
                    
                    xmin = int(x_center - width / 2)
                    ymin = int(y_center - height / 2)
                    xmax = int(x_center + width / 2)
                    ymax = int(y_center + height / 2)
                    
                    boxes.append({
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax,
                        'label': f'class_{class_id}'
                    })
        
        return boxes
    
    def load_json_annotation(self, json_file):
        """
        Load JSON format annotation
        
        Args:
            json_file: Path to JSON annotation file
        
        Returns:
            Dictionary with annotation data
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    
    def load_dataset(self, test_size=0.2, val_size=0.1):
        """
        Load entire dataset
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
        
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        images = []
        bboxes = []
        
        # Get all image files
        image_files = [f for f in os.listdir(self.images_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(self.images_dir, img_file)
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Load annotation
            base_name = os.path.splitext(img_file)[0]
            
            if self.annotation_format == 'pascal_voc':
                ann_file = os.path.join(self.annotations_dir, base_name + '.xml')
                if os.path.exists(ann_file):
                    ann_data = self.load_pascal_voc_annotation(ann_file)
                    if ann_data['boxes']:
                        box = ann_data['boxes'][0]  # Take first box
                        # Convert to [x, y, w, h] format
                        bbox = [
                            box['xmin'],
                            box['ymin'],
                            box['xmax'] - box['xmin'],
                            box['ymax'] - box['ymin']
                        ]
                        images.append(image)
                        bboxes.append(bbox)
            
            elif self.annotation_format == 'yolo':
                ann_file = os.path.join(self.annotations_dir, base_name + '.txt')
                if os.path.exists(ann_file):
                    h, w = image.shape[:2]
                    boxes = self.load_yolo_annotation(ann_file, w, h)
                    if boxes:
                        box = boxes[0]
                        bbox = [
                            box['xmin'],
                            box['ymin'],
                            box['xmax'] - box['xmin'],
                            box['ymax'] - box['ymin']
                        ]
                        images.append(image)
                        bboxes.append(bbox)
        
        # Convert to numpy arrays
        images = np.array(images)
        bboxes = np.array(bboxes)
        
        # Split dataset
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            images, bboxes, test_size=test_size, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=42
        )
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def preprocess_for_detection(self, images, bboxes, target_size=(224, 224)):
        """
        Preprocess images and bboxes for detection model
        
        Args:
            images: Array of images
            bboxes: Array of bounding boxes [x, y, w, h]
            target_size: Target image size
        
        Returns:
            (preprocessed_images, normalized_bboxes)
        """
        processed_images = []
        normalized_bboxes = []
        
        for img, bbox in zip(images, bboxes):
            # Get original dimensions
            h, w = img.shape[:2]
            
            # Resize image
            resized = cv2.resize(img, target_size)
            normalized_img = resized.astype('float32') / 255.0
            processed_images.append(normalized_img)
            
            # Normalize bbox coordinates to [0, 1]
            x, y, width, height = bbox
            norm_bbox = [
                x / w,
                y / h,
                width / w,
                height / h
            ]
            normalized_bboxes.append(norm_bbox)
        
        return np.array(processed_images), np.array(normalized_bboxes)
    
    def generate_sample_data(self, num_samples=100):
        """
        Generate synthetic sample data for testing
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            (images, bboxes)
        """
        print(f"Generating {num_samples} synthetic samples...")
        
        images = []
        bboxes = []
        
        for _ in range(num_samples):
            # Create random image
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Generate random bbox (simulating a license plate)
            x = np.random.randint(50, 400)
            y = np.random.randint(50, 300)
            w = np.random.randint(100, 200)
            h = np.random.randint(30, 60)
            
            # Draw a rectangle to simulate plate
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), -1)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            
            images.append(img)
            bboxes.append([x, y, w, h])
        
        return np.array(images), np.array(bboxes)


def create_character_dataset(plate_images, labels, img_height=32, img_width=128):
    """
    Create character recognition dataset from plate images
    
    Args:
        plate_images: List of plate region images
        labels: List of corresponding text labels
        img_height: Target height
        img_width: Target width
    
    Returns:
        (X, y) where X is images and y is one-hot encoded labels
    """
    X = []
    y = []
    
    for img, label in zip(plate_images, labels):
        # Preprocess image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        resized = cv2.resize(gray, (img_width, img_height))
        normalized = resized.astype('float32') / 255.0
        normalized = np.expand_dims(normalized, axis=-1)
        
        X.append(normalized)
        y.append(label)
    
    return np.array(X), np.array(y)


if __name__ == '__main__':
    """Test data loader"""
    print("Testing data loader...")
    
    # Example usage
    loader = LicensePlateDataLoader('../../data/raw')
    
    # Generate sample data for testing
    images, bboxes = loader.generate_sample_data(num_samples=50)
    print(f"Generated {len(images)} samples")
    print(f"Image shape: {images[0].shape}")
    print(f"Bbox shape: {bboxes[0].shape}")
