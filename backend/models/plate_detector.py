"""
CNN Model for License Plate Detection
Uses a custom CNN architecture to detect license plates in images
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


class PlateDetector:
    """
    License Plate Detector using CNN
    Detects the bounding box of license plates in images
    """
    
    def __init__(self, input_shape=(224, 224, 3)):
        """
        Initialize the plate detector
        
        Args:
            input_shape: Input image shape (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = None
        self.cascade = self._load_cascade()
    
    def _load_cascade(self):
        """Load Haar Cascade for initial detection (fallback method)"""
        try:
            # Try to load pre-trained Haar Cascade for Russian plates (works for Indian too)
            cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                print("Warning: Haar Cascade not loaded")
                return None
            return cascade
        except:
            return None
    
    def build_model(self):
        """
        Build CNN model for plate detection
        Uses MobileNetV2 as backbone for efficiency
        """
        # Use MobileNetV2 as feature extractor
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build detection head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(4, activation='sigmoid')  # [x, y, width, height] normalized to 0-1
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the plate detector model
        
        Args:
            X_train: Training images
            y_train: Training bounding boxes [x, y, w, h] normalized
            X_val: Validation images
            y_val: Validation bounding boxes
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_plate_detector.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def detect(self, image):
        """
        Detect license plate in image
        
        Args:
            image: Input image (BGR format from OpenCV)
        
        Returns:
            Bounding box [x, y, width, height] or None if no plate detected
        """
        # Try Haar Cascade first (fast method)
        if self.cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            plates = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(25, 25)
            )
            
            if len(plates) > 0:
                # Return the largest detected plate
                largest_plate = max(plates, key=lambda p: p[2] * p[3])
                return tuple(largest_plate)
        
        # If CNN model is loaded, use it
        if self.model is not None:
            try:
                # Preprocess image
                img_resized = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
                img_normalized = img_resized.astype('float32') / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                # Predict bounding box
                pred = self.model.predict(img_batch, verbose=0)[0]
                
                # Denormalize coordinates
                h, w = image.shape[:2]
                x = int(pred[0] * w)
                y = int(pred[1] * h)
                width = int(pred[2] * w)
                height = int(pred[3] * h)
                
                # Validate bounding box
                if width > 20 and height > 10:
                    return (x, y, width, height)
            except Exception as e:
                print(f"CNN detection error: {e}")
        
        # Fallback: Use contour detection
        return self._detect_by_contours(image)
    
    def _detect_by_contours(self, image):
        """
        Fallback method: Detect plate using contour detection
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Bounding box or None
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Edge detection
            edged = cv2.Canny(filtered, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(
                edged.copy(),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            plate_contour = None
            
            # Find rectangular contours
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
                
                # License plates typically have 4 corners
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    
                    # Indian license plates have aspect ratio between 2:1 and 4:1
                    if 2.0 <= aspect_ratio <= 4.5:
                        plate_contour = approx
                        return (x, y, w, h)
            
            return None
        
        except Exception as e:
            print(f"Contour detection error: {e}")
            return None
    
    def save_model(self, filepath):
        """Save model to file"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        try:
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
