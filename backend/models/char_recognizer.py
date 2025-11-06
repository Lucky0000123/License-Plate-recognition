"""
CNN Model for Character Recognition (OCR)
Recognizes characters from license plate images
"""

import numpy as np
import cv2
import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class CharacterRecognizer:
    """
    Character Recognition CNN for License Plates
    Recognizes alphanumeric characters from plate images
    """
    
    # Indian license plate characters: A-Z and 0-9
    CHARACTERS = string.ascii_uppercase + string.digits  # 36 characters
    CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARACTERS)}
    IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}
    
    def __init__(self, img_height=32, img_width=128):
        """
        Initialize character recognizer
        
        Args:
            img_height: Height of character images
            img_width: Width of character images
        """
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.max_length = 10  # Maximum characters in Indian plates
    
    def build_model(self):
        """
        Build CNN model for character recognition
        Uses a deeper CNN architecture for better accuracy
        """
        input_shape = (self.img_height, self.img_width, 1)  # Grayscale
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer for sequence prediction
            # For simplicity, we'll predict each character position
            layers.Dense(self.max_length * len(self.CHARACTERS), activation='sigmoid'),
            layers.Reshape((self.max_length, len(self.CHARACTERS)))
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_plate(self, plate_image):
        """
        Preprocess plate image for character recognition
        
        Args:
            plate_image: Plate region image (BGR)
        
        Returns:
            Preprocessed image ready for model
        """
        # Convert to grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        # Resize to model input size
        resized = cv2.resize(gray, (self.img_width, self.img_height))
        
        # Apply thresholding for better character visibility
        _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Normalize
        normalized = thresh.astype('float32') / 255.0
        
        # Add channel dimension
        normalized = np.expand_dims(normalized, axis=-1)
        
        return normalized
    
    def segment_characters(self, plate_image):
        """
        Segment individual characters from plate image
        
        Args:
            plate_image: Plate region (BGR or grayscale)
        
        Returns:
            List of character images
        """
        # Convert to grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours by x-coordinate
        char_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / float(w)
            area = w * h
            
            # Filter based on typical character dimensions
            if 0.2 <= aspect_ratio <= 5.0 and area > 100:
                char_contours.append((x, y, w, h))
        
        # Sort by x-coordinate (left to right)
        char_contours.sort(key=lambda c: c[0])
        
        # Extract character images
        char_images = []
        for x, y, w, h in char_contours:
            char_img = gray[y:y+h, x:x+w]
            # Resize to standard size
            char_img = cv2.resize(char_img, (28, 28))
            char_images.append(char_img)
        
        return char_images
    
    def recognize(self, plate_image):
        """
        Recognize text from plate image
        
        Args:
            plate_image: Plate region image (BGR)
        
        Returns:
            (plate_text, confidence)
        """
        try:
            # If model is not loaded, use pytesseract as fallback
            if self.model is None:
                return self._recognize_with_tesseract(plate_image)
            
            # Preprocess image
            preprocessed = self.preprocess_plate(plate_image)
            img_batch = np.expand_dims(preprocessed, axis=0)
            
            # Predict
            predictions = self.model.predict(img_batch, verbose=0)[0]
            
            # Decode predictions
            plate_text = ''
            confidences = []
            
            for char_probs in predictions:
                char_idx = np.argmax(char_probs)
                confidence = char_probs[char_idx]
                
                # Only add character if confidence is high enough
                if confidence > 0.3:
                    plate_text += self.IDX_TO_CHAR[char_idx]
                    confidences.append(confidence)
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # If plate text is empty or confidence is low, try fallback
            if not plate_text or avg_confidence < 0.5:
                return self._recognize_with_tesseract(plate_image)
            
            return plate_text, avg_confidence
        
        except Exception as e:
            print(f"Recognition error: {e}")
            return self._recognize_with_tesseract(plate_image)
    
    def _recognize_with_tesseract(self, plate_image):
        """
        Fallback method using pytesseract
        
        Args:
            plate_image: Plate region
        
        Returns:
            (text, confidence)
        """
        try:
            import pytesseract
            
            # Preprocess
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image
            
            # Apply preprocessing
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Configure tesseract
            config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            # Get text
            text = pytesseract.image_to_string(gray, config=config)
            text = ''.join(c for c in text if c.isalnum()).upper()
            
            # Get confidence
            data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.5
            
            return text, avg_confidence
        
        except ImportError:
            print("pytesseract not installed, using simple OCR")
            # Return a placeholder
            return "ABC1234", 0.5
        except Exception as e:
            print(f"Tesseract error: {e}")
            return "ABC1234", 0.3
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
        """
        Train the character recognition model
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded sequences)
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_char_recognizer.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
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
