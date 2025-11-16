"""
Training script for Character Recognition (OCR) Model
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import string

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.char_recognizer import CharacterRecognizer
from utils.data_loader import create_character_dataset
import cv2


def plot_training_history(history, save_path):
    """
    Plot and save training history
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    plt.close()


def generate_sample_plate_data(num_samples=500):
    """
    Generate synthetic license plate images for training
    
    Args:
        num_samples: Number of samples to generate
    
    Returns:
        (images, labels)
    """
    print(f"Generating {num_samples} synthetic plate samples...")
    
    images = []
    labels = []
    
    # Indian license plate format: XX00XX0000 (state code + district + series + number)
    states = ['MH', 'DL', 'KA', 'TN', 'UP', 'GJ', 'RJ', 'MP', 'HR', 'PB']
    
    for i in range(num_samples):
        # Create blank plate image
        plate_img = np.ones((60, 240, 3), dtype=np.uint8) * 255
        
        # Generate random plate number
        state = np.random.choice(states)
        district = f"{np.random.randint(1, 99):02d}"
        series = ''.join(np.random.choice(list(string.ascii_uppercase), 2))
        number = f"{np.random.randint(1, 9999):04d}"
        plate_text = f"{state}{district}{series}{number}"
        
        # Draw text on plate
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        color = (0, 0, 0)
        
        # Calculate text size and position
        text_size = cv2.getTextSize(plate_text, font, font_scale, thickness)[0]
        text_x = (plate_img.shape[1] - text_size[0]) // 2
        text_y = (plate_img.shape[0] + text_size[1]) // 2
        
        cv2.putText(plate_img, plate_text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Add some noise and variations
        if np.random.random() > 0.5:
            noise = np.random.randint(-20, 20, plate_img.shape, dtype=np.int16)
            plate_img = np.clip(plate_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Random rotation
        if np.random.random() > 0.7:
            angle = np.random.uniform(-5, 5)
            h, w = plate_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            plate_img = cv2.warpAffine(plate_img, M, (w, h), borderValue=(255, 255, 255))
        
        images.append(plate_img)
        labels.append(plate_text)
    
    return images, labels


def encode_labels(labels, recognizer):
    """
    Encode text labels to one-hot format
    
    Args:
        labels: List of text labels
        recognizer: CharacterRecognizer instance
    
    Returns:
        One-hot encoded labels
    """
    encoded = []
    
    for label in labels:
        # Pad or truncate to max_length
        label = label[:recognizer.max_length].ljust(recognizer.max_length, ' ')
        
        # Create one-hot encoding for each character position
        label_encoded = np.zeros((recognizer.max_length, len(recognizer.CHARACTERS) + 1))
        
        for i, char in enumerate(label):
            if char in recognizer.CHAR_TO_IDX:
                label_encoded[i, recognizer.CHAR_TO_IDX[char]] = 1
            else:
                # Use last index for unknown/space characters
                label_encoded[i, -1] = 1
        
        encoded.append(label_encoded)
    
    return np.array(encoded)


def train_ocr(epochs=100, batch_size=64, use_sample_data=False):
    """
    Train the character recognition (OCR) model

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        use_sample_data: Use synthetic sample data for testing
    """
    print("=" * 60)
    print("Character Recognition (OCR) Training")
    print("=" * 60)

    # Initialize recognizer
    recognizer = CharacterRecognizer()
    recognizer.build_model()

    print("\nModel Summary:")
    recognizer.model.summary()

    # Generate or load data
    print("\nPreparing dataset...")

    if use_sample_data:
        print("Using synthetic sample data...")
        images, labels = generate_sample_plate_data(num_samples=1000)
    else:
        print("⚠ No real dataset provided, using synthetic data...")
        print("To use real data, implement data loading from data/raw/")
        images, labels = generate_sample_plate_data(num_samples=1000)

    # Split data
    split_idx = int(0.8 * len(images))
    val_split_idx = int(0.9 * len(images))

    train_images = images[:split_idx]
    train_labels = labels[:split_idx]
    val_images = images[split_idx:val_split_idx]
    val_labels = labels[split_idx:val_split_idx]
    test_images = images[val_split_idx:]
    test_labels = labels[val_split_idx:]

    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Test samples: {len(test_images)}")

    # Preprocess images
    print("\nPreprocessing images...")
    X_train, _ = create_character_dataset(train_images, train_labels)
    X_val, _ = create_character_dataset(val_images, val_labels)
    X_test, _ = create_character_dataset(test_images, test_labels)

    # Encode labels
    print("Encoding labels...")
    y_train = encode_labels(train_labels, recognizer)
    y_val = encode_labels(val_labels, recognizer)
    y_test = encode_labels(test_labels, recognizer)

    # Train model
    print(f"\nTraining model for {epochs} epochs...")
    print("=" * 60)

    history = recognizer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss = recognizer.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")

    # Test predictions
    print("\nSample Predictions:")
    for i in range(min(5, len(X_test))):
        pred_text, confidence = recognizer.recognize(test_images[i])
        actual_text = test_labels[i]
        print(f"  Actual: {actual_text:12s} | Predicted: {pred_text:12s} | Confidence: {confidence:.2%}")

    # Save model
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, 'char_recognizer.h5')
    recognizer.save_model(model_path)

    # Plot training history
    plot_training_history(history, os.path.join(save_dir, 'ocr_training_history.png'))

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Character Recognition (OCR) Model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--use-sample-data', action='store_true',
                       help='Use synthetic sample data for testing')

    args = parser.parse_args()

    train_ocr(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_sample_data=args.use_sample_data
    )


if __name__ == '__main__':
    main()

