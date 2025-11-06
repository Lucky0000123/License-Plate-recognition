"""
Training script for License Plate Detector
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.plate_detector import PlateDetector
from utils.data_loader import LicensePlateDataLoader


def plot_training_history(history, save_path='detector_training_history.png'):
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
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot MAE
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Training and Validation MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")


def train_detector(data_dir, epochs=50, batch_size=32, use_sample_data=False):
    """
    Train the plate detector model
    
    Args:
        data_dir: Directory containing training data
        epochs: Number of training epochs
        batch_size: Batch size for training
        use_sample_data: Use synthetic sample data for testing
    """
    print("=" * 60)
    print("License Plate Detector Training")
    print("=" * 60)
    
    # Initialize detector
    detector = PlateDetector()
    detector.build_model()
    
    print("\nModel Summary:")
    detector.model.summary()
    
    # Load data
    print("\nLoading dataset...")
    loader = LicensePlateDataLoader(data_dir)
    
    if use_sample_data:
        print("Using synthetic sample data...")
        images, bboxes = loader.generate_sample_data(num_samples=200)
        
        # Split data
        split_idx = int(0.8 * len(images))
        val_split_idx = int(0.9 * len(images))
        
        X_train = images[:split_idx]
        y_train = bboxes[:split_idx]
        X_val = images[split_idx:val_split_idx]
        y_val = bboxes[split_idx:val_split_idx]
        X_test = images[val_split_idx:]
        y_test = bboxes[val_split_idx:]
    else:
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = loader.load_dataset()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to synthetic data...")
            images, bboxes = loader.generate_sample_data(num_samples=200)
            
            split_idx = int(0.8 * len(images))
            val_split_idx = int(0.9 * len(images))
            
            X_train = images[:split_idx]
            y_train = bboxes[:split_idx]
            X_val = images[split_idx:val_split_idx]
            y_val = bboxes[split_idx:val_split_idx]
            X_test = images[val_split_idx:]
            y_test = bboxes[val_split_idx:]
    
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train_processed, y_train_processed = loader.preprocess_for_detection(X_train, y_train)
    X_val_processed, y_val_processed = loader.preprocess_for_detection(X_val, y_val)
    X_test_processed, y_test_processed = loader.preprocess_for_detection(X_test, y_test)
    
    # Train model
    print(f"\nTraining model for {epochs} epochs...")
    print("=" * 60)
    
    history = detector.train(
        X_train_processed, y_train_processed,
        X_val_processed, y_val_processed,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_mae = detector.model.evaluate(X_test_processed, y_test_processed)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Save model
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'plate_detector.h5')
    detector.save_model(model_path)
    
    # Plot training history
    plot_training_history(history, os.path.join(save_dir, 'detector_training_history.png'))
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train License Plate Detector')
    parser.add_argument('--data-dir', type=str, default='../../data/raw',
                       help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--use-sample-data', action='store_true',
                       help='Use synthetic sample data for testing')
    
    args = parser.parse_args()
    
    train_detector(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_sample_data=args.use_sample_data
    )


if __name__ == '__main__':
    main()
