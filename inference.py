"""Inference script for making predictions with trained model."""

import numpy as np
import argparse
from PIL import Image

from models.mlp import MLP
from nn.activations import softmax
from config import INPUT_SIZE, DATA_DIR


def load_model_weights(model, filepath):
    """Load model weights from file."""
    weights = np.load(filepath, allow_pickle=True).item()
    model.l1.W = weights['l1_W']
    model.l1.b = weights['l1_b']
    model.l2.W = weights['l2_W']
    model.l2.b = weights['l2_b']
    return model


def save_model_weights(model, filepath):
    """Save model weights to file."""
    weights = {
        'l1_W': model.l1.W,
        'l1_b': model.l1.b,
        'l2_W': model.l2.W,
        'l2_b': model.l2.b,
    }
    np.save(filepath, weights)


def preprocess_image(image_path):
    """Load and preprocess image for model."""
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Flatten to 784
    img_flat = img_array.flatten().reshape(1, -1)
    
    return img_flat


def predict(image_path, model_weights=None):
    """Make prediction on a single image."""
    # Initialize model
    model = MLP()
    
    # Load weights if provided
    if model_weights:
        model = load_model_weights(model, model_weights)
    
    # Preprocess image
    img = preprocess_image(image_path)
    
    # Forward pass
    output = model.forward(img.T)
    
    # Get prediction
    prediction = np.argmax(output, axis=0)[0]
    confidence = np.max(output) * 100
    
    return prediction, confidence, output


def main():
    parser = argparse.ArgumentParser(description='MNIST inference')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights')
    
    args = parser.parse_args()
    
    try:
        pred, conf, probs = predict(args.image, args.weights)
        print(f"\nPrediction: {pred}")
        print(f"Confidence: {conf:.2f}%")
        print(f"\nClass probabilities:")
        for i, prob in enumerate(probs.flatten()):
            print(f"  {i}: {prob*100:.2f}%")
    except FileNotFoundError:
        print(f"Error: Image file not found: {args.image}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
