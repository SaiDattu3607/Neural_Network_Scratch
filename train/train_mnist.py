"""MNIST training script."""

import numpy as np

from data.mnist_loader import load_mnist
from data.dataloader import DataLoader
from models.mlp import MLP
from nn.activations import softmax
from nn.losses import categorical_cross_entropy


def one_hot(labels, num_classes=10):
    """Convert class labels to one-hot encoding.
    
    Args:
        labels: Array of class labels
        num_classes: Number of classes
        
    Returns:
        One-hot encoded array of shape (num_samples, num_classes)
    """
    return np.eye(num_classes)[labels]


def train_mnist(epochs=5, batch_size=64, lr=0.1):
    """Train MLP on MNIST dataset.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate for SGD
    """
    # Load data
    (train_X, train_y), (test_X, test_y) = load_mnist("data/raw")

    train_X = train_X.astype(np.float32)
    test_X = test_X.astype(np.float32)

    train_y_onehot = one_hot(train_y)
    test_y_onehot = one_hot(test_y)

    # Create data loader
    train_loader = DataLoader(train_X, train_y_onehot, batch_size=batch_size, shuffle=True)
    model = MLP()

    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_losses = []

        for X_batch, y_batch in train_loader:
            x = X_batch.T  # (batch_size, 784) -> (784, batch_size)
            y = y_batch.T  # (batch_size, 10) -> (10, batch_size)

            # Forward pass
            a2 = model.forward(x)
            loss = categorical_cross_entropy(y, a2)
            epoch_losses.append(loss)

            # Backward pass
            grad_output = a2 - y
            grad_hidden = model.l2.backward(grad_output)
            grad_hidden = grad_hidden * (model.a1 > 0)  # ReLU derivative
            model.l1.backward(grad_hidden)

            # Update weights with SGD
            model.l2.W -= lr * model.l2.dW
            model.l2.b -= lr * np.sum(model.l2.db, axis=1, keepdims=True)
            model.l1.W -= lr * model.l1.dW
            model.l1.b -= lr * np.sum(model.l1.db, axis=1, keepdims=True)

        # Evaluate on test set
        test_preds = np.argmax(model.forward(test_X.T), axis=0)
        test_accuracy = np.mean(test_preds == test_y)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss: {np.mean(epoch_losses):.4f} | "
            f"test accuracy: {test_accuracy:.4f}"
        )


if __name__ == "__main__":
    train_mnist(epochs=10, batch_size=128, lr=0.01)
