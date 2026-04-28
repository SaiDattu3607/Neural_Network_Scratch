"""Loss functions for training neural networks."""

import numpy as np


def categorical_cross_entropy(y, y_hat):
    """Categorical cross-entropy loss.
    
    Standard loss for multi-class classification.
    Loss = -sum(y * log(y_hat)) / batch_size
    
    Args:
        y: One-hot encoded true labels of shape (num_classes, batch_size)
        y_hat: Predicted probabilities of shape (num_classes, batch_size)
        
    Returns:
        Scalar loss value
    """
    eps = 1e-8
    y_hat = np.clip(y_hat, eps, 1 - eps)  # Numerical stability
    return -np.sum(y * np.log(y_hat)) / y.shape[1]