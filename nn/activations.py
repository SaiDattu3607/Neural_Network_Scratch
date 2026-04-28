"""Activation functions for neural networks."""

import numpy as np


def sigmoid(x):
    """Sigmoid activation function: 1 / (1 + e^-x).
    
    Squashes input to range (0, 1). Good for binary classification.
    
    Args:
        x: Input array
        
    Returns:
        Activated output
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """Softmax activation function for multi-class classification.
    
    Converts logits to probability distribution.
    Uses numerical stability trick: subtract max before exp.
    
    Args:
        x: Input logits of shape (num_classes, batch_size)
        
    Returns:
        Probabilities summing to 1 along class axis
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def relu(x):
    """Rectified Linear Unit: max(0, x).
    
    Good for hidden layers. Sparse activations and faster learning.
    
    Args:
        x: Input array
        
    Returns:
        Activated output with negatives clipped to 0
    """
    return np.maximum(0, x)