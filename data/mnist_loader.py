"""MNIST dataset loader from binary format."""

import numpy as np
import struct
import os


def load_images(filepath):
    """Load images from MNIST binary file.
    
    MNIST format: 16-byte header followed by image data
    Header: [magic_number, num_images, rows, cols]
    
    Args:
        filepath: Path to MNIST images file (*.idx3-ubyte)
        
    Returns:
        Array of shape (num_images, 784) with values in [0, 1]
    """
    with open(filepath, 'rb') as f:
        # Read header
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
        
        # Normalize (0–255 → 0–1)
        images = images / 255.0
        
        return images
    

def load_labels(filepath):
    """Load labels from MNIST binary file.
    
    MNIST format: 8-byte header followed by label data
    Header: [magic_number, num_labels]
    
    Args:
        filepath: Path to MNIST labels file (*.idx1-ubyte)
        
    Returns:
        Array of shape (num_labels,) with values in [0, 9]
    """
    with open(filepath, 'rb') as f:
        # Read header
        magic, num = struct.unpack(">II", f.read(8))
        
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        return labels
    

def load_mnist(data_dir):
    """Load complete MNIST dataset.
    
    Args:
        data_dir: Directory containing MNIST binary files
        
    Returns:
        Tuple of ((train_images, train_labels), (test_images, test_labels))
    """
    train_images = load_images(os.path.join(data_dir, "train-images.idx3-ubyte"))
    train_labels = load_labels(os.path.join(data_dir, "train-labels.idx1-ubyte"))
    
    test_images = load_images(os.path.join(data_dir, "t10k-images.idx3-ubyte"))
    test_labels = load_labels(os.path.join(data_dir, "t10k-labels.idx1-ubyte"))
    
    return (train_images, train_labels), (test_images, test_labels) 