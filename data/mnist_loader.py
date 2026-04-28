import numpy as np
import struct
import os

def load_images(filepath):
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
    with open(filepath, 'rb') as f:
        # Read header
        magic, num = struct.unpack(">II", f.read(8))
        
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        return labels
    

def load_mnist(data_dir):
    train_images = load_images(os.path.join(data_dir, "train-images.idx3-ubyte"))
    train_labels = load_labels(os.path.join(data_dir, "train-labels.idx1-ubyte"))
    
    test_images = load_images(os.path.join(data_dir, "t10k-images.idx3-ubyte"))
    test_labels = load_labels(os.path.join(data_dir, "t10k-labels.idx1-ubyte"))
    
    return (train_images, train_labels), (test_images, test_labels) 