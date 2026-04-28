"""Data loading utilities."""

import numpy as np


class DataLoader:
    """Mini-batch data loader with shuffling support."""
    
    def __init__(self, X, y, batch_size=32, shuffle=True):
        """Initialize DataLoader.
        
        Args:
            X: Feature array of shape (num_samples, num_features)
            y: Label array of shape (num_samples, num_classes)
            batch_size: Size of mini-batches
            shuffle: Whether to shuffle data before each iteration
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]

    def __iter__(self):
        """Iterate over mini-batches.
        
        Yields:
            Tuple of (X_batch, y_batch) for each batch
        """
        # Create indices
        indices = np.arange(self.num_samples)

        # Shuffle if needed
        if self.shuffle:
            np.random.shuffle(indices)

        # Yield batches
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]

            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]

            yield X_batch, y_batch