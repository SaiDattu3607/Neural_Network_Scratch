import numpy as np

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]

    def __iter__(self):
        # create indices
        indices = np.arange(self.num_samples)

        # shuffle if needed
        if self.shuffle:
            np.random.shuffle(indices)

        # yield batches
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]

            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]

            yield X_batch, y_batch # instead of returning everything at once,
                                   # it pauses, returns one batch and continue.