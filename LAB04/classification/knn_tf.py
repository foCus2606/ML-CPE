import numpy as np
import tensorflow as tf


class TFKNNClassifier:
    def __init__(self, k=5, batch_size=32):
        self.k = k
        self.batch_size = batch_size

    def fit(self, X, y):
        self.X_train = tf.constant(X, dtype=tf.float32)
        self.y_train = tf.constant(y, dtype=tf.int32)
        self.n_classes = int(np.max(y)) + 1
        return self

    def _predict_batch(self, X_batch):
        X_batch = tf.constant(X_batch, dtype=tf.float32)

        # Euclidean distance
        diff = X_batch[:, None, :] - self.X_train[None, :, :]
        dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))

        # find k nearest neighbors
        _, idx = tf.math.top_k(-dist, k=self.k)
        neighbor_labels = tf.gather(self.y_train, idx)

        # majority voting
        onehot = tf.one_hot(neighbor_labels, depth=self.n_classes)
        votes = tf.reduce_sum(onehot, axis=1)

        return tf.argmax(votes, axis=1).numpy()

    def predict(self, X):
        preds = []

        # predict in batches to reduce memory usage for image features
        for start in range(0, len(X), self.batch_size):
            batch = X[start:start + self.batch_size]
            preds.append(self._predict_batch(batch))

        return np.concatenate(preds).astype("int32")

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))
