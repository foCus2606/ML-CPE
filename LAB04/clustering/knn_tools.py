import numpy as np
import tensorflow as tf


class KNNClusterAssigner:
    def __init__(self, k=5, batch_size=32):
        self.k = k
        self.batch_size = batch_size

    def fit(self, X, cluster_labels):
        self.X = tf.constant(X, dtype=tf.float32)
        self.labels = tf.constant(cluster_labels, dtype=tf.int32)
        self.n_clusters = int(cluster_labels.max()) + 1
        return self

    def _predict_batch(self, X_batch):
        X_batch = tf.constant(X_batch, dtype=tf.float32)

        diff = X_batch[:, None, :] - self.X[None, :, :]
        dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))

        _, idx = tf.math.top_k(-dist, k=self.k)
        neighbor_labels = tf.gather(self.labels, idx)

        onehot = tf.one_hot(
            neighbor_labels,
            depth=self.n_clusters,
        )
        votes = tf.reduce_sum(onehot, axis=1)

        return tf.argmax(votes, axis=1).numpy().astype("int32")

    def predict(self, X_new):
        preds = []

        for start in range(0, len(X_new), self.batch_size):
            batch = X_new[start:start + self.batch_size]
            preds.append(self._predict_batch(batch))

        return np.concatenate(preds)
