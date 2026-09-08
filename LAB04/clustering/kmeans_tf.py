import numpy as np
import tensorflow as tf


class TFKMeans:
    def __init__(self, n_clusters=4, max_iter=100, seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed

    def _distance(self, X, centroids):
        diff = X[:, None, :] - centroids[None, :, :]
        return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))

    def fit(self, X):
        X = tf.constant(X, dtype=tf.float32)
        n_samples = X.shape[0]

        rng = np.random.default_rng(self.seed)
        start_idx = rng.choice(
            n_samples,
            size=self.n_clusters,
            replace=False,
        )
        centroids = tf.gather(X, start_idx)

        for step in range(self.max_iter):
            dist = self._distance(X, centroids)
            labels = tf.argmin(dist, axis=1, output_type=tf.int32)

            new_centroids = []
            for c in range(self.n_clusters):
                members = tf.boolean_mask(X, labels == c)

                if tf.shape(members)[0] > 0:
                    new_centroids.append(tf.reduce_mean(members, axis=0))
                else:
                    new_centroids.append(centroids[c])

            new_centroids = tf.stack(new_centroids)

            moved = float(
                tf.reduce_max(tf.abs(new_centroids - centroids))
            )
            centroids = new_centroids

            if moved < 1e-4:
                break

        dist = self._distance(X, centroids)
        self.labels_ = tf.argmin(
            dist,
            axis=1,
            output_type=tf.int32,
        ).numpy()

        self.centroids_ = centroids.numpy()
        self.n_iter_ = step + 1
        self.inertia_ = float(
            tf.reduce_sum(tf.square(tf.reduce_min(dist, axis=1)))
        )

        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
