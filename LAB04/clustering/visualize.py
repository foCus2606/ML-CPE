import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_elbow(k_values, inertias, out_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_values, inertias, "o-")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_clusters_pca(X, labels, out_path):
    """
    Image data has hundreds of features, so PCA is used only for
    2D visualization of the clusters.
    """
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(7.5, 6))

    for c in range(labels.max() + 1):
        members = labels == c
        plt.scatter(
            X_2d[members, 0],
            X_2d[members, 1],
            s=20,
            alpha=0.6,
            label=f"Cluster {c}",
        )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("K-Means Clustering of Cat Images")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
