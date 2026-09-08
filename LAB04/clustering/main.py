import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import data_loader
import visualize
from kmeans_tf import TFKMeans
from knn_tools import KNNClusterAssigner

OUT_DIR = Path(__file__).resolve().parent / "outputs"
N_CLUSTERS = 5
KNN_K = 5


def title(text):
    print("\n" + "--" * 30)
    print(text)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    title("STEP 1 : load cat image dataset")

    data = data_loader.load_data()
    X = data["X"]
    df = data["df"]

    print(f"size data : {X.shape[0]} images x {X.shape[1]} features")
    print(f"image size: {data['img_size']} x {data['img_size']} RGB")
    print("breeds in dataset:", data["class_names"])

    title("STEP 2 : how many clusters should we use?")

    k_values = [2, 3, 4, 5, 6, 7, 8]
    inertias = []
    silhouettes = []

    for k in k_values:
        km = TFKMeans(n_clusters=k).fit(X)
        sil = silhouette_score(X, km.labels_)
        inertias.append(km.inertia_)
        silhouettes.append(sil)

        print(
            f"k = {k} -> inertia = {km.inertia_:10.1f} "
            f"silhouette = {sil:.3f}"
        )

    visualize.plot_elbow(
        k_values,
        inertias,
        OUT_DIR / "01_elbow.png",
    )

    best_sil_k = k_values[int(np.argmax(silhouettes))]
    print(f"\nBest silhouette k = {best_sil_k}")
    print(f"Selected N_CLUSTERS = {N_CLUSTERS}")

    title(f"STEP 3 : Run K-Means (k = {N_CLUSTERS})")

    km = TFKMeans(n_clusters=N_CLUSTERS)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)

    print(f"iterations       : {km.n_iter_}")
    print(f"Inertia          : {km.inertia_:.1f}")
    print(f"Silhouette score : {sil:.3f}")
    print(f"members/cluster  : {np.bincount(labels).tolist()}")

    if sil < 0.25:
        print("\n[Note] Low silhouette score = weak natural clusters.")
        print("K-Means will still create clusters, so interpret them carefully.")

    visualize.plot_clusters_pca(
        X,
        labels,
        OUT_DIR / "02_clusters_pca.png",
    )

    title("STEP 4 : analyze each cluster")

    result = df.copy()
    result["cluster"] = labels
    
    cluster_breed_table = pd.crosstab(
        result["cluster"],
        result["breed"],
    )

    print("\nNumber of images by cluster and breed:")
    print(cluster_breed_table.to_string())

    cluster_breed_table.to_csv(
        OUT_DIR / "cluster_breed_summary.csv",
        encoding="utf-8-sig",
    )

    title(f"STEP 5 : use KNN to assign new images to clusters (k = {KNN_K})")

    n_known = int(len(X) * 0.8)

    X_known = X[:n_known]
    labels_known = labels[:n_known]

    X_new = X[n_known:]
    labels_new = labels[n_known:]

    assigner = KNNClusterAssigner(k=KNN_K)
    assigner.fit(X_known, labels_known)
    knn_pred = assigner.predict(X_new)

    accuracy = float(np.mean(knn_pred == labels_new))

    print(f"known images : {len(X_known)}")
    print(f"new images   : {len(X_new)}")
    print(
        "KNN cluster assignment accuracy compared with K-Means : "
        f"{accuracy * 100:.1f}%"
    )

    title("save results")

    result.to_csv(
        OUT_DIR / "clustered_cats.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for f in sorted(OUT_DIR.iterdir()):
        print(f" - outputs/{f.name}")


if __name__ == "__main__":
    main()
