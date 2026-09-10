from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def create_feature_transformer(X_train, pca_components=150):
    """
    Standardize input features before SVM training.
    PCA is kept from the teacher's example to reduce image dimensions.
    """

    n_components = min(pca_components, X_train.shape[0], X_train.shape[1])

    transformer = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=n_components,
            whiten=True,
            random_state=42
        ))
    ])

    X_train_scaled = transformer.fit_transform(X_train)

    return transformer, X_train_scaled


def train_svm_models(X_train, y_train):
    """
    Train the 3 kernels required by LAB 5:
    Linear, Polynomial, and RBF.
    """

    transformer, X_train_scaled = create_feature_transformer(X_train)

    models = {
        "Linear": SVC(kernel="linear", C=1.0),
        "Polynomial": SVC(kernel="poly", degree=3, C=1.0, gamma="scale"),
        "RBF": SVC(kernel="rbf", C=10, gamma="scale")
    }

    for kernel_name, model in models.items():
        print(f"Training {kernel_name} kernel...")
        model.fit(X_train_scaled, y_train)

    return models, transformer


def predict_svm(model, transformer, X_test):
    X_test_scaled = transformer.transform(X_test)
    predictions = model.predict(X_test_scaled)

    return predictions
