import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from zonotope.plot import ORANGE, PINK, TURQUOISE, VIOLET


def prepare_2d_dataset(three_classes: bool = False):
    """
    Prepare a 2D classification dataset for visualization
    Returns:
        X_train, X_test, y_train, y_test: Train/test split data
        scaler: Fitted scaler for transforming new points
    """
    # Generate a more interesting dataset with curved decision boundaries
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

    if three_classes:
        # Add a third class
        X_class2, y_class2 = make_classification(
            n_samples=500,
            n_features=2,
            n_classes=1,
            n_redundant=0,
            n_informative=2,
            random_state=42,
        )
        X_class2 = X_class2 + np.array([2.0, 2.0])  # Shift the third class

        # Combine datasets
        X = np.vstack([X, X_class2])
        y = np.concatenate([y, np.ones(500) * 2])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def prepare_high_dim_dataset(input_dim=10, n_classes=4):
    """
    Prepare a higher-dimensional classification dataset
    """
    X, y = make_classification(
        n_samples=2000,
        n_features=input_dim,
        n_classes=n_classes,
        n_informative=input_dim,
        n_redundant=0,
        random_state=42,
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def plot_dataset(X, y, title="Dataset"):
    """Plot a 2D dataset with different classes"""
    plt.figure(figsize=(10, 8))

    # Use our custom color palette
    colors = [ORANGE, VIOLET, PINK, TURQUOISE]

    for i in np.unique(y):
        plt.scatter(
            X[y == i, 0],
            X[y == i, 1],
            color=colors[int(i) % len(colors)],
            label=f"Class {i}",
            alpha=0.7,
            edgecolor="white",
            s=60,
        )

    plt.title(title, fontsize=14)
    plt.legend(frameon=True, facecolor="white", edgecolor=VIOLET, framealpha=0.8)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()
