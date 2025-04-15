from typing import List

import numpy as np
import torch as t
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from zonotope.plot.theme import ORANGE, PINK, TURQUOISE, VIOLET


class SimpleNN(t.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        self.layers = t.nn.ModuleList()
        self.layers.append(t.nn.Linear(input_dim, hidden_dims[0]))

        for i in range(len(hidden_dims) - 1):
            self.layers.append(t.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.layers.append(t.nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = t.nn.functional.relu(layer(x))
        return self.layers[-1](x)


def train_model(model, X_train, y_train, X_test, y_test, epochs=500, lr=0.01):
    """
    Train the neural network model

    Args:
        model: Neural network model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        train_losses: List of training losses
        test_accuracies: List of test accuracies
    """
    # Convert numpy arrays to torch tensors
    X_train_tensor = t.tensor(X_train, dtype=t.float32)
    y_train_tensor = t.tensor(y_train, dtype=t.long)
    X_test_tensor = t.tensor(X_test, dtype=t.float32)
    y_test_tensor = t.tensor(y_test, dtype=t.long)

    # Define loss function and optimizer
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_accuracies = []

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Evaluate on test set
        if epoch % 50 == 0:
            with t.no_grad():
                test_outputs = model(X_test_tensor)
                predicted = test_outputs.argmax(dim=1)
                accuracy = (predicted == y_test_tensor).sum().item() / len(
                    y_test_tensor
                )
                test_accuracies.append(accuracy)
                print(
                    f"Epoch {epoch}, Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}"
                )

    # Final evaluation
    with t.no_grad():
        test_outputs = model(X_test_tensor)
        predicted = test_outputs.argmax(dim=1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f"Final Test Accuracy: {accuracy:.4f}")

    return train_losses, test_accuracies


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


def plot_training_history(losses, accuracies=None):
    """Plot training loss and accuracy history"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training loss
    epochs = np.arange(len(losses))
    ax1.plot(epochs, losses, color=ORANGE, linewidth=2, label="Training Loss")
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", color=ORANGE, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=ORANGE)

    # Plot test accuracy if provided
    if accuracies is not None:
        ax2 = ax1.twinx()
        accuracy_epochs = np.linspace(0, len(losses), len(accuracies))
        ax2.plot(
            accuracy_epochs,
            accuracies,
            color=VIOLET,
            linewidth=2,
            label="Test Accuracy",
        )
        ax2.set_ylabel("Accuracy", color=VIOLET, fontsize=12)
        ax2.tick_params(axis="y", labelcolor=VIOLET)

    plt.title("Training History", fontsize=14)
    fig.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add a custom legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if accuracies is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(loc="best")

    plt.show()


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
