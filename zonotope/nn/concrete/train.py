import numpy as np
import torch as t
from matplotlib import pyplot as plt

from zonotope.plot import ORANGE, VIOLET


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
        lines2, labels2 = ax2.get_legend_handles_labels()  # type: ignore
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(loc="best")

    plt.show()
