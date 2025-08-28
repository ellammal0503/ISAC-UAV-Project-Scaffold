"""
Evaluation metrics for UAV sensing.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def classification_metrics(y_true, y_pred):
    """
    Compute classification metrics (precision, recall, F1).
    
    Args:
        y_true (list[int]): ground truth labels
        y_pred (list[int]): predicted labels
    
    Returns:
        dict with precision, recall, f1
    """
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def localization_error(true_positions, est_positions):
    """
    Compute localization error (Euclidean distance).
    
    Args:
        true_positions (np.ndarray): shape (N, 3) true UAV positions [x,y,z]
        est_positions (np.ndarray): shape (N, 3) estimated UAV positions
    
    Returns:
        float: mean error
    """
    errors = np.linalg.norm(true_positions - est_positions, axis=1)
    return np.mean(errors)

def velocity_error(true_velocities, est_velocities):
    """
    Compute velocity estimation error (m/s).
    
    Args:
        true_velocities (np.ndarray): shape (N,) true velocities
        est_velocities (np.ndarray): shape (N,) estimated velocities
    
    Returns:
        float: mean absolute error
    """
    return np.mean(np.abs(true_velocities - est_velocities))
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
def train_model(model, dataloader, num_epochs=10, lr=1e-3, device="cpu"):
    """
    Train a PyTorch model and return training history.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "acc": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(dataloader)
        acc = correct / total
        history["loss"].append(avg_loss)
        history["acc"].append(acc)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    return model, history   # ✅ Return both


def evaluate_model(model, dataloader, device="cpu"):
    """
    Evaluate a trained model and return accuracy.
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return accuracy_score(y_true, y_pred)
