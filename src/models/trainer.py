"""
Training utilities for ML and DL models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
'''
def train_model(model, dataloader, num_epochs=10, lr=1e-3, device="cpu"):
    """
    Train a PyTorch model.
    
    Args:
        model: nn.Module
        dataloader: PyTorch DataLoader
        num_epochs: number of epochs
        lr: learning rate
        device: "cpu" or "cuda"
    
    Returns:
        trained model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
    return model
'''
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
    Evaluate a trained model.
    
    Returns:
        accuracy (float)
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
