"""
Main entrypoint for ISAC-UAV project.

Runs a complete pipeline:
1. Simulate UAV scenario and channel
2. Generate ISAC echoes
3. Apply signal processing pipeline
4. Train/evaluate AI models
"""

import argparse
from src.sim.uav_scenarios import generate_uav_scenario
from src.sim.channel_38901 import Channel38901
from src.sim.echo_generator import EchoGenerator
from src.processing.rdm import compute_rdm
#from src.processing.cfar import cfar_detection
from src.processing.cfar import cfar_detect
#from src.processing.doa_music import doa_estimation
from src.processing.doa_music import music_doa
from src.processing.feature_extraction import extract_features
#from src.models.trainer import ModelTrainer
from src.models.trainer import train_model,evaluate_model
#from src.utils.plotting import plot_rdm, plot_doa
from src.utils.plotting import plot_rdm, plot_doa_spectrum
from src import config
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import torch
from torch.utils.data import TensorDataset, DataLoader
from models.baseline import BaselineNN
from utils.metrics import train_model, evaluate_model
from models.baselines import CNNBaseline, RNNBaseline, TransformerBaseline
import os
import torch
import matplotlib.pyplot as plt
from utils.metrics import train_model, evaluate_model
#from utils.models import BaselineMLP, BaselineCNN, BaselineRNN, BaselineTransformer

# -----------------------------
# Helper functions for training multiple models
# -----------------------------
def save_model(model, name):
    path = f"results/models/{name}.pth"
    os.makedirs(os.path.dirname(path), exist_ok=True)  # <-- ensure parent folder exists
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved {name} model -> {path}")

# from models.baselines import MLPBaseline, CNNBaseline, RNNBaseline, TransformerBaseline
def run_all_models(dataloader, input_dim, num_classes=2, num_epochs=5):
    models = {
        "MLP": BaselineNN(input_dim=input_dim, hidden_dim=64, num_classes=num_classes),
        "CNN": CNNBaseline(input_dim=input_dim, num_classes=num_classes),
        "RNN": RNNBaseline(input_dim=input_dim, hidden_dim=32, num_classes=num_classes),
        "Transformer": TransformerBaseline(input_dim=input_dim, num_heads=2, num_classes=num_classes),
    }

    histories = {}
    for name, model in models.items():
        print(f"\n[INFO] Training {name} model...")
        model, history = train_model(model, dataloader, num_epochs=num_epochs)  # unpack
        acc = evaluate_model(model, dataloader)
        print(f"[RESULT] {name} Accuracy: {acc:.2f}")
        save_model(model, name)
        histories[name] = history  # now it's a dict with 'loss' & 'acc'
    return histories

def plot_histories(histories):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    for name, hist in histories.items():
        plt.plot(hist["loss"], label=f"{name}")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    for name, hist in histories.items():
        plt.plot(hist["acc"], label=f"{name}")
    plt.title("Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Save plot
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/model_comparison.png")
    print("[INFO] Saved comparison plot -> results/plots/model_comparison.png")
    plt.close()

def run_pipeline():
    # ----------------------------
    # Step 1: UAV Scenario
    # ----------------------------
    scenario = generate_uav_scenario()
    print(f"[INFO] Generated UAV scenario: {scenario}")

    # ----------------------------
    # Step 2: Channel
    # ----------------------------
    channel = Channel38901(scenario)
    print("[INFO] Channel initialized")

    # ----------------------------
    # Step 3: Echo
    # ----------------------------
    echo_gen = EchoGenerator(channel)
    rx_signal = echo_gen.generate_echo()
    print("[INFO] Echo signal generated")

    # ----------------------------
    # Step 4: RDM
    # ----------------------------
    rdm = compute_rdm(rx_signal)
    plot_rdm(rdm)

    # ----------------------------
    # Step 5: CFAR detection
    # ----------------------------
    detections = cfar_detect(rdm)
    print(f"[INFO] Detections: {detections}")

    # ----------------------------
    # Step 6: DoA estimation (MUSIC)
    # ----------------------------
    n_antennas = 8
    rx_signal = rx_signal.reshape(1, -1)       # reshape 1D -> 2D
    rx_signal = np.tile(rx_signal, (n_antennas, 1))
    print(f"[DEBUG] Patched rx_signal shape: {rx_signal.shape}")

    doa_est, angles, spectrum = music_doa(rx_signal)
    plot_doa_spectrum(doa_est, angles, spectrum)
   

    # Step 7: Features
    features, labels = extract_features(detections, doa_est)
    print(f"[INFO] Extracted {len(features)} features")

    # ----------------------------
    # Add realistic UAV/ISAC noise
    # ----------------------------
    features = np.array(features, dtype=np.float32)

    # Noise parameters
    gaussian_noise_std = 0.03  # Gaussian amplitude noise
    dropout_prob = 0.05        # Randomly zero-out some features
    scale_variation = 0.02      # Small multiplicative variation

    # Apply Gaussian noise
    features += np.random.normal(0, gaussian_noise_std, features.shape)

    # Apply random feature dropout
    mask = np.random.rand(*features.shape) > dropout_prob
    features *= mask

    # Apply small multiplicative variation
    features *= (1 + np.random.normal(0, scale_variation, features.shape))

    # Wrap into PyTorch tensors
    X_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
  
    # ----------------------------
    # Step 8: Convert to tensors & DataLoader
    # ----------------------------
    X_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # ----------------------------
    # Step 9: Train all baseline models
    # ----------------------------
    histories = run_all_models(
        dataloader,
        input_dim=X_tensor.shape[1],
        num_classes=len(set(labels)),
        num_epochs=5
    )

    # ----------------------------
    # Step 10: Plot training histories
    # ----------------------------
    plot_histories(histories)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ISAC-UAV pipeline")
    parser.add_argument("--mode", type=str, default="demo", help="demo/train/test")
    args = parser.parse_args()

    run_pipeline()
