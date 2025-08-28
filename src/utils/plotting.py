"""
Plotting utilities for ISAC-UAV project.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_rdm(rdm, title="Range-Doppler Map"):
    """
    Plot a Range-Doppler Map (RDM).
    
    Args:
        rdm (np.ndarray): 2D array of RDM values
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(20 * np.log10(np.abs(rdm) + 1e-6), aspect="auto", cmap="jet")
    plt.title(title)
    plt.xlabel("Doppler bins")
    plt.ylabel("Range bins")
    plt.colorbar(label="dB")
    plt.show()
#old
'''
def plot_doa_spectrum(angles, spectrum, title="DOA Spectrum"):
    """
    Plot Direction of Arrival (DOA) spectrum.
    
    Args:
        angles (np.ndarray): angle grid (degrees)
        spectrum (np.ndarray): MUSIC or beamforming spectrum
    """
    plt.figure(figsize=(6, 4))
    plt.plot(angles, spectrum)
    plt.title(title)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Spectrum power")
    plt.grid(True)
    plt.show()

def plot_training_curve(train_losses, val_losses=None):
    """
    Plot training loss curve.
    
    Args:
        train_losses (list[float]): training loss per epoch
        val_losses (list[float], optional): validation loss per epoch
    """
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    if val_losses:
        plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
'''
import matplotlib.pyplot as plt
import numpy as np

def plot_doa_spectrum(doas, angles, spectrum):
    plt.figure()
    plt.plot(angles, 10 * np.log10(np.abs(spectrum)))
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Spectrum (dB)")
    plt.title("MUSIC DoA Spectrum")

    # Mark estimated DoAs
    for d in doas:
        plt.axvline(x=d, color='r', linestyle='--', label=f"DoA: {d:.1f}°")

    plt.legend()
    plt.grid(True)
    plt.show()


