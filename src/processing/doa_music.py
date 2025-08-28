"""
Direction-of-Arrival (DoA) estimation using MUSIC algorithm.
"""

import numpy as np
from numpy.linalg import eig

import numpy as np

def music_doa(rx_signal, n_antennas=8, n_sources=2):
    if rx_signal.ndim == 1:
        rx_signal = rx_signal.reshape(n_antennas, -1)
        rx_signal = np.tile(rx_signal, (n_antennas, 1))
        print(f"[WARN] Auto-patched rx_signal shape: {rx_signal.shape}")

    # Covariance
    R = np.dot(rx_signal, rx_signal.conj().T) / rx_signal.shape[1]

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    noise_subspace = eigvecs[:, n_sources:]

    # MUSIC spectrum
    angles = np.linspace(-90, 90, 181)
    spectrum = []
    for theta in angles:
        steering = np.exp(-1j * np.pi * np.arange(n_antennas) * np.sin(np.deg2rad(theta)))
        steering = steering.reshape(-1, 1)
        spec_val = 1 / np.linalg.norm(noise_subspace.conj().T @ steering)
        spectrum.append(spec_val)
    spectrum = np.abs(spectrum)

    # Estimated DoAs = peaks
    doa_est = angles[np.argsort(spectrum)[-n_sources:]]

    #return doa_est, spectrum
    return doa_est, angles, spectrum


#old
'''
#doa_estimation refer main.py
def music_doa(cov_matrix, num_sources, n_antennas, scan_angles=np.linspace(-90, 90, 181)):
    """
    MUSIC DoA estimation.
    
    Args:
        cov_matrix (np.ndarray): covariance matrix (n_antennas x n_antennas)
        num_sources (int): number of UAVs/targets
        n_antennas (int): array size
        scan_angles (np.ndarray): grid of angles in degrees
    
    Returns:
        tuple: (angles, spectrum)
    """
    # Eigen decomposition
    eigvals, eigvecs = eig(cov_matrix)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    
    # Noise subspace
    En = eigvecs[:, num_sources:]
    
    # Steering vectors
    spectrum = []
    for theta in scan_angles:
        sv = np.exp(-1j * np.pi * np.arange(n_antennas) * np.sin(np.deg2rad(theta)))
        sv = sv[:, None]
        ps = 1 / np.linalg.norm(En.conj().T @ sv)**2
        spectrum.append(ps)
    
    spectrum = np.array(spectrum)
    est_angles = scan_angles[np.argsort(spectrum)[-num_sources:]]
    
    return est_angles, spectrum
'''