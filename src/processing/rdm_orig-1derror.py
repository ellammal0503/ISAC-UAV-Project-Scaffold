"""
Range-Doppler Map (RDM) computation.

Takes echoes from EchoGenerator and computes:
- FFT across fast-time → Range
- FFT across slow-time → Doppler
"""

import numpy as np

def compute_rdm(echo_matrix, n_fft_range=256, n_fft_doppler=64):
    """
    Compute Range-Doppler Map (RDM).
    
    Args:
        echo_matrix (np.ndarray): shape (num_pulses, samples_per_pulse)
        n_fft_range (int): FFT size for range
        n_fft_doppler (int): FFT size for Doppler
    
    Returns:
        np.ndarray: Range-Doppler Map (magnitude)
    """
    # Range FFT (fast-time)
    range_fft = np.fft.fft(echo_matrix, n=n_fft_range, axis=1)

    # Doppler FFT (slow-time)
    rdm = np.fft.fftshift(np.fft.fft(range_fft, n=n_fft_doppler, axis=0), axes=0)

    return np.abs(rdm)
