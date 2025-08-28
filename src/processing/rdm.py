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
    Accepts echo_matrix as [num_chirps, num_samples] or flattens if 1D.
    """
    import numpy as np

    # Ensure at least 2D
    echo_matrix = np.atleast_2d(echo_matrix)

    # FFT along range dimension
    range_fft = np.fft.fft(echo_matrix, n=n_fft_range, axis=1)

    # FFT along doppler dimension
    rdm = np.fft.fftshift(np.fft.fft(range_fft, n=n_fft_doppler, axis=0), axes=0)

    return np.abs(rdm)
