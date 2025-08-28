"""
CFAR detector for target detection in RDM.
"""

import numpy as np

def cfar_detect(rdm, guard_cells=2, training_cells=8, rate=1e-3): #cfar_detection refer main.py
    """
    Apply Cell Averaging CFAR (CA-CFAR).
    
    Args:
        rdm (np.ndarray): Range-Doppler map
        guard_cells (int): guard cells around CUT
        training_cells (int): training cells per side
        rate (float): desired false alarm rate
    
    Returns:
        np.ndarray: binary detection map
    """
    num_doppler, num_range = rdm.shape
    detections = np.zeros_like(rdm)

    for i in range(training_cells+guard_cells, num_doppler-(training_cells+guard_cells)):
        for j in range(training_cells+guard_cells, num_range-(training_cells+guard_cells)):
            
            # Exclude guard + CUT
            training_region = rdm[
                i-(training_cells+guard_cells):i+(training_cells+guard_cells)+1,
                j-(training_cells+guard_cells):j+(training_cells+guard_cells)+1
            ]
            cut_region = rdm[
                i-guard_cells:i+guard_cells+1,
                j-guard_cells:j+guard_cells+1
            ]
            
            training_power = np.sum(training_region) - np.sum(cut_region)
            n_train = training_region.size - cut_region.size
            
            noise_level = training_power / n_train
            threshold = noise_level * (rate**(-1/n_train) - 1)
            
            if rdm[i, j] > threshold:
                detections[i, j] = 1

    return detections
