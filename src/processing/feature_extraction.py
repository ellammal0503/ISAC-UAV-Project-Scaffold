"""
Feature extraction for ML model training.
"""

import numpy as np
import numpy as np
def extract_features(detections, doa_estimates):
    features = []
    labels = []

    # Example feature extraction: concatenate detection strengths with DoA values
    for i, det in enumerate(detections):
        feat_vec = np.concatenate([det.flatten(), np.atleast_1d(doa_estimates)])
        features.append(feat_vec)
        labels.append(1 if np.any(det) else 0)

    return np.array(features), np.array(labels)

'''
def extract_features(detections, doa_estimates, scenario, noise_level=0.05):
    """
    Extract features from detections and DoA estimates.
    Adds Gaussian noise for realism.
    
    Args:
        detections: np.ndarray
        doa_estimates: np.ndarray
        scenario: list of UAV dicts (with 'los' flags)
        noise_level: float, std of Gaussian noise
    
    Returns:
        features (np.ndarray), labels (np.ndarray)
    """
    # Flatten detections
    det_features = detections.flatten()

    # Use DoA estimates as additional features
    doa_features = np.array(doa_estimates).flatten()

    # Concatenate
    features = np.concatenate([det_features, doa_features])

    # Normalize
    if np.std(features) > 0:
        features = (features - np.mean(features)) / np.std(features)

    # Add Gaussian noise
    noisy_features = features + np.random.normal(0, noise_level, size=features.shape)

    # === FIX: create labels aligned with number of samples ===
    # Use LOS flag of UAVs as labels (binary classification)
    labels = np.array([1 if uav["los"] else 0 for uav in scenario])

    # Replicate noisy_features to match number of labels
    features_out = np.tile(noisy_features, (len(labels), 1))

    return features_out, labels



#old
def extract_features(detections, doa_estimates):
    features = []
    labels = []

    # Example feature extraction: concatenate detection strengths with DoA values
    for i, det in enumerate(detections):
        feat_vec = np.concatenate([det.flatten(), np.atleast_1d(doa_estimates)])
        features.append(feat_vec)
        labels.append(1 if np.any(det) else 0)

    return np.array(features), np.array(labels)
   
#working

def extract_features(rdm, detections, doa_estimates):
    """
    Extract features for AI-based parameter estimation.
    
    Args:
        rdm (np.ndarray): Range-Doppler map
        detections (np.ndarray): binary detection map
        doa_estimates (list): estimated angles from MUSIC
    
    Returns:
        dict: extracted features (range, velocity, doa)
    """
    detected_points = np.argwhere(detections > 0)

    features = []
    for dp in detected_points:
        doppler_idx, range_idx = dp
        r_val = range_idx
        v_val = doppler_idx
        doa_val = doa_estimates
        features.append({
            "range_bin": r_val,
            "velocity_bin": v_val,
            "doa": doa_val
        })
    
    return features

'''
