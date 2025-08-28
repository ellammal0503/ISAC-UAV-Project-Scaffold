# Signal Processing Pipeline

Implemented in `src/processing/`.

## Steps

### 1. Range-Doppler Map (RDM)
- Apply matched filtering
- FFT in fast-time (range) and slow-time (Doppler)

### 2. Constant False Alarm Rate (CFAR)
- Adaptive thresholding in RDM
- Used to detect UAV echoes under clutter

### 3. DoA Estimation
- MUSIC algorithm (subspace method)
- Optionally ESPRIT
- Input: covariance matrix of array signals

### 4. Feature Extraction
- Detected peaks in (Range, Doppler, Angle)
- Form structured features for ML models/DL models
- Features used as inputs to:
    MLP
    CNN
    RNN
    Transformer

Notes
Supports multi-target scenarios with multiple UAVs
Feature dimensionality: 64 features per UAV (configurable)
Optional: visualizations via RDM and DoA plots for analysis