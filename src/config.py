"""
Global configuration for ISAC-UAV project.
Modify values here to change default simulation/training parameters.
"""

CONFIG = {
    "num_targets": 2,
    "carrier_frequency": 28e9,   # 28 GHz mmWave
    "bandwidth": 100e6,          # 100 MHz
    "tx_power": 30,              # dBm
    "noise_figure": 5,           # dB
    "antenna_elements": 8,       # ULA elements
    "rdm_fft_size": 256,
    "cfar_threshold": 12,        # dB
    "model_checkpoint": "./checkpoints/",
    "dataset_path": "./data/"
}
