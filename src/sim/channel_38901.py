"""
Channel model wrapper for ISAC UAV scenarios
based on 3GPP TR 38.901-j00 (Rel-19) §7.9.

This is a simplified placeholder:
- LOS/NLOS attenuation
- Pathloss
- Small-scale fading
"""

import numpy as np

class Channel38901:
    def __init__(self, scenario, carrier_freq=28e9):
        self.scenario = scenario
        self.fc = carrier_freq

    def pathloss(self, d, los=True):
        # Free-space pathloss (Friis)
        c = 3e8
        wavelength = c / self.fc
        fspl = 20*np.log10(4*np.pi*d / wavelength)
        if not los:
            return fspl + 20  # add penalty for NLOS
        return fspl

    def fading(self):
        # Rayleigh fading
        return np.random.rayleigh(scale=1.0)

    def apply(self, signal):
        """Apply channel effects to transmitted signal."""
        out_signals = []
        for target in self.scenario:
            d = np.linalg.norm(target["pos"])
            pl = self.pathloss(d, target["los"])
            fad = self.fading()
            attenuated = signal * (fad * 10**(-pl/20))
            out_signals.append(attenuated)
        return np.sum(out_signals, axis=0)
