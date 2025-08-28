"""
Echo generator for ISAC monostatic sensing.

Uses channel model to generate echoes for UAV targets.
"""

import numpy as np
from src.sim.channel_38901 import Channel38901

class EchoGenerator:
    def __init__(self, channel: Channel38901, pulse_length=128):
        self.channel = channel
        self.pulse_length = pulse_length

    def generate_pulse(self):
        """Simple baseband pulse (BPSK)"""
        return np.random.choice([-1, 1], size=self.pulse_length) + 0j

    def generate_echo(self):
        tx_signal = self.generate_pulse()
        rx_signal = self.channel.apply(tx_signal)
        return rx_signal
