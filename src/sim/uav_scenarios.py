"""
UAV Scenario Generator based on 3GPP TR 38.901-j00 §7.9 (ISAC UAV).

Generates:
- UAV positions (2D/3D)
- UAV velocities (horizontal/vertical)
- UAV sizes (two options)
- LOS/NLOS state
"""

import numpy as np
import random
from src import config

def generate_uav_scenario(num_targets=None):
    if num_targets is None:
        num_targets = config.CONFIG["num_targets"]

    scenario = []
    for _ in range(num_targets):
        # Random height option
        height_option = random.choice(["uniform", "fixed"])
        if height_option == "uniform":
            z = np.random.uniform(1.5, 300)
        else:
            z = random.choice([25, 50, 100, 200, 300])

        # Horizontal distribution option
        x = np.random.uniform(-500, 500)
        y = np.random.uniform(-500, 500)

        # Velocity
        v_h = np.random.uniform(0, 180/3.6)   # km/h → m/s
        v_v = random.choice([0, 20/3.6, 40/3.6])  # vertical velocity m/s

        # Random size option
        size = random.choice([(1.6, 1.5, 0.7), (0.3, 0.4, 0.2)])

        # LOS/NLOS
        los = random.choice([True, False])

        target = {
            "pos": (x, y, z),
            "vel": (v_h, v_v),
            "size": size,
            "los": los,
        }
        scenario.append(target)

    return scenario
