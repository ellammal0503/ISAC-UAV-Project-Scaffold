# Simulation Framework

The simulation engine (src/sim/) generates synthetic ISAC echoes under 3GPP conditions.

## Steps
1. **UAV Scenario Generation**
  - Sample N UAV targets with positions and velocities
  - Apply LOS/NLOS flags based on environment
  - Enforce minimum distance constraints per TR 36.777
  - Optionally vary height, speed, and size per scenario

2. **Channel Application**
   - Generate multipath fading, shadowing, delay spread, Doppler
   - Use CDL models from TR 38.901 §7.7/§7.9
   - Include pathloss models for UMi/UMa/RMa/SMa environments

3. **Echo Generation**
   - Compute received waveform: `r(t) = s(t) * h(t) + n(t)`
   - where s(t) is transmitted signal, h(t) is channel response, n(t) is noise
   - Apply pathloss, RCS, AWGN
   - Optionally simulate multiple antennas for array processing

4. **Output**
   - Range-Azimuth-Information (RAI) cube
   - Metadata: true range, velocity, DoA
   - Ready for downstream signal processing and AI model training

Notes
- Supports multi-target scenarios
- Configurable parameters for number of UAVs, channel type, SNR
- Visualizations for RDM and DoA spectrum available for debugging/demo