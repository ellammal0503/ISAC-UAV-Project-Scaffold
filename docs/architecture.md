# System Architecture

## Components
1. **Scenario Simulation (src/sim/)**
   - UAV distribution in 3D (per 3GPP Table 7.9.1-1)
   - Velocity sampling (0–180 km/h horizontal, optional vertical 20–40 km/h)
   - Channel model per 3GPP TR 38.901 §7.9

2. **Signal Processing (src/processing/)**
   - Range-Doppler Map (RDM)
   - CFAR detection
   - Direction of Arrival (DoA) estimation using MUSIC/ESPRIT
   - Feature extraction for ML models

3. **AI Models (src/models/)**
   - Baseline Deep Learning: MLP, CNN, RNN, Transformer
   - Trainer for supervised classification of LOS/NLOS


4. **Evaluation & Utilities (src/utils/)**
   - Metrics: Accuracy for classification
   - Visualization: RDM, DoA spectrum plots

## Workflow
1. Scenario generation (UAVs, channel, LOS/NLOS)
2. Echo synthesis (monostatic radar returns)
3. Signal processing (RDM, CFAR, DoA)
4. Feature extraction & ML inference (LOS/NLOS classification)
5. Performance evaluation (accuracy, plots)
